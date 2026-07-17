"""Memory system — SQLite-backed episodic, semantic, and reflective memory.

Four layers:

1. Working memory  — held in the agent's message list; not persisted here.
2. Episodic memory — discrete events (one agent run). Tokenised keyword recall
   plus optional vector recall over summary embeddings.
3. Semantic memory — facts/preferences with local embeddings. Vector recall
   with cosine similarity over embeddings stored as JSON in SQLite. Falls
   back to keyword search when embeddings are unavailable.
4. Reflective memory (journal) — observations/hypotheses/revisions written by
   the nightly reflection process. Private; retrieved into context, never
   quoted verbatim to the user. Supports supersession (revisions close the
   loop) and confidence decay / retirement (consolidation).

No external vector service. Embeddings are computed via an injected
``embedder`` callable and stored inline. This keeps the system to a single
SQLite file and zero extra services.
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from pathlib import Path

import sqlite3

try:
    import pysqlite3.dbapi2 as _pysqlite3  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - not installed on all interpreters
    _pysqlite3 = None  # type: ignore[misc]

from memlife._locked_conn import _LockedConn
from memlife._schema import SchemaMixin
from memlife._runs import RunMixin
from memlife._gc import GCMixin
from memlife._triples import TripleMixin
from memlife._embeddings import EmbedMixin
from memlife._episodes import EpisodeStore
from memlife._facts import FactStore
from memlife._journal import JournalStore
from memlife.protocols import Embedder
from memlife.config import MemoryConfig
from memlife.namespace import validate_namespace, list_namespaces, warn_if_cloud_sync_path
from memlife.vector_backends import create_vector_backend

logger = logging.getLogger(__name__)

# Cap on trace events stored per run (oldest are dropped beyond this).
TRACE_EVENT_LIMIT = 200

# Maximum confidence allowed for a stored fact. 1.0 is reserved: it implies
# immutable certainty, which blocks revision. Cap below 1.0 so every fact
# remains updateable.
MAX_FACT_CONFIDENCE = 0.99


class MemoryStore(SchemaMixin, RunMixin, GCMixin, TripleMixin, EmbedMixin, EpisodeStore, FactStore, JournalStore):
    """SQLite-backed memory: episodes, facts, journal, run/checkpoint bookkeeping.

    The class body intentionally contains only lifecycle plumbing. All
    domain logic lives in focused mixins under ``src/memlife/_*.py``.
    """

    def __init__(
        self,
        config: MemoryConfig | None = None,
        embedder: Embedder | None = None,
    ):
        config = config or MemoryConfig()
        config.validate()
        self.db_path = config.db_path or self._resolve_db_path(config)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        warn_if_cloud_sync_path(Path(self.db_path).parent)
        self.embedder = embedder
        self.config = config
        self._init_store_attrs(config)
        self._init_vector_backend()

    @staticmethod
    def _resolve_db_path(config: MemoryConfig) -> str:
        """Resolve namespace layout when db_path is not explicitly set."""
        namespace = validate_namespace(config.namespace or "_default")
        data_dir = Path(config.data_dir or "./memlife_data")
        return str(data_dir / namespace / "memlife.db")

    def _init_store_attrs(self, config: MemoryConfig) -> None:
        """Initialise per-instance attributes after db_path is resolved."""
        # Embedding model name from config — stored with each vector for versioning.
        self.embedding_model_name: str = config.embedding_model
        # Cosine above which two facts are treated as the same fact at store
        # time and merged (the loser superseded). Set from Config by the agent;
        # tests construct MemoryStore directly so this default is the source of
        # truth when unwired. See Config.fact_merge_threshold / fact_conflict_threshold.
        self.fact_merge_threshold: float = 0.90
        # Cosine above which two facts are treated as a candidate contradiction
        # (flagged in reflection, not auto-merged). Set from Config by the agent;
        # same default-sourcing pattern as fact_merge_threshold above.
        # MF-008: was missing — check_conflicts() raised AttributeError.
        self.fact_conflict_threshold: float = 0.75
        # Consecutive embedding failure counter — resets on success, escalates
        # logging at 5+. Exposed in embedding_health() for /stats.
        self._embed_failures: int = 0
        self._conn: _LockedConn | None = None
        # RLock: serialises all DB access from threads sharing one store.
        # Reentrant so transaction() can hold the lock across multiple
        # statements without deadlocking on individual conn.execute() calls.
        self._lock = threading.RLock()
        # MV2-006: path counters for recall diagnostics / memlife://stats.
        self._recall_counters: dict[str, int] = {
            "retrieve_calls": 0,
            "episodes_considered": 0,
            "facts_considered": 0,
            "journal_considered": 0,
            "vector_fallback_to_keyword": 0,
            "polyphonic_fusion_calls": 0,
            "voice_hits_vector": 0,
            "voice_hits_text": 0,
            "voice_hits_source": 0,
            "voice_hits_veracity": 0,
            "voice_hits_recency": 0,
        }

    def _init_vector_backend(self) -> None:
        """Create the vector backend instance for this store.

        The backend is scoped to this store/namespace/connection.  Legacy
        ``use_sqlite_vec=True`` selects sqlite_vec when ``vector_backend`` is
        left at its default "json" value.
        """
        cfg = self.config
        backend_name = cfg.resolved_vector_backend()
        backend = create_vector_backend(backend_name, self)
        if backend_name == "sqlite_vec" and not backend.available():
            logger.warning(
                "sqlite-vec requested but not available (extension missing or "
                "loading unsupported); falling back to json vector backend"
            )
            backend = create_vector_backend("json", self)
        self.vector_backend = backend

    @property
    def conn(self) -> _LockedConn:
        if self._conn is None:
            with self._lock:
                if self._conn is None:
                    raw, sqlite_mod = self._connect(self.db_path)
                    raw.row_factory = sqlite_mod.Row
                    raw.execute(
                        f"PRAGMA journal_mode={self.config.sqlite_journal_mode}"
                    )
                    raw.execute(
                        f"PRAGMA busy_timeout={self.config.sqlite_busy_timeout_ms}"
                    )
                    # Wrap in _LockedConn so every subsequent execute/commit
                    # serialises through the RLock. This makes the shared
                    # connection safe under the MCP server's thread pool.
                    self._conn = _LockedConn(raw, self._lock)
                    self._init_schema()
                    self._migrate()
        return self._conn

    @staticmethod
    def _connect(db_path: str):
        """Open a SQLite connection, preferring pysqlite3 when it can load extensions.

        The stdlib ``sqlite3`` module is sometimes compiled without
        ``ENABLE_LOAD_EXTENSION`` (e.g. manylinux wheels).  pysqlite3-binary
        ships a build that does support extensions, which lets sqlite-vec
        load.  Fall back to stdlib if pysqlite3 is unavailable or also lacks
        extension loading.

        Returns ``(connection, sqlite_module)`` so callers can use the
        matching ``Row`` factory and other module-level constants.
        """
        if _pysqlite3 is not None:
            try:
                raw = _pysqlite3.connect(db_path, check_same_thread=False)
                if hasattr(raw, "enable_load_extension") and hasattr(raw, "load_extension"):
                    return raw, _pysqlite3
                raw.close()
            except Exception:  # pragma: no cover
                pass
        return sqlite3.connect(db_path, check_same_thread=False), sqlite3

    @contextmanager
    def transaction(self):
        """Context manager for multi-statement atomicity.

        Holds the RLock across the entire block so a sequence of
        execute/commit calls runs atomically with respect to other
        threads. The lock is reentrant, so individual conn.execute()
        calls inside the block don't deadlock.
        """
        with self._lock:
            yield self.conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # MF-016: context manager support for clean resource handling.
    def __enter__(self) -> "MemoryStore":
        return self

    @classmethod
    def list_namespaces(cls, data_dir: str | Path) -> list[str]:
        """Return the names of existing namespace directories under data_dir."""
        return list_namespaces(data_dir)

    def switch_namespace(self, new_namespace: str) -> "MemoryStore":
        """Return a new store instance for a different namespace.

        The new store reuses the current embedder and the same config class,
        but points at the new namespace's SQLite file.
        """
        validate_namespace(new_namespace)
        new_config = self.config.__class__(**self.config.__dict__)
        new_config.namespace = new_namespace
        return self.__class__(config=new_config, embedder=self.embedder)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
