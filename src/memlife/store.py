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
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path

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
        self.db_path = config.db_path or self._resolve_db_path(config)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self.config = config
        self._init_store_attrs(config)

    @staticmethod
    def _resolve_db_path(config: MemoryConfig) -> str:
        """Resolve namespace layout when db_path is not explicitly set."""
        namespace = (config.namespace or "default").strip()
        if not namespace:
            namespace = "default"
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

    @property
    def conn(self) -> _LockedConn:
        if self._conn is None:
            with self._lock:
                if self._conn is None:
                    raw = sqlite3.connect(self.db_path, check_same_thread=False)
                    raw.row_factory = sqlite3.Row
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

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
