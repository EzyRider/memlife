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

import json
import logging
import re
import sqlite3
import threading
import time
import uuid
from pathlib import Path

# An embedder turns a list of texts into a list of float vectors.
# It is optional: when absent, semantic/episodic recall degrades to keyword.

logger = logging.getLogger(__name__)

# Cap on trace events stored per run (oldest are dropped beyond this).
TRACE_EVENT_LIMIT = 200

# Maximum confidence allowed for a stored fact. 1.0 is reserved: it implies
# immutable certainty, which blocks revision. Cap below 1.0 so every fact
# remains updateable.
MAX_FACT_CONFIDENCE = 0.99


from memlife.models import Episode, Fact, JournalEntry
from memlife.vectors import cosine, recency_weight
from memlife.protocols import Embedder
from memlife.config import MemoryConfig
# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------
class MemoryStore:
    """SQLite-backed memory: episodes, facts, journal, run/checkpoint bookkeeping."""

    def __init__(
        self,
        config: MemoryConfig | None = None,
        embedder: Embedder | None = None,
    ):
        config = config or MemoryConfig()
        Path(config.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = config.db_path
        self.embedder = embedder
        self.config = config
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
        self._conn: sqlite3.Connection | None = None
        # Serialises connection creation + migration, and (with check_same_thread
        # off) all DB access from threads sharing one store. The async API can
        # dispatch on multiple threads, so the connection is shared-but-locked.
        self._lock = threading.Lock()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            with self._lock:
                if self._conn is None:
                    self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
                    self._conn.row_factory = sqlite3.Row
                    self._conn.execute(
                        f"PRAGMA journal_mode={self.config.sqlite_journal_mode}"
                    )
                    self._conn.execute(
                        f"PRAGMA busy_timeout={self.config.sqlite_busy_timeout_ms}"
                    )
                    self._init_schema()
                    self._migrate()
        return self._conn

    def _init_schema(self) -> None:
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS episodes (
                id TEXT PRIMARY KEY, task TEXT NOT NULL,
                outcome TEXT NOT NULL DEFAULT 'running',
                summary TEXT DEFAULT '', tool_calls_json TEXT DEFAULT '[]',
                created_at REAL NOT NULL,
                embedding_json TEXT DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS agent_runs (
                id TEXT PRIMARY KEY, task TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'running',
                created_at REAL NOT NULL, completed_at REAL,
                model_used TEXT, total_tokens INTEGER DEFAULT 0,
                error_message TEXT,
                trace_json TEXT DEFAULT '[]'
            );
            CREATE TABLE IF NOT EXISTS checkpoints (
                id TEXT PRIMARY KEY, run_id TEXT NOT NULL,
                step_index INTEGER NOT NULL, step_description TEXT,
                state_json TEXT NOT NULL, tool_calls_json TEXT DEFAULT '[]',
                observation TEXT, outcome TEXT,
                tokens_used INTEGER DEFAULT 0, created_at REAL NOT NULL,
                UNIQUE(run_id, step_index)
            );
            CREATE INDEX IF NOT EXISTS idx_cp_run
                ON checkpoints(run_id, step_index);
            CREATE TABLE IF NOT EXISTS facts (
                id TEXT PRIMARY KEY, content TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'agent',
                confidence REAL DEFAULT 0.5,
                embedding_json TEXT DEFAULT '',
                created_at REAL NOT NULL, updated_at REAL NOT NULL,
                superseded_by TEXT DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS idx_facts_content
                ON facts(content);
            CREATE TABLE IF NOT EXISTS journal (
                id TEXT PRIMARY KEY, type TEXT NOT NULL,
                content TEXT NOT NULL, confidence REAL DEFAULT 0.5,
                source_episodes_json TEXT DEFAULT '[]',
                private INTEGER DEFAULT 1,
                created_at REAL NOT NULL,
                superseded_by TEXT DEFAULT '',
                embedding_json TEXT DEFAULT '',
                last_detected INTEGER DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_journal_type
                ON journal(type);
            CREATE INDEX IF NOT EXISTS idx_journal_created
                ON journal(created_at);
            CREATE TABLE IF NOT EXISTS reflection_queue (
                id TEXT PRIMARY KEY, episode_id TEXT NOT NULL,
                queued_at REAL NOT NULL, reflected INTEGER DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY, name TEXT NOT NULL,
                created_at REAL NOT NULL, updated_at REAL NOT NULL,
                model_used TEXT DEFAULT '',
                conversation_json TEXT DEFAULT '[]',
                rolling_summary TEXT DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS idx_sessions_updated
                ON sessions(updated_at);
            CREATE TABLE IF NOT EXISTS reflection_metrics (
                id TEXT PRIMARY KEY,
                created_at REAL NOT NULL,
                episodes_considered INTEGER DEFAULT 0,
                observations_proposed INTEGER DEFAULT 0,
                observations_kept INTEGER DEFAULT 0,
                hypotheses_proposed INTEGER DEFAULT 0,
                hypotheses_kept INTEGER DEFAULT 0,
                revisions_proposed INTEGER DEFAULT 0,
                revisions_kept INTEGER DEFAULT 0,
                contradictions_found INTEGER DEFAULT 0,
                avg_confidence REAL DEFAULT 0.0,
                keep_rate REAL DEFAULT 0.0,
                consolidated_retired INTEGER DEFAULT 0,
                consolidated_merged INTEGER DEFAULT 0,
                total_journal_entries INTEGER DEFAULT 0,
                total_facts INTEGER DEFAULT 0,
                total_episodes INTEGER DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS episode_tools (
                episode_id TEXT NOT NULL,
                tool_name TEXT NOT NULL,
                created_at REAL NOT NULL,
                PRIMARY KEY (episode_id, tool_name)
            );
            CREATE INDEX IF NOT EXISTS idx_episode_tools_name
                ON episode_tools(tool_name);
        """)
        self.conn.commit()

    def _migrate(self) -> None:
        """Add columns/indexes introduced after the initial schema, for existing DBs.

        Each step is idempotent and resilient: indexes use ``IF NOT EXISTS``, and
        the reflection-queue unique index is preceded by a dedup of any
        pre-existing duplicate ``episode_id`` rows (the old code allowed dupes).
        """
        cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(episodes)")}
        if "embedding_json" not in cols:
            self.conn.execute("ALTER TABLE episodes ADD COLUMN embedding_json TEXT DEFAULT ''")
        jcols = {r["name"] for r in self.conn.execute("PRAGMA table_info(journal)")}
        if "superseded_by" not in jcols:
            self.conn.execute("ALTER TABLE journal ADD COLUMN superseded_by TEXT DEFAULT ''")
        if "embedding_json" not in jcols:
            self.conn.execute("ALTER TABLE journal ADD COLUMN embedding_json TEXT DEFAULT ''")
        if "last_detected" not in jcols:
            self.conn.execute("ALTER TABLE journal ADD COLUMN last_detected INTEGER DEFAULT 0")
        rcols = {r["name"] for r in self.conn.execute("PRAGMA table_info(agent_runs)")}
        if "trace_json" not in rcols:
            self.conn.execute("ALTER TABLE agent_runs ADD COLUMN trace_json TEXT DEFAULT '[]'")

        scols = {r["name"] for r in self.conn.execute("PRAGMA table_info(sessions)")}
        if "rolling_summary" not in scols:
            self.conn.execute("ALTER TABLE sessions ADD COLUMN rolling_summary TEXT DEFAULT ''")

        # Dedup reflection_queue before adding the unique index, so an existing
        # DB with duplicate episode_id rows (from pre-unique behaviour) can still
        # migrate. Keep the earliest-queued row per episode_id.
        self.conn.execute(
            "DELETE FROM reflection_queue WHERE id NOT IN ("
            "  SELECT id FROM reflection_queue q1 WHERE queued_at = ("
            "    SELECT MIN(queued_at) FROM reflection_queue q2 "
            "    WHERE q2.episode_id = q1.episode_id))"
        )
        self.conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_rqueue_episode "
            "ON reflection_queue(episode_id)"
        )

        # Partial indexes so vector-recall's ``embedding_json != ''`` filter
        # doesn't full-scan the whole table once embeddings are common.
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_episodes_embedded "
            "ON episodes(embedding_json) WHERE embedding_json != ''"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_facts_embedded "
            "ON facts(embedding_json) WHERE embedding_json != ''"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_journal_embedded "
            "ON journal(embedding_json) WHERE embedding_json != ''"
        )

        # Cap any pre-existing facts that were stored at immutable confidence
        # (1.0) before the MAX_FACT_CONFIDENCE rule existed.
        self.conn.execute(
            "UPDATE facts SET confidence = ? WHERE confidence > ? AND superseded_by = ''",
            (MAX_FACT_CONFIDENCE, MAX_FACT_CONFIDENCE),
        )

        # Embedding model versioning: add embedding_model column to all
        # three tables so we can detect when the model changes and trigger
        # a backfill. Existing rows get '' (unknown — treated as "needs
        # backfill" if a model name is configured).
        for table in ("facts", "journal", "episodes"):
            cols = {r["name"] for r in self.conn.execute(f"PRAGMA table_info({table})")}
            if "embedding_model" not in cols:
                self.conn.execute(
                    f"ALTER TABLE {table} ADD COLUMN embedding_model TEXT DEFAULT ''"
                )

        self.conn.commit()

        # Backfill episode_tools index for existing episodes that predate
        # the index table. Parse tool_calls_json and populate the index.
        indexed_count = self.conn.execute(
            "SELECT COUNT(*) FROM episode_tools"
        ).fetchone()[0]
        if indexed_count == 0:
            rows = self.conn.execute(
                "SELECT id, tool_calls_json, created_at FROM episodes "
                "WHERE tool_calls_json != '[]'"
            ).fetchall()
            for row in rows:
                ep_id, tc_json, created = row[0], row[1], row[2]
                try:
                    calls = json.loads(tc_json) if tc_json else []
                except (json.JSONDecodeError, TypeError):
                    continue
                seen: set[str] = set()
                for tc in calls:
                    name = tc.get("tool") if isinstance(tc, dict) else None
                    if name and name not in seen:
                        seen.add(name)
                        self.conn.execute(
                            "INSERT OR IGNORE INTO episode_tools "
                            "(episode_id, tool_name, created_at) VALUES (?, ?, ?)",
                            (ep_id, name, created),
                        )
            self.conn.commit()

    # ------------------------------------------------------------------
    # Embedding helper
    # ------------------------------------------------------------------
    async def embed_texts(self, texts: list[str]) -> list[list[float]] | None:
        """Best-effort embedding; returns None if no embedder or it fails.

        Failures are logged at WARNING (not DEBUG) so silent degradation to
        keyword recall is visible to operators — an absent or misconfigured
        embedder otherwise looks like the system "just works" on keywords.
        """
        if self.embedder is None or not texts:
            return None
        try:
            result = await self.embedder.embed(texts)
            if result is not None:
                self._embed_failures = 0  # reset on success
            return result
        except Exception as exc:  # noqa: BLE001
            self._embed_failures += 1
            logger.warning(
                "embed_texts failed (attempt #%d, falling back to keyword recall): %s",
                self._embed_failures, exc,
            )
            if self._embed_failures >= 5:
                logger.error(
                    "embed_texts has failed %d consecutive times — semantic "
                    "recall is degraded to keyword-only. Check the embedder "
                    "and Ollama availability.",
                    self._embed_failures,
                )
            return None

    # ------------------------------------------------------------------
    # Episodes
    # ------------------------------------------------------------------
    def remember(
        self, task: str, outcome: str, summary: str = "",
        tool_calls: list[dict] | None = None,
    ) -> str:
        """Store an episode. Returns the episode ID. (Sync; embedding is a
        separate async step via :meth:`embed_episode`.)

        Tool call names are indexed in ``episode_tools`` for fast
        ``search_episodes_by_tool`` queries.
        """
        ep_id = f"ep_{uuid.uuid4().hex[:12]}"
        now = time.time()
        self.conn.execute(
            "INSERT INTO episodes (id, task, outcome, summary, "
            "tool_calls_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (ep_id, task, outcome, summary,
             json.dumps(tool_calls or []), now),
        )
        # Index tool names for episodic search by behaviour.
        if tool_calls:
            seen_tools: set[str] = set()
            for tc in tool_calls:
                name = tc.get("tool") if isinstance(tc, dict) else None
                if name and name not in seen_tools:
                    seen_tools.add(name)
                    self.conn.execute(
                        "INSERT OR IGNORE INTO episode_tools "
                        "(episode_id, tool_name, created_at) VALUES (?, ?, ?)",
                        (ep_id, name, now),
                    )
        self.conn.commit()
        return ep_id

    async def embed_episode(self, ep_id: str) -> None:
        """Compute and store an embedding for an episode's task+summary."""
        row = self.conn.execute(
            "SELECT task, summary FROM episodes WHERE id = ?", (ep_id,)
        ).fetchone()
        if not row:
            return
        text = f"{row[0]}\n{row[1] or ''}".strip()
        if not text:
            return
        vecs = await self.embed_texts([text])
        if vecs:
            self.conn.execute(
                "UPDATE episodes SET embedding_json = ?, embedding_model = ? WHERE id = ?",
                (json.dumps(vecs[0]), self.embedding_model_name, ep_id),
            )
            self.conn.commit()

    def _tokenize(self, query: str) -> list[str]:
        return [t for t in re.findall(r"[A-Za-z0-9_]+", query.lower()) if len(t) >= 3]

    def recall(self, query: str, limit: int = 5) -> list[Episode]:
        """Tokenised keyword recall over task and summary, ranked by match count
        × recency. Falls back to recent() if the query has no usable tokens.
        """
        tokens = self._tokenize(query)
        if not tokens:
            return self.recent(limit=limit)
        clauses = []
        params: list = []
        for t in tokens:
            clauses.append("(task LIKE ? OR summary LIKE ?)")
            params += [f"%{t}%", f"%{t}%"]
        where = " OR ".join(clauses)
        rows = self.conn.execute(
            f"SELECT id, task, outcome, summary, tool_calls_json, created_at, "
            f"embedding_json FROM episodes WHERE ({where}) "
            f"ORDER BY created_at DESC LIMIT ?",
            (*params, limit * 4),
        ).fetchall()
        eps = [Episode.from_row(tuple(r)) for r in rows]
        # Rank by normalised relevance × recency. _relevance ∈ [0,1] so the
        # agent's cross-layer ranker (relevance × confidence × recency) is on a
        # common scale with facts/journal.
        n_tokens = max(1, len(tokens))
        for ep in eps:
            hay = (ep.task + " " + ep.summary).lower()
            match_count = sum(1 for t in tokens if t in hay)
            ep._relevance = match_count / n_tokens  # type: ignore[attr-defined]
            ep._score = ep._relevance * recency_weight(ep.created_at)  # type: ignore[attr-defined]
        eps.sort(key=lambda e: getattr(e, "_score", 0), reverse=True)
        return eps[:limit]

    async def recall_episodes_vector(
        self, query_vector: list[float], limit: int = 5,
    ) -> list[Episode]:
        """Vector recall over episodes that have embeddings, scored by
        cosine × recency_weight."""
        t0 = time.time()
        rows = self.conn.execute(
            "SELECT id, task, outcome, summary, tool_calls_json, created_at, "
            "embedding_json FROM episodes WHERE embedding_json != '' "
            "ORDER BY created_at DESC LIMIT 500"
        ).fetchall()
        eps = [Episode.from_row(tuple(r)) for r in rows]
        scored = []
        for ep in eps:
            v = ep.embedding
            if v is None:
                continue
            sim = cosine(query_vector, v)
            rw = recency_weight(ep.created_at)
            # Attach the normalised relevance so the agent's cross-layer ranker
            # (relevance × confidence × recency) actually includes cosine —
            # previously this was computed only to sort and then discarded.
            ep._relevance = sim  # type: ignore[attr-defined]
            ep._score = sim * rw  # type: ignore[attr-defined]
            scored.append((sim * rw, ep))
        scored.sort(key=lambda t: t[0], reverse=True)
        dt = (time.time() - t0) * 1000
        if dt > 50:
            import logging
            logging.getLogger(__name__).info(
                "episode vector recall: %.0fms for %d episodes (%d with embeddings)",
                dt, len(rows), len(scored),
            )
        return [ep for s, ep in scored[:limit] if s > 0.0]

    def recent(self, limit: int = 10) -> list[Episode]:
        """Return most recent episodes."""
        rows = self.conn.execute(
            "SELECT id, task, outcome, summary, tool_calls_json, created_at, "
            "embedding_json FROM episodes ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [Episode.from_row(tuple(r)) for r in rows]

    def episodes_since(self, since: float, limit: int = 200) -> list[Episode]:
        """Episodes created at/after a timestamp (for nightly reflection)."""
        rows = self.conn.execute(
            "SELECT id, task, outcome, summary, tool_calls_json, created_at, "
            "embedding_json FROM episodes WHERE created_at >= ? "
            "ORDER BY created_at DESC LIMIT ?",
            (since, limit),
        ).fetchall()
        return [Episode.from_row(tuple(r)) for r in rows]

    def episodes_sample_historical(
        self, before: float, limit: int = 15, bins: int = 5,
    ) -> list[Episode]:
        """Sample representative episodes from before ``before`` (epoch),
        spread across time so the reflector sees long-term patterns, not
        just a contiguous block.

        Divides the time range from the oldest episode to ``before`` into
        ``bins`` equal segments and takes ``limit // bins`` episodes from
        each segment (most recent in each). This gives the model a view
        that spans days or weeks rather than just the last N episodes.
        """
        oldest = self.conn.execute(
            "SELECT MIN(created_at) FROM episodes"
        ).fetchone()[0]
        if not oldest or oldest >= before:
            return []
        per_bin = max(1, limit // bins)
        results: list[Episode] = []
        seg_size = (before - oldest) / bins
        for i in range(bins):
            seg_start = oldest + (seg_size * i)
            seg_end = oldest + (seg_size * (i + 1))
            rows = self.conn.execute(
                "SELECT id, task, outcome, summary, tool_calls_json, "
                "created_at, embedding_json FROM episodes "
                "WHERE created_at >= ? AND created_at < ? "
                "ORDER BY created_at DESC LIMIT ?",
                (seg_start, seg_end, per_bin),
            ).fetchall()
            results.extend(Episode.from_row(tuple(r)) for r in rows)
        return results[:limit]

    def journal_thematic_summary(self, limit: int = 10) -> list[JournalEntry]:
        """Return a spread of older journal entries (observations and
        hypotheses only, no contradictions) for the reflection prompt's
        historical context. Samples across time rather than just the most
        recent entries."""
        # _active_journal_sql already appends ORDER BY created_at DESC,
        # so we just add the LIMIT via parameter.
        rows = self.conn.execute(
            self._active_journal_sql(
                "id, type, content, confidence, source_episodes_json, "
                "private, created_at, superseded_by, embedding_json",
                "AND type IN ('observation', 'hypothesis')"
            ) + " LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._journal_from_row(r) for r in rows]

    def episodes_by_ids(self, ids: list[str]) -> list[Episode]:
        """Fetch specific episodes by id (preserving no particular order)."""
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        rows = self.conn.execute(
            f"SELECT id, task, outcome, summary, tool_calls_json, created_at, "
            f"embedding_json FROM episodes WHERE id IN ({placeholders})",
            tuple(ids),
        ).fetchall()
        by_id = {r[0]: Episode.from_row(tuple(r)) for r in rows}
        return [by_id[i] for i in ids if i in by_id]

    def search_episodes_by_tool(
        self, tool_name: str, outcome: str | None = None, limit: int = 20,
    ) -> list[Episode]:
        """Find episodes where a specific tool was used. Optionally filter
        by outcome ('success', 'failed', etc.). Uses the episode_tools
        index for fast lookups without parsing JSON blobs."""
        sql = (
            "SELECT e.id, e.task, e.outcome, e.summary, "
            "e.tool_calls_json, e.created_at, e.embedding_json "
            "FROM episodes e "
            "JOIN episode_tools et ON et.episode_id = e.id "
            "WHERE et.tool_name = ?"
        )
        params: list = [tool_name]
        if outcome:
            sql += " AND e.outcome = ?"
            params.append(outcome)
        sql += " ORDER BY e.created_at DESC LIMIT ?"
        params.append(limit)
        rows = self.conn.execute(sql, params).fetchall()
        return [Episode.from_row(tuple(r)) for r in rows]

    # ------------------------------------------------------------------
    # Semantic memory — facts
    # ------------------------------------------------------------------
    async def store_fact(
        self, content: str, source: str = "agent",
        confidence: float = 0.5, embed: bool = True,
    ) -> str:
        """Store a fact, optionally embedding it. Returns the fact ID.

        Confidence is clamped to ``MAX_FACT_CONFIDENCE`` so no fact is ever
        stored as immutable (1.0).

        Dedup is layered (later layers only run when the earlier miss):
          1. Exact-normalised content (case/whitespace/punctuation-tail) → same
             fact; return the existing ID.
          2. Containment — one content is a substring of the other → keep the
             longer (more specific) wording, bump its updated_at, don't store
             the shorter redundant one.
          3. Semantic — cosine ≥ ``fact_merge_threshold`` over stored embeddings
             → the same fact reworded, or a correction. Merge by superseding the
             lower-confidence one: the higher-confidence content wins, and on a
             tie the newer content wins (so an equal-confidence correction takes
             effect instead of leaving the stale fact active alongside it).

        With no embedder, only layers 1–2 run (rephrased near-duplicates are
        stored, as before — degrades, doesn't break).
        """
        confidence = min(float(confidence), MAX_FACT_CONFIDENCE)

        norm = self._normalize(content)
        new_lower = content.strip().lower()

        # Layer 1: exact-normalised dedup (cheap; skips the embedding call).
        for fid, raw in self.conn.execute(
            "SELECT id, content FROM facts WHERE superseded_by = ''"
        ).fetchall():
            if self._normalize(raw) == norm:
                return fid

        # Embed once; reused for both the cosine merge and the insert.
        new_vec = None
        if embed:
            vecs = await self.embed_texts([content])
            if vecs:
                new_vec = vecs[0]
        embedding_json = json.dumps(new_vec) if new_vec else ""

        active = self._active_facts()

        # Layer 2: containment — keep the longer (more specific) content.
        for f in active:
            ex = f.content.strip().lower()
            if not ex or ex == new_lower:
                continue
            if ex in new_lower or new_lower in ex:
                if len(f.content) >= len(content):
                    self._refresh_fact(f.id, confidence)
                    return f.id
                return await self._supersede_fact(
                    f.id, content, source, confidence, embedding_json)

        # Layer 3: semantic merge over stored embeddings.
        if new_vec is not None:
            best: Fact | None = None
            best_sim = 0.0
            for f in active:
                if f.embedding is None:
                    continue
                sim = cosine(new_vec, f.embedding)
                if sim > best_sim:
                    best, best_sim = f, sim
            if best is not None and best_sim >= self.fact_merge_threshold:
                # Higher confidence wins; tie → newer (new) wins.
                if confidence >= best.confidence:
                    return await self._supersede_fact(
                        best.id, content, source, confidence, embedding_json)
                # Existing wins: refresh it, drop the weaker duplicate.
                self._refresh_fact(best.id, confidence)
                return best.id

        # No merge: insert as a new fact.
        fact_id = f"fact_{uuid.uuid4().hex[:12]}"
        now = time.time()
        self.conn.execute(
            "INSERT INTO facts (id, content, source, confidence, "
            "embedding_json, embedding_model, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (fact_id, content, source, confidence, embedding_json,
             self.embedding_model_name if new_vec else "", now, now),
        )
        self.conn.commit()
        return fact_id

    def _refresh_fact(self, fact_id: str, extra_confidence: float) -> None:
        """Bump a kept fact's updated_at and raise its confidence to the max —
        used when a near-duplicate is folded into an existing fact.
        Confidence is capped at ``MAX_FACT_CONFIDENCE``."""
        now = time.time()
        self.conn.execute(
            "UPDATE facts SET updated_at = ?, "
            "confidence = MIN(MAX(confidence, ?), ?) WHERE id = ?",
            (now, extra_confidence, MAX_FACT_CONFIDENCE, fact_id),
        )
        self.conn.commit()

    def adjust_confidence(self, fact_id: str, delta: float) -> None:
        """Adjust a fact's confidence by delta, clamped to [0.0, MAX_FACT_CONFIDENCE]."""
        self.conn.execute(
            "UPDATE facts SET confidence = MAX(0.0, MIN(?, confidence + ?)), "
            "updated_at = ? WHERE id = ?",
            (MAX_FACT_CONFIDENCE, delta, time.time(), fact_id),
        )
        self.conn.commit()

    async def _supersede_fact(
        self, old_id: str, new_content: str, source: str,
        confidence: float, embedding_json: str,
    ) -> str:
        """Atomically store a new fact and mark ``old_id`` superseded by it
        (the revise_fact SAVEPOINT pattern). Returns the new fact id.
        Incoming confidence is capped at ``MAX_FACT_CONFIDENCE``."""
        new_id = f"fact_{uuid.uuid4().hex[:12]}"
        now = time.time()
        capped = min(float(confidence), MAX_FACT_CONFIDENCE)
        try:
            self.conn.execute("SAVEPOINT supersede_fact")
            self.conn.execute(
                "INSERT INTO facts (id, content, source, confidence, "
                "embedding_json, embedding_model, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (new_id, new_content, source, capped, embedding_json,
                 self.embedding_model_name if embedding_json else "", now, now),
            )
            self.conn.execute(
                "UPDATE facts SET superseded_by = ?, updated_at = ? WHERE id = ?",
                (new_id, now, old_id),
            )
            self.conn.execute("RELEASE SAVEPOINT supersede_fact")
        except Exception:
            self.conn.execute("ROLLBACK TO SAVEPOINT supersede_fact")
            self.conn.execute("RELEASE SAVEPOINT supersede_fact")
            raise
        return new_id

    def _active_facts(self) -> list[Fact]:
        """All non-superseded facts (with or without embeddings)."""
        rows = self.conn.execute(
            "SELECT id, content, source, confidence, embedding_json, "
            "created_at, updated_at, superseded_by FROM facts "
            "WHERE superseded_by = ''"
        ).fetchall()
        return [Fact(
            id=r[0], content=r[1], source=r[2], confidence=r[3],
            embedding_json=r[4] or "", created_at=r[5], updated_at=r[6],
            superseded_by=r[7] or "",
        ) for r in rows]

    def fact_by_id(self, fact_id: str) -> Fact | None:
        """Fetch a single fact by id, regardless of supersession status."""
        row = self.conn.execute(
            "SELECT id, content, source, confidence, embedding_json, "
            "created_at, updated_at, superseded_by FROM facts WHERE id = ?",
            (fact_id,),
        ).fetchone()
        if not row:
            return None
        return Fact(
            id=row[0], content=row[1], source=row[2], confidence=row[3],
            embedding_json=row[4] or "", created_at=row[5], updated_at=row[6],
            superseded_by=row[7] or "",
        )

    def resolve_fact(self, fact_id: str) -> Fact | None:
        """Resolve a fact id to the currently-active version.

        If the fact has been superseded by another fact, follow the chain
        until an active (non-superseded) fact is found. Returns None if the
        fact is missing, retired, or the chain ends in a missing/retired fact.
        """
        seen: set[str] = set()
        current_id = fact_id
        while current_id and current_id not in seen:
            seen.add(current_id)
            fact = self.fact_by_id(current_id)
            if fact is None:
                return None
            if not fact.superseded_by:
                return fact
            if fact.retired:
                return None
            current_id = fact.superseded_by
        return None

    def list_contradictions(self, limit: int = 20) -> list[dict]:
        """Return stored contradiction journal entries resolved to current facts.

        Each item contains the contradiction journal id, when it was detected,
        and the two conflicting facts resolved to their currently-active
        versions (following supersession chains). If a fact was retired or the
        chain cannot be followed, the original fact id and content are
        returned marked as missing/retired so the contradiction stays visible.
        """
        rows = self.conn.execute(
            "SELECT id, content, confidence, source_episodes_json, created_at "
            "FROM journal WHERE type = 'contradiction' AND superseded_by = '' "
            "ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        results: list[dict] = []
        for row in rows:
            jid, content, conf, source_json, created_at = row
            fact_ids: list[str] = []
            try:
                fact_ids = json.loads(source_json or "[]")
            except json.JSONDecodeError:
                pass
            facts: list[dict] = []
            for fid in fact_ids[:2]:
                resolved = self.resolve_fact(fid)
                if resolved:
                    facts.append({
                        "id": resolved.id,
                        "content": resolved.content,
                        "confidence": resolved.confidence,
                        "active": True,
                    })
                else:
                    original = self.fact_by_id(fid)
                    facts.append({
                        "id": fid,
                        "content": original.content if original else "(unknown)",
                        "confidence": original.confidence if original else 0.0,
                        "active": False,
                        "missing": original is None,
                        "retired": bool(original and original.retired),
                    })
            # A contradiction is considered unresolved if both sides still
            # resolve to active, distinct facts.
            unresolved = (
                len(facts) == 2
                and facts[0]["active"]
                and facts[1]["active"]
                and facts[0]["id"] != facts[1]["id"]
            )
            results.append({
                "id": jid,
                "content": content,
                "confidence": conf,
                "created_at": created_at,
                "facts": facts,
                "unresolved": unresolved,
            })
        return results

    @staticmethod
    def _normalize(content: str) -> str:
        """Normalise a fact for dedup: collapse whitespace, lowercase, strip punctuation tails."""
        return re.sub(r"\s+", " ", content.strip().lower()).rstrip(".")

    async def check_conflicts(self, content: str) -> list[dict]:
        """Check for existing active facts that semantically overlap with new content.

        Returns a list of dicts with id, content, similarity, confidence for
        each candidate conflict (cosine >= fact_conflict_threshold). Uses
        embeddings when available, falls back to keyword overlap otherwise.
        Embedding failures are logged by embed_texts() — not silently swallowed.
        """
        conflicts: list[dict] = []
        new_vec = None
        # embed_texts handles failure logging internally — no silent swallow.
        vecs = await self.embed_texts([content])
        if vecs:
            new_vec = vecs[0]

        active = self._active_facts()

        if new_vec is not None:
            for f in active:
                if f.embedding is None:
                    continue
                sim = cosine(new_vec, f.embedding)
                if sim >= self.fact_conflict_threshold:
                    conflicts.append({
                        "id": f.id,
                        "content": f.content,
                        "similarity": round(sim, 3),
                        "confidence": f.confidence,
                    })
            conflicts.sort(key=lambda c: c["similarity"], reverse=True)
            return conflicts[:5]

        # Keyword fallback — token overlap, same as recall_facts uses.
        tokens = self._tokenize(content)
        if not tokens:
            return []
        for f in active:
            hay = f.content.lower()
            match_count = sum(1 for t in tokens if t in hay)
            if match_count == 0:
                continue
            score = match_count / max(1, len(tokens))
            if score >= 0.5:
                conflicts.append({
                    "id": f.id,
                    "content": f.content,
                    "similarity": round(score, 3),
                    "confidence": f.confidence,
                })
        conflicts.sort(key=lambda c: c["similarity"], reverse=True)
        return conflicts[:5]

    def expire_fact(self, fact_id: str, reason: str = "expired") -> bool:
        """Mark a fact as expired/superseded without a replacement fact.

        Uses the ``__retired__:<reason>`` sentinel, same pattern as
        retire_journal. Returns True if a row was updated, False if the
        fact was missing or already superseded.
        """
        cur = self.conn.execute(
            "UPDATE facts SET superseded_by = ?, updated_at = ? "
            "WHERE id = ? AND superseded_by = ''",
            (f"__retired__:{reason}", time.time(), fact_id),
        )
        self.conn.commit()
        return cur.rowcount > 0

    async def recall_facts(
        self, query: str, limit: int = 5, query_vector: list[float] | None = None,
    ) -> list[Fact]:
        """Vector recall over facts when embeddings are available, scored by
        cosine × confidence × recency; keyword fallback with same weighting."""
        facts_with_vec = self._facts_with_embeddings()
        if query_vector is None and self.embedder is not None and query.strip():
            query_vector = (await self.embed_texts([query]) or [None])[0]  # type: ignore[arg-type]
        if query_vector is not None and facts_with_vec:
            scored = []
            for f in facts_with_vec:
                if f.embedding is None:
                    continue
                sim = cosine(query_vector, f.embedding)
                f._relevance = sim  # type: ignore[attr-defined]
                f._score = sim * f.confidence * recency_weight(f.updated_at)  # type: ignore[attr-defined]
                scored.append((f._score, f))
            scored.sort(key=lambda t: t[0], reverse=True)
            top = [f for s, f in scored[:limit] if s > 0.0]
            if top:
                return top
        # Keyword fallback — tokenised LIKE, scored by confidence × recency.
        tokens = self._tokenize(query)
        if not tokens:
            return []
        clauses = " OR ".join("content LIKE ?" for _ in tokens)
        params = [f"%{t}%" for t in tokens]
        rows = self.conn.execute(
            f"SELECT id, content, source, confidence, embedding_json, "
            f"created_at, updated_at, superseded_by FROM facts "
            f"WHERE ({clauses}) AND superseded_by = '' "
            f"ORDER BY updated_at DESC LIMIT ?",
            (*params, limit * 3),
        ).fetchall()
        facts = [Fact(
            id=r[0], content=r[1], source=r[2], confidence=r[3],
            embedding_json=r[4] or "", created_at=r[5], updated_at=r[6],
            superseded_by=r[7] or "",
        ) for r in rows]
        # Score by normalised relevance × confidence × recency.
        n_tokens = max(1, len(tokens))
        for f in facts:
            match = sum(1 for t in tokens if t in f.content.lower())
            f._relevance = match / n_tokens  # type: ignore[attr-defined]
            f._score = f._relevance * f.confidence * recency_weight(f.updated_at)  # type: ignore[attr-defined]
        facts.sort(key=lambda f: getattr(f, "_score", 0), reverse=True)
        return facts[:limit]

    def _facts_with_embeddings(self) -> list[Fact]:
        rows = self.conn.execute(
            "SELECT id, content, source, confidence, embedding_json, "
            "created_at, updated_at, superseded_by FROM facts "
            "WHERE embedding_json != '' AND superseded_by = ''"
        ).fetchall()
        return [Fact(
            id=r[0], content=r[1], source=r[2], confidence=r[3],
            embedding_json=r[4] or "", created_at=r[5], updated_at=r[6],
            superseded_by=r[7] or "",
        ) for r in rows]

    def _facts_with_embeddings_since(self, since: float) -> list[Fact]:
        """Active facts with embeddings that were created or updated since
        ``since``. Used by incremental contradiction detection so we only
        compare new/changed facts against the full set, not all pairs."""
        rows = self.conn.execute(
            "SELECT id, content, source, confidence, embedding_json, "
            "created_at, updated_at, superseded_by FROM facts "
            "WHERE embedding_json != '' AND superseded_by = '' "
            "AND (created_at >= ? OR updated_at >= ?)",
            (since, since),
        ).fetchall()
        return [Fact(
            id=r[0], content=r[1], source=r[2], confidence=r[3],
            embedding_json=r[4] or "", created_at=r[5], updated_at=r[6],
            superseded_by=r[7] or "",
        ) for r in rows]

    def _active_facts_since(self, since: float) -> list[Fact]:
        """Active facts (with or without embeddings) created or updated since
        ``since``. Lexical fallback path for incremental contradiction detection."""
        rows = self.conn.execute(
            "SELECT id, content, source, confidence, embedding_json, "
            "created_at, updated_at, superseded_by FROM facts "
            "WHERE superseded_by = '' AND (created_at >= ? OR updated_at >= ?)",
            (since, since),
        ).fetchall()
        return [Fact(
            id=r[0], content=r[1], source=r[2], confidence=r[3],
            embedding_json=r[4] or "", created_at=r[5], updated_at=r[6],
            superseded_by=r[7] or "",
        ) for r in rows]

    async def revise_fact(self, fact_id: str, new_content: str, confidence: float = 0.7) -> str:
        """Mark an old fact superseded and store a revised one with a fresh embedding.

        The revised fact gets its own ``created_at`` (now) so decay/recency treat
        it as fresh, and a fresh embedding for the *new* content. Returns the new
        fact id, or ``""`` if ``fact_id`` does not exist (no silent no-op).
        """
        row = self.conn.execute(
            "SELECT source FROM facts WHERE id = ?", (fact_id,)
        ).fetchone()
        if not row:
            logger.warning("revise_fact: fact %r not found; no revision stored.", fact_id)
            return ""
        embedding_json = ""
        vecs = await self.embed_texts([new_content])
        if vecs:
            embedding_json = json.dumps(vecs[0])
        return await self._supersede_fact(
            fact_id, new_content, row[0], confidence, embedding_json)

    # ------------------------------------------------------------------
    # Reflective memory — journal
    # ------------------------------------------------------------------
    def add_journal_entry(
        self, type: str, content: str, confidence: float = 0.5,
        source_episodes: list[str] | None = None, private: bool = True,
        last_detected: int = 0,
    ) -> str:
        """Store a journal entry. Returns its id. (Sync; embedding is a
        separate async step via :meth:`embed_journal_entry`.)"""
        jid = f"jrn_{uuid.uuid4().hex[:12]}"
        self.conn.execute(
            "INSERT INTO journal (id, type, content, confidence, "
            "source_episodes_json, private, created_at, last_detected) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (jid, type, content, confidence,
             json.dumps(source_episodes or []),
             1 if private else 0, time.time(), last_detected),
        )
        self.conn.commit()
        return jid

    async def embed_journal_entry(self, jid: str) -> None:
        """Compute and store an embedding for a journal entry's content."""
        row = self.conn.execute(
            "SELECT content FROM journal WHERE id = ?", (jid,)
        ).fetchone()
        if not row or not row[0]:
            return
        vecs = await self.embed_texts([row[0]])
        if vecs:
            self.conn.execute(
                "UPDATE journal SET embedding_json = ?, embedding_model = ? WHERE id = ?",
                (json.dumps(vecs[0]), self.embedding_model_name, jid),
            )
            self.conn.commit()

    async def recall_journal_vector(
        self, query_vector: list[float], limit: int = 5,
    ) -> list[JournalEntry]:
        """Vector recall over journal entries that have embeddings.

        Contradictions are excluded — they're unresolved tensions recorded
        for reflection, not beliefs to inject into a turn (see
        ``_active_journal_sql`` for the same filter on the keyword paths).
        """
        rows = self.conn.execute(
            "SELECT id, type, content, confidence, source_episodes_json, "
            "private, created_at, superseded_by, embedding_json "
            "FROM journal WHERE superseded_by = '' AND embedding_json != '' "
            "AND type != 'contradiction' "
            "ORDER BY created_at DESC LIMIT 500"
        ).fetchall()
        entries = [self._journal_from_row(r) for r in rows]
        scored = []
        for j in entries:
            v = j.embedding
            if v is None:
                continue
            sim = cosine(query_vector, v)
            j._relevance = sim  # type: ignore[attr-defined]
            scored.append((sim, j))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [j for s, j in scored[:limit] if s > 0.0]

    def search_journal(self, query: str, limit: int = 10) -> list[JournalEntry]:
        """Keyword search over active journal entries (observations and
        hypotheses only, no contradictions). Used by the memory_search_journal
        tool so the agent can pull relevant past reflections."""
        tokens = self._tokenize(query)
        if not tokens:
            return self.journal_recent(limit=limit)
        # _active_journal_sql already appends ORDER BY created_at DESC,
        # so we just add LIMIT via parameter.
        rows = self.conn.execute(
            self._active_journal_sql(
                "id, type, content, confidence, source_episodes_json, "
                "private, created_at, superseded_by, embedding_json",
                "AND type IN ('observation', 'hypothesis')"
            ) + " LIMIT 500"
        ).fetchall()
        entries = [self._journal_from_row(r) for r in rows]
        scored = []
        for j in entries:
            text = j.content.lower()
            hits = sum(1 for t in tokens if t in text)
            if hits > 0:
                j._relevance = hits / len(tokens)  # type: ignore[attr-defined]
                scored.append((j._relevance, j))  # type: ignore[attr-defined]
        scored.sort(key=lambda t: t[0], reverse=True)
        return [j for _, j in scored[:limit]]

    def search_episodes_by_keyword(
        self, query: str, limit: int = 10,
    ) -> list[Episode]:
        """Keyword search over episodes by task and summary text."""
        tokens = self._tokenize(query)
        if not tokens:
            return self.recent(limit=limit)
        rows = self.conn.execute(
            "SELECT id, task, outcome, summary, tool_calls_json, "
            "created_at, embedding_json FROM episodes "
            "ORDER BY created_at DESC LIMIT 500"
        ).fetchall()
        episodes = [Episode.from_row(tuple(r)) for r in rows]
        scored = []
        for ep in episodes:
            text = (ep.task + " " + (ep.summary or "")).lower()
            hits = sum(1 for t in tokens if t in text)
            if hits > 0:
                scored.append((hits, ep))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [ep for _, ep in scored[:limit]]

    async def retrieve(self, query: str, config: MemoryConfig | None = None) -> str:
        """Retrieve and rank memories across all layers.

        This is the canonical retrieval method. Delegates to
        :func:`memlife.retrieval.retrieve`. If ``config`` is None,
        falls back to ``self.config``.
        """
        from memlife.retrieval import retrieve as _retrieve
        return await _retrieve(self, query, config or self.config)

    def supersede_journal(self, target_id: str, by_id: str) -> bool:
        """Mark a prior journal entry superseded by a new one (closes the loop).

        Returns True if a row was updated; False (with a warning) if the target
        is missing or already superseded/retired — a revision that closes no
        loop is almost certainly a bogus ``revises`` id from the model.
        """
        cur = self.conn.execute(
            "UPDATE journal SET superseded_by = ? WHERE id = ? AND superseded_by = ''",
            (by_id, target_id),
        )
        self.conn.commit()
        if cur.rowcount == 0:
            logger.warning(
                "supersede_journal: target %r not updated (missing or already "
                "superseded/retired); revision %r closes no loop.",
                target_id, by_id,
            )
            return False
        return True

    def retire_journal(self, entry_id: str, reason: str = "low-confidence") -> None:
        """Retire a journal entry from retrieval without deleting it.

        Uses the ``__retired__:<reason>`` sentinel in ``superseded_by``; this is
        distinct from real supersession (see :meth:`supersede_journal`) and is
        reported separately via :attr:`JournalEntry.retired`.
        """
        self.conn.execute(
            "UPDATE journal SET superseded_by = ? WHERE id = ? AND superseded_by = ''",
            (f"__retired__:{reason}", entry_id),
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    # Contradiction lifecycle — dedupe (MF-001) + retirement (MF-004)
    # ------------------------------------------------------------------

    def has_active_contradiction(self, fact_a: str, fact_b: str) -> bool:
        """Return True if an active contradiction already covers this fact pair.

        The pair is considered the same regardless of ordering.
        """
        row = self.conn.execute(
            "SELECT 1 FROM journal "
            "WHERE superseded_by = '' AND type = 'contradiction' "
            "AND ("
            "  (json_extract(source_episodes_json, '$[0]') = ? "
            "   AND json_extract(source_episodes_json, '$[1]') = ?) "
            " OR "
            "  (json_extract(source_episodes_json, '$[0]') = ? "
            "   AND json_extract(source_episodes_json, '$[1]') = ?)"
            ") "
            "LIMIT 1",
            (fact_a, fact_b, fact_b, fact_a),
        ).fetchone()
        return row is not None

    def touch_active_contradiction(
        self, fact_a: str, fact_b: str, cycle: int
    ) -> int:
        """Update last_detected for an active contradiction covering this pair.

        Returns the number of rows updated (0 or 1). The pair is considered
        the same regardless of ordering.
        """
        cur = self.conn.execute(
            "UPDATE journal SET last_detected = ? "
            "WHERE superseded_by = '' AND type = 'contradiction' "
            "AND ("
            "  (json_extract(source_episodes_json, '$[0]') = ? "
            "   AND json_extract(source_episodes_json, '$[1]') = ?) "
            " OR "
            "  (json_extract(source_episodes_json, '$[0]') = ? "
            "   AND json_extract(source_episodes_json, '$[1]') = ?)"
            ")",
            (cycle, fact_a, fact_b, fact_b, fact_a),
        )
        self.conn.commit()
        return cur.rowcount

    def reinforce_unresolved_contradictions(self, current_cycle: int) -> int:
        """Update last_detected for contradictions that are still unresolved.

        A contradiction is unresolved if both source facts still resolve to
        active, distinct facts. These tensions stay alive across reflection
        passes and should not be retired just because they were first created
        a while ago. Returns the number of contradictions reinforced.
        """
        rows = self.conn.execute(
            "SELECT id, source_episodes_json FROM journal "
            "WHERE superseded_by = '' AND type = 'contradiction'"
        ).fetchall()
        reinforced = 0
        for jid, source_json in rows:
            fact_ids: list[str] = []
            try:
                fact_ids = json.loads(source_json or "[]")
            except json.JSONDecodeError:
                continue
            if len(fact_ids) != 2:
                continue
            a = self.resolve_fact(fact_ids[0])
            b = self.resolve_fact(fact_ids[1])
            if a is None or b is None or a.id == b.id:
                continue
            self.conn.execute(
                "UPDATE journal SET last_detected = ? WHERE id = ?",
                (current_cycle, jid),
            )
            reinforced += 1
        self.conn.commit()
        return reinforced

    def retire_stale_contradictions(
        self, current_cycle: int, retirement_cycles: int
    ) -> list[str]:
        """Retire active contradictions not seen in ``retirement_cycles`` passes.

        Returns the IDs of contradictions retired in this pass. Retired entries
        are marked with ``__retired__:stale-contradiction`` so they remain in
        the table but are excluded from active retrieval.
        """
        if retirement_cycles <= 0:
            return []
        cutoff = current_cycle - retirement_cycles
        rows = self.conn.execute(
            "SELECT id FROM journal "
            "WHERE superseded_by = '' AND type = 'contradiction' "
            "AND last_detected <= ?",
            (cutoff,),
        ).fetchall()
        retired_ids: list[str] = []
        for (jid,) in rows:
            self.retire_journal(jid, reason="stale-contradiction")
            retired_ids.append(jid)
        return retired_ids

    def _journal_from_row(self, r) -> JournalEntry:
        return JournalEntry(
            id=r[0], type=r[1], content=r[2], confidence=r[3],
            source_episodes_json=r[4] or "[]", private=bool(r[5]),
            created_at=r[6], superseded_by=r[7] or "",
            embedding_json=(r[8] or "") if len(r) > 8 else "",
            last_detected=r[9] if len(r) > 9 else 0,
        )

    def _active_journal_sql(self, select: str, extra_where: str = "") -> str:
        return (
            f"SELECT {select} FROM journal WHERE superseded_by = '' "
            f"AND type != 'contradiction' "
            f"{extra_where} ORDER BY created_at DESC"
        )

    def journal_contradictions(self, limit: int = 20) -> list[JournalEntry]:
        """Active contradiction journal entries, newest first."""
        rows = self.conn.execute(
            "SELECT id, type, content, confidence, source_episodes_json, "
            "private, created_at, superseded_by, embedding_json, last_detected "
            "FROM journal "
            "WHERE superseded_by = '' AND type = 'contradiction' "
            "ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._journal_from_row(r) for r in rows]

    def journal_recent(self, limit: int = 5) -> list[JournalEntry]:
        rows = self.conn.execute(
            self._active_journal_sql(
                "id, type, content, confidence, source_episodes_json, "
                "private, created_at, superseded_by, embedding_json"
            ) + " LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._journal_from_row(r) for r in rows]

    def journal_by_type(self, type: str, limit: int = 10) -> list[JournalEntry]:
        rows = self.conn.execute(
            self._active_journal_sql(
                "id, type, content, confidence, source_episodes_json, "
                "private, created_at, superseded_by, embedding_json",
                "AND type = ?"
            ) + " LIMIT ?",
            (type, limit),
        ).fetchall()
        return [self._journal_from_row(r) for r in rows]

    def journal_relevant(self, query: str, limit: int = 3) -> list[JournalEntry]:
        """Keyword recall over active journal entries, ranked by recency×confidence."""
        tokens = self._tokenize(query)
        if not tokens:
            rows = self.conn.execute(
                self._active_journal_sql(
                    "id, type, content, confidence, source_episodes_json, "
                    "private, created_at, superseded_by, embedding_json"
                ) + " LIMIT ?",
                (limit,),
            ).fetchall()
            return [self._journal_from_row(r) for r in rows]
        clauses = " OR ".join("content LIKE ?" for _ in tokens)
        params = [f"%{t}%" for t in tokens]
        rows = self.conn.execute(
            f"SELECT id, type, content, confidence, source_episodes_json, "
            f"private, created_at, superseded_by, embedding_json FROM journal "
            f"WHERE superseded_by = '' AND type != 'contradiction' "
            f"AND ({clauses}) "
            f"ORDER BY created_at DESC LIMIT ?",
            (*params, limit * 4),
        ).fetchall()
        entries = [self._journal_from_row(r) for r in rows]
        n_tokens = max(1, len(tokens))
        for j in entries:
            hay = j.content.lower()
            j._match_score = sum(1 for t in tokens if t in hay)  # type: ignore[attr-defined]
            j._relevance = j._match_score / n_tokens  # type: ignore[attr-defined]
        entries.sort(
            key=lambda e: (getattr(e, "_match_score", 0), e.confidence, e.created_at),
            reverse=True,
        )
        return entries[:limit]

    def consolidate_journal(
        self, halflife_days: float = 30.0, floor: float = 0.15,
    ) -> dict:
        """Retire entries whose decayed confidence has fallen below ``floor``,
        and merge near-duplicate active observations. Returns counts.

        Retirement is batched into a single transaction (one commit, not one
        per entry). Merging uses token-Jaccard similarity (>= 0.8) rather than a
        raw 40-char prefix, so two observations that share a long prefix but
        diverge substantially are no longer collapsed — and two paraphrases
        that share no prefix are.
        """
        retired = 0
        merged = 0
        active = self.journal_recent(limit=500)
        # Retire low effective-confidence observations/hypotheses — batched.
        to_retire = [
            j.id for j in active
            if j.type in ("observation", "hypothesis")
            and j.effective_confidence(halflife_days, floor) <= floor
        ]
        if to_retire:
            placeholders = ",".join("?" * len(to_retire))
            self.conn.execute(
                f"UPDATE journal SET superseded_by = '__retired__:low-confidence' "
                f"WHERE id IN ({placeholders}) AND superseded_by = ''",
                tuple(to_retire),
            )
            self.conn.commit()
            retired = len(to_retire)
        # Merge near-duplicate active observations by token-Jaccard similarity.
        surviving = self.journal_by_type("observation", limit=500)
        token_sets = [set(self._tokenize(j.content)) for j in surviving]
        superseded_ids: set[str] = set()
        for i, j in enumerate(surviving):
            ti = token_sets[i]
            if not ti or j.id in superseded_ids:
                continue
            for k in range(i + 1, len(surviving)):
                if surviving[k].id in superseded_ids:
                    continue
                tk = token_sets[k]
                if not tk:
                    continue
                union = ti | tk
                sim = len(ti & tk) / len(union)
                # surviving is newest-first, so j (i) is newer than k (i<k).
                # Supersede the older (k) with the newer (j).
                if sim >= 0.8:
                    if self.supersede_journal(surviving[k].id, j.id):
                        merged += 1
                        superseded_ids.add(surviving[k].id)
        return {"retired": retired, "merged": merged}

    def queue_reflection(self, episode_id: str) -> None:
        """Flag an episode for the nightly reflection to consider.

        Idempotent at the DB level: the unique index on ``episode_id`` means a
        re-queue of the same episode is a no-op (``INSERT OR IGNORE``), with no
        TOCTOU window between a SELECT and an INSERT.
        """
        qid = f"rq_{uuid.uuid4().hex[:12]}"
        self.conn.execute(
            "INSERT OR IGNORE INTO reflection_queue (id, episode_id, queued_at) "
            "VALUES (?, ?, ?)",
            (qid, episode_id, time.time()),
        )
        self.conn.commit()

    def pending_reflections(self) -> list[str]:
        rows = self.conn.execute(
            "SELECT episode_id FROM reflection_queue WHERE reflected = 0 "
            "ORDER BY queued_at ASC"
        ).fetchall()
        return [r[0] for r in rows]

    def mark_reflected(self, episode_id: str) -> None:
        self.conn.execute(
            "UPDATE reflection_queue SET reflected = 1 WHERE episode_id = ?",
            (episode_id,),
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    # Checkpointing (unchanged from MVP)
    # ------------------------------------------------------------------
    def start_run(self, task: str, model: str = "") -> str:
        run_id = f"run_{uuid.uuid4().hex[:12]}"
        self.conn.execute(
            "INSERT INTO agent_runs (id, task, status, created_at, model_used) "
            "VALUES (?, ?, 'running', ?, ?)",
            (run_id, task, time.time(), model),
        )
        self.conn.commit()
        return run_id

    def save_checkpoint(
        self, run_id: str, step_index: int, step_description: str,
        state: dict, tool_calls: list[dict] | None = None,
        observation: str = "", outcome: str = "success",
        tokens_used: int = 0,
    ) -> str:
        cp_id = f"cp_{uuid.uuid4().hex[:12]}"
        self.conn.execute(
            "INSERT OR REPLACE INTO checkpoints "
            "(id, run_id, step_index, step_description, state_json, "
            "tool_calls_json, observation, outcome, tokens_used, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (cp_id, run_id, step_index, step_description,
             json.dumps(state), json.dumps(tool_calls or []),
             observation, outcome, tokens_used, time.time()),
        )
        self.conn.commit()
        return cp_id

    def get_last_checkpoint(self, run_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT state_json FROM checkpoints WHERE run_id = ? "
            "ORDER BY step_index DESC LIMIT 1",
            (run_id,),
        ).fetchone()
        if row:
            return json.loads(row[0])
        return None

    def complete_run(
        self, run_id: str, total_tokens: int = 0, error: str = ""
    ) -> None:
        status = "failed" if error else "completed"
        self.conn.execute(
            "UPDATE agent_runs SET status = ?, completed_at = ?, "
            "total_tokens = ?, error_message = ? WHERE id = ?",
            (status, time.time(), total_tokens, error or None, run_id),
        )
        self.conn.commit()

    def trace_event(self, run_id: str, event: str, detail: dict | None = None) -> None:
        """Append a structured trace event to a run.

        Capped at ``TRACE_EVENT_LIMIT`` events (oldest dropped) so the trace
        blob can't grow without bound across a long run. Still a read-modify-
        write of the JSON blob — fine for the single-writer agent loop.
        """
        row = self.conn.execute(
            "SELECT trace_json FROM agent_runs WHERE id = ?", (run_id,)
        ).fetchone()
        if not row:
            return
        try:
            trace = json.loads(row[0])
        except json.JSONDecodeError:
            trace = []
        trace.append({"ts": time.time(), "event": event, "detail": detail or {}})
        if len(trace) > TRACE_EVENT_LIMIT:
            trace = trace[-TRACE_EVENT_LIMIT:]
        self.conn.execute(
            "UPDATE agent_runs SET trace_json = ? WHERE id = ?",
            (json.dumps(trace), run_id),
        )
        self.conn.commit()

    def get_incomplete_run(self) -> dict | None:
        row = self.conn.execute(
            "SELECT id, task, model_used FROM agent_runs "
            "WHERE status = 'running' ORDER BY created_at DESC LIMIT 1",
        ).fetchone()
        if row:
            return {"id": row[0], "task": row[1], "model_used": row[2]}
        return None

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # Embedding health + backfill
    # ------------------------------------------------------------------
    def embedding_health(self) -> dict:
        """Return a snapshot of embedding coverage across all memory layers.

        Reports how many facts, journal entries, and episodes have embeddings
        vs how many are missing them, plus the consecutive failure counter.
        Also reports how many embeddings are stale (created with a different
        model than the currently configured one). Use this to detect silent
        degradation of the semantic layer.
        """
        def _count(table: str) -> dict:
            total = self.conn.execute(
                f"SELECT COUNT(*) FROM {table}"
            ).fetchone()[0]
            with_vec = self.conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE embedding_json != ''"
            ).fetchone()[0]
            # Stale = has a vector but the model name doesn't match current.
            # Only counts if we actually have a model name configured.
            stale = 0
            if self.embedding_model_name:
                stale = self.conn.execute(
                    f"SELECT COUNT(*) FROM {table} "
                    f"WHERE embedding_json != '' "
                    f"AND embedding_model != ?",
                    (self.embedding_model_name,),
                ).fetchone()[0]
            return {"total": total, "with_embeddings": with_vec,
                    "missing": total - with_vec, "stale": stale}

        return {
            "facts": _count("facts"),
            "journal": _count("journal"),
            "episodes": _count("episodes"),
            "embedding_model": self.embedding_model_name,
            "consecutive_failures": self._embed_failures,
            "embedder_present": self.embedder is not None,
        }

    async def backfill_embeddings(self, batch_size: int = 20) -> dict:
        """Re-embed facts, journal entries, and episodes that are missing vectors
        or whose vectors were created with a different embedding model.

        Processes in batches to avoid hammering the embedder. Skips items where
        the content is empty. Returns counts of how many were embedded and how
        many failed. Safe to run repeatedly — only processes items with empty
        embedding_json or a mismatched embedding_model.
        """
        results = {"facts_embedded": 0, "journal_embedded": 0,
                    "episodes_embedded": 0, "failed": 0}

        # Build the stale-condition: missing OR model mismatch.
        # When no model name is configured, only missing embeddings are
        # re-embedded (backward compatible).
        model_clause = ""
        model_params: list = []
        if self.embedding_model_name:
            model_clause = " OR embedding_model != ?"
            model_params = [self.embedding_model_name]

        # Facts without embeddings or with stale model.
        fact_rows = self.conn.execute(
            f"SELECT id, content FROM facts "
            f"WHERE (embedding_json = ''{model_clause}) AND content != ''",
            model_params,
        ).fetchall()
        for i in range(0, len(fact_rows), batch_size):
            batch = fact_rows[i:i + batch_size]
            texts = [r[1] for r in batch]
            vecs = await self.embed_texts(texts)
            if vecs is None:
                results["failed"] += len(batch)
                continue
            for row, vec in zip(batch, vecs):
                if vec:
                    self.conn.execute(
                        "UPDATE facts SET embedding_json = ?, embedding_model = ? WHERE id = ?",
                        (json.dumps(vec), self.embedding_model_name, row[0]),
                    )
                    results["facts_embedded"] += 1
                else:
                    results["failed"] += 1
            self.conn.commit()

        # Journal entries without embeddings (skip contradictions — they're
        # not retrieved, so embedding them is wasted work) or with stale model.
        j_rows = self.conn.execute(
            f"SELECT id, content FROM journal "
            f"WHERE (embedding_json = ''{model_clause}) "
            f"AND content != '' AND type != 'contradiction'",
            model_params,
        ).fetchall()
        for i in range(0, len(j_rows), batch_size):
            batch = j_rows[i:i + batch_size]
            texts = [r[1] for r in batch]
            vecs = await self.embed_texts(texts)
            if vecs is None:
                results["failed"] += len(batch)
                continue
            for row, vec in zip(batch, vecs):
                if vec:
                    self.conn.execute(
                        "UPDATE journal SET embedding_json = ?, embedding_model = ? WHERE id = ?",
                        (json.dumps(vec), self.embedding_model_name, row[0]),
                    )
                    results["journal_embedded"] += 1
                else:
                    results["failed"] += 1
            self.conn.commit()

        # Episodes without embeddings or with stale model.
        ep_rows = self.conn.execute(
            f"SELECT id, task, summary FROM episodes "
            f"WHERE (embedding_json = ''{model_clause}) AND task != ''",
            model_params,
        ).fetchall()
        ep_texts = [f"{r[1]}\n{r[2] or ''}".strip() for r in ep_rows]
        for i in range(0, len(ep_rows), batch_size):
            batch_texts = ep_texts[i:i + batch_size]
            batch_rows = ep_rows[i:i + batch_size]
            vecs = await self.embed_texts(batch_texts)
            if vecs is None:
                results["failed"] += len(batch_rows)
                continue
            for row, vec in zip(batch_rows, vecs):
                if vec:
                    self.conn.execute(
                        "UPDATE episodes SET embedding_json = ?, embedding_model = ? WHERE id = ?",
                        (json.dumps(vec), self.embedding_model_name, row[0]),
                    )
                    results["episodes_embedded"] += 1
                else:
                    results["failed"] += 1
            self.conn.commit()

        return results

    # ------------------------------------------------------------------
    # Sessions — named conversation containers
    # ------------------------------------------------------------------
    def list_sessions(self, limit: int = 20) -> list[dict]:
        rows = self.conn.execute(
            "SELECT id, name, created_at, updated_at, model_used "
            "FROM sessions ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            {"id": r[0], "name": r[1], "created_at": r[2],
             "updated_at": r[3], "model_used": r[4]}
            for r in rows
        ]

    def create_session(self, name: str, model: str = "") -> str:
        sid = f"sess_{uuid.uuid4().hex[:12]}"
        now = time.time()
        self.conn.execute(
            "INSERT INTO sessions (id, name, created_at, updated_at, model_used) "
            "VALUES (?, ?, ?, ?, ?)",
            (sid, name, now, now, model),
        )
        self.conn.commit()
        return sid

    def load_session(self, session_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT id, name, model_used, conversation_json, rolling_summary "
            "FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        if not row:
            return None
        try:
            conversation = json.loads(row[3])
        except json.JSONDecodeError:
            conversation = []
        return {
            "id": row[0], "name": row[1], "model_used": row[2],
            "conversation": conversation, "rolling_summary": row[4] or "",
        }

    def save_session(self, session_id: str, conversation: list[dict],
                     model: str = "", rolling_summary: str = "") -> None:
        self.conn.execute(
            "UPDATE sessions SET conversation_json = ?, updated_at = ?, "
            "model_used = CASE WHEN ? != '' THEN ? ELSE model_used END, "
            "rolling_summary = CASE WHEN ? != '' THEN ? ELSE rolling_summary END "
            "WHERE id = ?",
            (json.dumps(conversation), time.time(), model, model,
             rolling_summary, rolling_summary, session_id),
        )
        self.conn.commit()

    def delete_session(self, session_id: str) -> bool:
        cur = self.conn.execute(
            "DELETE FROM sessions WHERE id = ?", (session_id,),
        )
        self.conn.commit()
        return cur.rowcount > 0

    # ------------------------------------------------------------------
    # Continuity metrics
    # ------------------------------------------------------------------
    def record_reflection_metrics(self, metrics: dict) -> str:
        """Store a reflection metrics snapshot. Returns the metric ID."""
        mid = f"met_{uuid.uuid4().hex[:12]}"
        self.conn.execute(
            "INSERT INTO reflection_metrics (id, created_at, "
            "episodes_considered, observations_proposed, observations_kept, "
            "hypotheses_proposed, hypotheses_kept, revisions_proposed, "
            "revisions_kept, contradictions_found, avg_confidence, keep_rate, "
            "consolidated_retired, consolidated_merged, total_journal_entries, "
            "total_facts, total_episodes) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (mid, time.time(),
             metrics.get("episodes_considered", 0),
             metrics.get("observations_proposed", 0),
             metrics.get("observations_kept", 0),
             metrics.get("hypotheses_proposed", 0),
             metrics.get("hypotheses_kept", 0),
             metrics.get("revisions_proposed", 0),
             metrics.get("revisions_kept", 0),
             metrics.get("contradictions_found", 0),
             metrics.get("avg_confidence", 0.0),
             metrics.get("keep_rate", 0.0),
             metrics.get("consolidated_retired", 0),
             metrics.get("consolidated_merged", 0),
             metrics.get("total_journal_entries", 0),
             metrics.get("total_facts", 0),
             metrics.get("total_episodes", 0)),
        )
        self.conn.commit()
        return mid

    def get_metrics_history(self, limit: int = 20) -> list[dict]:
        """Return recent reflection metrics, newest first."""
        rows = self.conn.execute(
            "SELECT * FROM reflection_metrics ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def run_gc(
        self,
        *,
        superseded_facts_days: int = 90,
        superseded_journal_days: int = 90,
        completed_runs_days: int = 60,
        metrics_days: int = 30,
        reflected_queue_days: int = 30,
    ) -> dict:
        """Run garbage collection on old/superseded data.

        Returns a dict with counts of what was pruned. All deletions are
        hard-deletes — the data is already superseded or obsolete. The
        backup rotation (ingrid-db-backup.sh, daily at 2am) provides
        the recovery path if something is deleted that shouldn't have been.

        Defaults are conservative:
          - Superseded facts: 90 days after supersession
          - Superseded journal entries: 90 days
          - Completed agent runs + their checkpoints: 60 days
          - Reflection metrics: 30 days
          - Reflected queue entries: 30 days
        """
        now = time.time()
        cutoff_facts = now - (superseded_facts_days * 86400)
        cutoff_journal = now - (superseded_journal_days * 86400)
        cutoff_runs = now - (completed_runs_days * 86400)
        cutoff_metrics = now - (metrics_days * 86400)
        cutoff_queue = now - (reflected_queue_days * 86400)

        pruned: dict[str, int] = {}

        # Superseded facts (superseded_by is set, and updated_at is old).
        cur = self.conn.execute(
            "DELETE FROM facts WHERE superseded_by != '' AND updated_at < ?",
            (cutoff_facts,),
        )
        pruned["superseded_facts"] = cur.rowcount

        # Superseded journal entries.
        cur = self.conn.execute(
            "DELETE FROM journal WHERE superseded_by != '' AND created_at < ?",
            (cutoff_journal,),
        )
        pruned["superseded_journal"] = cur.rowcount

        # Completed agent runs and their checkpoints.
        old_run_ids = [
            r[0] for r in self.conn.execute(
                "SELECT id FROM agent_runs "
                "WHERE status != 'running' AND completed_at IS NOT NULL "
                "AND completed_at < ?",
                (cutoff_runs,),
            ).fetchall()
        ]
        if old_run_ids:
            placeholders = ",".join("?" * len(old_run_ids))
            cur = self.conn.execute(
                f"DELETE FROM checkpoints WHERE run_id IN ({placeholders})",
                old_run_ids,
            )
            pruned["checkpoints"] = cur.rowcount
            cur = self.conn.execute(
                f"DELETE FROM agent_runs WHERE id IN ({placeholders})",
                old_run_ids,
            )
            pruned["agent_runs"] = cur.rowcount
        else:
            pruned["checkpoints"] = 0
            pruned["agent_runs"] = 0

        # Old reflection metrics.
        cur = self.conn.execute(
            "DELETE FROM reflection_metrics WHERE created_at < ?",
            (cutoff_metrics,),
        )
        pruned["reflection_metrics"] = cur.rowcount

        # Reflected queue entries (already processed, just noise).
        cur = self.conn.execute(
            "DELETE FROM reflection_queue WHERE reflected = 1 AND queued_at < ?",
            (cutoff_queue,),
        )
        pruned["reflected_queue"] = cur.rowcount

        self.conn.commit()

        # Reclaim space — VACUUM rebuilds the file. This is the one
        # operation that actually shrinks the file on disk.
        old_size = self.conn.execute(
            "SELECT page_count * page_size FROM pragma_page_count(), pragma_page_size()"
        ).fetchone()[0]
        self.conn.execute("VACUUM")
        new_size = self.conn.execute(
            "SELECT page_count * page_size FROM pragma_page_count(), pragma_page_size()"
        ).fetchone()[0]

        pruned["db_size_before_mb"] = round(old_size / 1024 / 1024, 1)
        pruned["db_size_after_mb"] = round(new_size / 1024 / 1024, 1)
        pruned["total_pruned"] = sum(
            v for k, v in pruned.items()
            if isinstance(v, int) and k not in ("db_size_before_mb", "db_size_after_mb")
        )
        return pruned

    def get_metrics_summary(self) -> dict:
        """Return aggregate metrics across all reflections plus current unresolved contradictions."""
        row = self.conn.execute(
            "SELECT COUNT(*) as total_reflections, "
            "AVG(keep_rate) as avg_keep_rate, "
            "AVG(avg_confidence) as avg_confidence, "
            "SUM(observations_kept) as total_obs_kept, "
            "SUM(hypotheses_kept) as total_hyp_kept, "
            "SUM(revisions_kept) as total_rev_kept, "
            "SUM(contradictions_found) as total_contradictions, "
            "SUM(consolidated_retired) as total_retired, "
            "SUM(consolidated_merged) as total_merged "
            "FROM reflection_metrics"
        ).fetchone()
        if not row or row[0] == 0:
            summary = {"total_reflections": 0}
        else:
            summary = dict(row)
        # Add live unresolved contradiction count (stored journal entries that
        # still resolve to two active, distinct facts).
        contradictions = self.list_contradictions(limit=1000)
        summary["unresolved_contradictions"] = sum(1 for c in contradictions if c["unresolved"])
        return summary

