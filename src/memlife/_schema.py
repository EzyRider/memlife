"""Create and migrate the SQLite schema.

Extracted from store.py as part of the mixin refactor.
"""

from __future__ import annotations

import json
import logging


logger = logging.getLogger(__name__)

# Maximum confidence allowed for a stored fact. 1.0 is reserved: it implies
# immutable certainty, which blocks revision. Cap below 1.0 so every fact
# remains updateable.
MAX_FACT_CONFIDENCE = 0.99


class SchemaMixin:
    """Create and migrate the SQLite schema."""

    db_path: str
    config: object
    _conn: object
    conn: object
    _lock: object

    def _init_schema(self) -> None:
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS episodes (
                id TEXT PRIMARY KEY, task TEXT NOT NULL,
                outcome TEXT NOT NULL DEFAULT 'running',
                summary TEXT DEFAULT '', tool_calls_json TEXT DEFAULT '[]',
                created_at REAL NOT NULL,
                embedding_json TEXT DEFAULT '',
                embedding_model TEXT DEFAULT '',
                is_gap_marker INTEGER DEFAULT 0
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
                superseded_by TEXT DEFAULT '',
                annotations_json TEXT DEFAULT '[]',
                embedding_model TEXT DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS idx_facts_content
                ON facts(content);
            CREATE TABLE IF NOT EXISTS journal (
                id TEXT PRIMARY KEY, type TEXT NOT NULL,
                content TEXT NOT NULL, confidence REAL DEFAULT 0.5,
                -- source_episodes_json: for observations/hypotheses/revisions
                -- this holds episode IDs. For contradictions it holds the two
                -- conflicting fact IDs (MF-016: documented overload, not renamed
                -- to avoid breaking existing databases).
                source_episodes_json TEXT DEFAULT '[]',
                private INTEGER DEFAULT 1,
                created_at REAL NOT NULL,
                superseded_by TEXT DEFAULT '',
                embedding_json TEXT DEFAULT '',
                embedding_model TEXT DEFAULT '',
                last_detected INTEGER DEFAULT 0,
                annotations_json TEXT DEFAULT '[]',
                links_json TEXT DEFAULT '[]'
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
            CREATE TABLE IF NOT EXISTS reflection_passes (
                id TEXT PRIMARY KEY,
                created_at REAL NOT NULL,
                episode_ids_json TEXT DEFAULT '[]',
                proposed_json TEXT DEFAULT '[]',
                kept_json TEXT DEFAULT '[]',
                dropped_json TEXT DEFAULT '[]',
                model_used TEXT DEFAULT '',
                critic_model_used TEXT DEFAULT '',
                total_timeout REAL DEFAULT 0,
                elapsed_seconds REAL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_reflection_passes_created
                ON reflection_passes(created_at);
            CREATE TABLE IF NOT EXISTS episode_tools (
                episode_id TEXT NOT NULL,
                tool_name TEXT NOT NULL,
                created_at REAL NOT NULL,
                PRIMARY KEY (episode_id, tool_name)
            );
            CREATE INDEX IF NOT EXISTS idx_episode_tools_name
                ON episode_tools(tool_name);

            -- Temporal triple store (MV2-003): subject-predicate-object facts
            -- with valid time ranges. Empty valid_until means currently true.
            CREATE TABLE IF NOT EXISTS temporal_triples (
                id TEXT PRIMARY KEY,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                valid_from REAL NOT NULL,
                valid_until REAL,
                fact_id TEXT,
                confidence REAL DEFAULT 0.5,
                created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_triples_subject_predicate
                ON temporal_triples(subject, predicate, valid_from);
            CREATE INDEX IF NOT EXISTS idx_triples_object
                ON temporal_triples(object);
            CREATE INDEX IF NOT EXISTS idx_triples_predicate
                ON temporal_triples(predicate);
            CREATE INDEX IF NOT EXISTS idx_triples_fact
                ON temporal_triples(fact_id);

            -- Entity normalization and aliases (MV2-010)
            CREATE TABLE IF NOT EXISTS entities (
                canonical_name TEXT PRIMARY KEY,
                aliases_json TEXT DEFAULT '[]',
                created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS entity_aliases (
                alias TEXT PRIMARY KEY,
                canonical_name TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_entity_aliases_canonical
                ON entity_aliases(canonical_name);

            -- Triple provenance: which episode/fact/journal asserted a triple
            CREATE TABLE IF NOT EXISTS triple_provenance (
                triple_id TEXT NOT NULL,
                source_kind TEXT NOT NULL,
                source_id TEXT NOT NULL,
                created_at REAL NOT NULL,
                PRIMARY KEY (triple_id, source_kind, source_id)
            );
            CREATE INDEX IF NOT EXISTS idx_provenance_source
                ON triple_provenance(source_kind, source_id);

            -- Embedding cache (0.6.0): content-addressable vectors keyed on
            -- (model_name, sha256(text)).  Vectors are stored as canonical JSON
            -- floats so switching vector_backend never leaves cache rows unreadable.
            CREATE TABLE IF NOT EXISTS embedding_cache (
                cache_key TEXT PRIMARY KEY,
                model_name TEXT NOT NULL,
                text_hash TEXT NOT NULL,
                vector_json TEXT NOT NULL,
                created_at REAL NOT NULL,
                last_used_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_embedding_cache_model_hash
                ON embedding_cache(model_name, text_hash);
            CREATE INDEX IF NOT EXISTS idx_embedding_cache_last_used
                ON embedding_cache(last_used_at);
        """)
        self.conn.commit()

    def migration_status(self) -> dict:
        """Return a snapshot of schema health and pending migrations.

        Checks whether tables/columns expected by the current code exist in
        the connected database, flags any missing items, and reports the
        SQLite version and page stats.  This is used by ``store.metrics()``
        and can be surfaced by operators before upgrading.
        """
        import sqlite3

        expected_tables = {
            "episodes", "agent_runs", "checkpoints", "facts", "journal",
            "reflection_queue", "sessions", "reflection_metrics",
            "reflection_passes", "episode_tools", "temporal_triples",
            "entities", "entity_aliases", "triple_provenance",
            "embedding_cache",
        }
        expected_columns = {
            ("episodes", "embedding_json"),
            ("episodes", "embedding_model"),
            ("episodes", "is_gap_marker"),
            ("facts", "embedding_json"),
            ("facts", "embedding_model"),
            ("facts", "annotations_json"),
            ("journal", "embedding_json"),
            ("journal", "embedding_model"),
            ("journal", "superseded_by"),
            ("journal", "last_detected"),
            ("journal", "annotations_json"),
            ("journal", "links_json"),
            ("agent_runs", "trace_json"),
            ("sessions", "rolling_summary"),
        }

        existing_tables = {
            r["name"]
            for r in self.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        missing_tables = sorted(expected_tables - existing_tables)

        missing_columns: list[str] = []
        for table, column in expected_columns:
            if table not in existing_tables:
                continue
            cols = {
                r["name"]
                for r in self.conn.execute(f"PRAGMA table_info({table})").fetchall()
            }
            if column not in cols:
                missing_columns.append(f"{table}.{column}")

        sqlite_version = sqlite3.sqlite_version
        page_size = 0
        page_count = 0
        try:
            page_size = self.conn.execute("PRAGMA page_size").fetchone()[0]
            page_count = self.conn.execute("PRAGMA page_count").fetchone()[0]
        except Exception:
            pass

        return {
            "sqlite_version": sqlite_version,
            "page_size": page_size,
            "page_count": page_count,
            "tables_expected": len(expected_tables),
            "tables_present": len(existing_tables & expected_tables),
            "missing_tables": missing_tables,
            "missing_columns": sorted(missing_columns),
            "healthy": not missing_tables and not missing_columns,
        }

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

        # Reflection audit table: added in 0.5.0.  Idempotent for existing DBs.
        rpass_cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(reflection_passes)")}
        if not rpass_cols:
            self.conn.executescript("""
                CREATE TABLE IF NOT EXISTS reflection_passes (
                    id TEXT PRIMARY KEY,
                    created_at REAL NOT NULL,
                    episode_ids_json TEXT DEFAULT '[]',
                    proposed_json TEXT DEFAULT '[]',
                    kept_json TEXT DEFAULT '[]',
                    dropped_json TEXT DEFAULT '[]',
                    model_used TEXT DEFAULT '',
                    critic_model_used TEXT DEFAULT '',
                    total_timeout REAL DEFAULT 0,
                    elapsed_seconds REAL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_reflection_passes_created
                    ON reflection_passes(created_at);
            """)

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

        # MV2-003 / MV2-010: ensure temporal_triples, entity, and provenance
        # tables exist in existing databases.
        tcols = {r["name"] for r in self.conn.execute("PRAGMA table_info(temporal_triples)")}
        if not tcols:
            self.conn.executescript("""
                CREATE TABLE IF NOT EXISTS temporal_triples (
                    id TEXT PRIMARY KEY,
                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    object TEXT NOT NULL,
                    valid_from REAL NOT NULL,
                    valid_until REAL,
                    fact_id TEXT,
                    confidence REAL DEFAULT 0.5,
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_triples_subject_predicate
                    ON temporal_triples(subject, predicate, valid_from);
                CREATE INDEX IF NOT EXISTS idx_triples_object
                    ON temporal_triples(object);
                CREATE INDEX IF NOT EXISTS idx_triples_predicate
                    ON temporal_triples(predicate);
                CREATE INDEX IF NOT EXISTS idx_triples_fact
                    ON temporal_triples(fact_id);

                CREATE TABLE IF NOT EXISTS entities (
                    canonical_name TEXT PRIMARY KEY,
                    aliases_json TEXT DEFAULT '[]',
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS entity_aliases (
                    alias TEXT PRIMARY KEY,
                    canonical_name TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_entity_aliases_canonical
                    ON entity_aliases(canonical_name);

                CREATE TABLE IF NOT EXISTS triple_provenance (
                    triple_id TEXT NOT NULL,
                    source_kind TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY (triple_id, source_kind, source_id)
                );
                CREATE INDEX IF NOT EXISTS idx_provenance_source
                    ON triple_provenance(source_kind, source_id);
            """)

        # MV2-010: ensure entity/provenance tables exist even if temporal_triples
        # was created before this migration step.
        if tcols:
            self.conn.executescript("""
                CREATE TABLE IF NOT EXISTS entities (
                    canonical_name TEXT PRIMARY KEY,
                    aliases_json TEXT DEFAULT '[]',
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS entity_aliases (
                    alias TEXT PRIMARY KEY,
                    canonical_name TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_entity_aliases_canonical
                    ON entity_aliases(canonical_name);

                CREATE TABLE IF NOT EXISTS triple_provenance (
                    triple_id TEXT NOT NULL,
                    source_kind TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY (triple_id, source_kind, source_id)
                );
                CREATE INDEX IF NOT EXISTS idx_provenance_source
                    ON triple_provenance(source_kind, source_id);
            """)

        # MV2-008: ensure episode gap-marker flag exists in existing databases.
        ecols = {r["name"] for r in self.conn.execute("PRAGMA table_info(episodes)")}
        if "is_gap_marker" not in ecols:
            self.conn.execute("ALTER TABLE episodes ADD COLUMN is_gap_marker INTEGER DEFAULT 0")

        # MV2-004: ensure annotations_json columns exist on facts and journal.
        fcols = {r["name"] for r in self.conn.execute("PRAGMA table_info(facts)")}
        if "annotations_json" not in fcols:
            self.conn.execute("ALTER TABLE facts ADD COLUMN annotations_json TEXT DEFAULT '[]'")
        # Re-read journal columns: earlier migration steps may have just added
        # columns, so the jcols snapshot from the start of _migrate() is stale.
        jcols = {r["name"] for r in self.conn.execute("PRAGMA table_info(journal)")}
        if "annotations_json" not in jcols:
            self.conn.execute("ALTER TABLE journal ADD COLUMN annotations_json TEXT DEFAULT '[]'")
        if "links_json" not in jcols:
            self.conn.execute("ALTER TABLE journal ADD COLUMN links_json TEXT DEFAULT '[]'")

        # 0.6.0: ensure embedding_cache table exists in existing databases.
        ccols = {r["name"] for r in self.conn.execute("PRAGMA table_info(embedding_cache)")}
        if not ccols:
            self.conn.executescript("""
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    cache_key TEXT PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    text_hash TEXT NOT NULL,
                    vector_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    last_used_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_embedding_cache_model_hash
                    ON embedding_cache(model_name, text_hash);
                CREATE INDEX IF NOT EXISTS idx_embedding_cache_last_used
                    ON embedding_cache(last_used_at);
            """)

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

