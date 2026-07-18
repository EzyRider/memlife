"""Prune stale memory rows and expose diagnostics.

Extracted from store.py as part of the mixin refactor.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memlife.models import Metrics


logger = logging.getLogger(__name__)


class GCMixin:
    """Prune stale memory rows and expose diagnostics."""

    db_path: str
    config: object
    _conn: object
    conn: object
    _lock: object

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

    def record_reflection_pass(self, pass_obj) -> str:
        """Persist a :class:`memlife.reflection.ReflectionPass`.

        The ``pass_obj`` may be a dict or a dataclass instance.  Returns the
        pass id.
        """
        if hasattr(pass_obj, "__dict__"):
            data = pass_obj.__dict__
        else:
            data = dict(pass_obj)
        pid = data.get("id") or f"rp_{uuid.uuid4().hex[:12]}"
        self.conn.execute(
            "INSERT INTO reflection_passes (id, created_at, episode_ids_json, "
            "proposed_json, kept_json, dropped_json, model_used, "
            "critic_model_used, total_timeout, elapsed_seconds) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                pid,
                data.get("created_at", time.time()),
                json.dumps(data.get("episode_ids", [])),
                json.dumps(data.get("proposed", [])),
                json.dumps(data.get("kept", [])),
                json.dumps(data.get("dropped", [])),
                data.get("model_used", ""),
                data.get("critic_model_used") or "",
                data.get("total_timeout", 0.0),
                data.get("elapsed_seconds", 0.0),
            ),
        )
        self.conn.commit()
        self._prune_reflection_passes()
        return pid

    def reflection_audit(
        self,
        *,
        limit: int = 20,
        before: float | None = None,
        after: float | None = None,
    ) -> list[dict]:
        """Return paginated reflection pass records for debugging.

        Passes are returned newest first.  ``before``/``after`` filter by
        ``created_at`` (exclusive).
        """
        where = []
        params: list = []
        if before is not None:
            where.append("created_at < ?")
            params.append(before)
        if after is not None:
            where.append("created_at > ?")
            params.append(after)
        where_sql = f"WHERE {' AND '.join(where)}" if where else ""
        rows = self.conn.execute(
            f"SELECT * FROM reflection_passes {where_sql} "
            f"ORDER BY created_at DESC LIMIT ?",
            (*params, limit),
        ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            for key in ("episode_ids_json", "proposed_json", "kept_json", "dropped_json"):
                try:
                    d[key.replace("_json", "")] = json.loads(d[key] or "[]")
                except json.JSONDecodeError:
                    d[key.replace("_json", "")] = []
                del d[key]
            result.append(d)
        return result

    def last_reflection_pass(self) -> dict | None:
        """Return the most recent reflection pass, or None."""
        rows = self.reflection_audit(limit=1)
        return rows[0] if rows else None

    def _prune_reflection_passes(self) -> None:
        """Cap reflection pass history by count and age.

        Uses ``reflection_pass_retention_count`` and
        ``reflection_pass_retention_days`` from ``self.config``.
        """
        count_cap = getattr(self.config, "reflection_pass_retention_count", 100)
        days_cap = getattr(self.config, "reflection_pass_retention_days", 90)
        if count_cap > 0:
            self.conn.execute(
                "DELETE FROM reflection_passes WHERE id NOT IN ("
                "  SELECT id FROM reflection_passes ORDER BY created_at DESC LIMIT ?"
                ")",
                (count_cap,),
            )
        if days_cap > 0:
            cutoff = time.time() - (days_cap * 86400)
            self.conn.execute(
                "DELETE FROM reflection_passes WHERE created_at < ?",
                (cutoff,),
            )
        self.conn.commit()

    def run_gc(
        self,
        *,
        superseded_facts_days: int = 90,
        superseded_journal_days: int = 90,
        completed_runs_days: int = 60,
        metrics_days: int = 30,
        reflected_queue_days: int = 30,
        episodes_days: int = 180,
        closed_triples_days: int = 90,
    ) -> dict:
        """Run garbage collection on old/superseded data.

        Returns a dict with counts of what was pruned. All deletions are
        hard-deletes — the data is already superseded or obsolete. Keep
        backups before running GC on production databases.

        Defaults are conservative:
          - Superseded facts: 90 days after supersession
          - Superseded journal entries: 90 days
          - Completed agent runs + their checkpoints: 60 days
          - Reflection metrics: 30 days
          - Reflected queue entries: 30 days
          - Closed temporal triples: 90 days after valid_until
        """
        now = time.time()
        cutoff_facts = now - (superseded_facts_days * 86400)
        cutoff_journal = now - (superseded_journal_days * 86400)
        cutoff_runs = now - (completed_runs_days * 86400)
        cutoff_metrics = now - (metrics_days * 86400)
        cutoff_queue = now - (reflected_queue_days * 86400)
        cutoff_triples = now - (closed_triples_days * 86400)

        pruned: dict[str, int] = {}

        # Superseded facts (superseded_by is set, and updated_at is old).
        old_fact_ids = [
            r[0] for r in self.conn.execute(
                "SELECT id FROM facts WHERE superseded_by != '' AND updated_at < ?",
                (cutoff_facts,),
            ).fetchall()
        ]
        if old_fact_ids:
            placeholders = ",".join("?" * len(old_fact_ids))
            cur = self.conn.execute(
                f"DELETE FROM facts WHERE id IN ({placeholders})",
                old_fact_ids,
            )
            pruned["superseded_facts"] = cur.rowcount
        else:
            pruned["superseded_facts"] = 0

        # Superseded journal entries.
        old_journal_ids = [
            r[0] for r in self.conn.execute(
                "SELECT id FROM journal WHERE superseded_by != '' AND created_at < ?",
                (cutoff_journal,),
            ).fetchall()
        ]
        if old_journal_ids:
            placeholders = ",".join("?" * len(old_journal_ids))
            cur = self.conn.execute(
                f"DELETE FROM journal WHERE id IN ({placeholders})",
                old_journal_ids,
            )
            pruned["superseded_journal"] = cur.rowcount
        else:
            pruned["superseded_journal"] = 0

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

        # Old episodes and their tool index entries (MF-009).
        cutoff_episodes = now - (episodes_days * 86400)
        old_episode_ids = [
            r[0] for r in self.conn.execute(
                "SELECT id FROM episodes WHERE created_at < ?",
                (cutoff_episodes,),
            ).fetchall()
        ]
        if old_episode_ids:
            placeholders = ",".join("?" * len(old_episode_ids))
            cur = self.conn.execute(
                f"DELETE FROM episodes WHERE id IN ({placeholders})",
                old_episode_ids,
            )
            pruned["episodes"] = cur.rowcount
        else:
            pruned["episodes"] = 0
        cur_tools = self.conn.execute(
            "DELETE FROM episode_tools WHERE episode_id NOT IN "
            "(SELECT id FROM episodes)"
        )
        pruned["episode_tools"] = cur_tools.rowcount

        # 0.6.0: prune auto-extracted "mentions" triples whose source row has
        # just been deleted. This keeps the entity graph reversible: when a fact,
        # episode, or journal entry is GC'd, its mention triples go too.
        all_deleted_source_ids = old_fact_ids + old_episode_ids + old_journal_ids
        if all_deleted_source_ids:
            placeholders = ",".join("?" * len(all_deleted_source_ids))
            cur = self.conn.execute(
                f"DELETE FROM temporal_triples WHERE predicate = 'mentions' "
                f"AND subject IN ({placeholders})",
                all_deleted_source_ids,
            )
            pruned["mention_triples_for_deleted_sources"] = cur.rowcount
        else:
            pruned["mention_triples_for_deleted_sources"] = 0

        # MV2-003/MV2-010: prune closed temporal triples, orphaned
        # provenance rows, and entities/aliases with no live triples.
        cur = self.conn.execute(
            "DELETE FROM temporal_triples WHERE valid_until IS NOT NULL "
            "AND valid_until < ?",
            (cutoff_triples,),
        )
        pruned["closed_triples"] = cur.rowcount

        cur = self.conn.execute(
            "DELETE FROM triple_provenance WHERE triple_id NOT IN "
            "(SELECT id FROM temporal_triples)"
        )
        pruned["orphan_provenance"] = cur.rowcount

        # Drop aliases and entities that no longer participate in any triple.
        cur = self.conn.execute(
            "DELETE FROM entity_aliases WHERE canonical_name NOT IN "
            "(SELECT DISTINCT subject FROM temporal_triples "
            " UNION SELECT DISTINCT object FROM temporal_triples)"
        )
        pruned["orphan_aliases"] = cur.rowcount

        cur = self.conn.execute(
            "DELETE FROM entities WHERE canonical_name NOT IN "
            "(SELECT DISTINCT subject FROM temporal_triples "
            " UNION SELECT DISTINCT object FROM temporal_triples)"
        )
        pruned["orphan_entities"] = cur.rowcount

        # 0.6.0: prune embedding cache rows that are no longer referenced by
        # any fact, journal entry, or episode under their model+text hash.
        # A cache row is referenced if there exists a row in facts/journal/
        # episodes whose embedding_model matches the cache model_name and whose
        # content hash equals the cache text_hash.  We compute the content hash
        # in Python once per source row and compare it to c.text_hash so the
        # database does not need a hash function.
        pruned["embedding_cache_unreferenced"] = self._prune_unreferenced_embedding_cache()

        # 0.6.0: enforce embedding_cache_max_mb LRU cap.  We approximate size
        # by the JSON byte length; this is good enough for lifecycle governance.
        max_mb = getattr(self.config, "embedding_cache_max_mb", 512)
        if max_mb > 0:
            max_bytes = max_mb * 1024 * 1024
            cur = self.conn.execute(
                "DELETE FROM embedding_cache WHERE cache_key IN ("
                "  SELECT cache_key FROM ("
                "    SELECT cache_key, SUM(LENGTH(vector_json)) OVER "
                "      (ORDER BY last_used_at DESC, created_at DESC) AS running_bytes"
                "    FROM embedding_cache"
                "  ) WHERE running_bytes > ?"
                ")",
                (max_bytes,),
            )
            pruned["embedding_cache_evicted_lru"] = cur.rowcount

        self.conn.commit()

        # MF-006: VACUUM is now a separate method — it needs an exclusive
        # lock and can stall active MCP turns. Callers should use
        # run_vacuum() separately when the store is idle.
        pruned["total_pruned"] = sum(
            v for k, v in pruned.items()
            if isinstance(v, int) and k not in ("db_size_before_mb", "db_size_after_mb")
        )
        return pruned

    def _prune_unreferenced_embedding_cache(self) -> int:
        """Delete embedding_cache rows no longer referenced by any fact, journal,
        or episode under the same (model_name, text_hash) pair.

        Returns the number of rows deleted.
        """
        import hashlib

        # Build the set of referenced (model_name, text_hash) pairs from the
        # three tables that store embeddings.  Contradictions are excluded
        # because they are never embedded (MF-005).
        # We stream rows directly from the cursor instead of fetchall() so
        # memory usage stays flat for large databases.
        referenced: set[tuple[str, str]] = set()
        for model, content in self.conn.execute(
            "SELECT embedding_model, content FROM facts WHERE embedding_json != ''"
        ):
            if model:
                referenced.add((model, hashlib.sha256(content.encode("utf-8")).hexdigest()))
        for model, content in self.conn.execute(
            "SELECT embedding_model, content FROM journal "
            "WHERE embedding_json != '' AND type != 'contradiction'"
        ):
            if model:
                referenced.add((model, hashlib.sha256(content.encode("utf-8")).hexdigest()))
        for model, task, summary in self.conn.execute(
            "SELECT embedding_model, task, summary FROM episodes "
            "WHERE embedding_json != '' AND is_gap_marker = 0"
        ):
            if model:
                text = f"{task}\n{summary or ''}".strip()
                referenced.add((model, hashlib.sha256(text.encode("utf-8")).hexdigest()))

        # Delete cache rows whose (model_name, text_hash) is not referenced.
        # We do this in batches to avoid a huge transaction for large caches.
        # Keyset pagination by cache_key is used instead of LIMIT/OFFSET so
        # deleting rows in one batch never causes the next batch to skip or
        # re-scan rows (and never loops infinitely when all rows are referenced).
        deleted = 0
        batch_size = 1000
        last_key = ""
        while True:
            rows = self.conn.execute(
                "SELECT cache_key, model_name, text_hash FROM embedding_cache "
                "WHERE cache_key > ? ORDER BY cache_key LIMIT ?",
                (last_key, batch_size),
            ).fetchall()
            if not rows:
                break
            to_delete = [
                key for key, model, text_hash in rows
                if (model, text_hash) not in referenced
            ]
            if to_delete:
                placeholders = ",".join("?" * len(to_delete))
                cur = self.conn.execute(
                    f"DELETE FROM embedding_cache WHERE cache_key IN ({placeholders})",
                    to_delete,
                )
                deleted += cur.rowcount
                self.conn.commit()
            last_key = rows[-1][0]
            if len(rows) < batch_size:
                break
        return deleted

    def run_vacuum(self) -> dict:
        """Reclaim disk space by rebuilding the database file.

        VACUUM needs an exclusive lock and can stall active operations.
        Run this separately from run_gc(), ideally when the store is idle
        or via explicit CLI invocation. MF-006.
        """
        old_size = self.conn.execute(
            "SELECT page_count * page_size FROM pragma_page_count(), pragma_page_size()"
        ).fetchone()[0]
        self.conn.execute("VACUUM")
        new_size = self.conn.execute(
            "SELECT page_count * page_size FROM pragma_page_count(), pragma_page_size()"
        ).fetchone()[0]
        return {
            "db_size_before_mb": round(old_size / 1024 / 1024, 1),
            "db_size_after_mb": round(new_size / 1024 / 1024, 1),
        }

    def recall_stats(self) -> dict:
        """Return recall path counters since store creation."""
        return self._recall_counters.copy()

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
            "SUM(consolidated_merged) as total_merged, "
            "MAX(created_at) as last_reflection_at "
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

    def metrics(self) -> Metrics:
        """Return a full snapshot of memory system health and diagnostics."""
        from memlife.models import Metrics

        # Aggregate counts in one pass. Only count embeddings for tables
        # that actually store embeddings; other tables just get a total.
        tables_with_embeddings = {"episodes", "facts", "journal"}
        counts: dict[str, dict[str, int]] = {}
        for r in self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall():
            name = r["name"]
            if name not in {
                "episodes", "facts", "journal", "sessions",
                "agent_runs", "temporal_triples", "entities",
            }:
                continue
            total = self.conn.execute(
                f"SELECT COUNT(*) FROM {name}"
            ).fetchone()[0]
            embedded = 0
            if name in tables_with_embeddings:
                embedded = self.conn.execute(
                    f"SELECT COUNT(*) FROM {name} WHERE embedding_json != ''"
                ).fetchone()[0]
            counts[name] = {"total": total, "embedded": embedded}

        active_facts = self.conn.execute(
            "SELECT COUNT(*) FROM facts WHERE superseded_by = ''"
        ).fetchone()[0]
        active_journal = self.conn.execute(
            "SELECT COUNT(*) FROM journal WHERE superseded_by = ''"
        ).fetchone()[0]
        contradictions = self.conn.execute(
            "SELECT COUNT(*) FROM journal WHERE superseded_by = '' AND type = 'contradiction'"
        ).fetchone()[0]
        user_corrections = self.conn.execute(
            "SELECT COUNT(*) FROM journal WHERE superseded_by = '' AND type = 'user_correction'"
        ).fetchone()[0]

        health = self.embedding_health()
        pending = sum(
            health.get(table, {}).get("missing", 0)
            for table in ("facts", "journal", "episodes")
        )

        summary = self.get_metrics_summary()
        recall = self.recall_stats()

        db_size = 0
        try:
            db_size = Path(self.db_path).stat().st_size
        except OSError:
            pass

        journal_mode = ""
        busy_timeout = 0
        try:
            journal_mode = self.conn.execute("PRAGMA journal_mode").fetchone()[0]
            busy_timeout = self.conn.execute("PRAGMA busy_timeout").fetchone()[0]
        except Exception:
            pass

        last_reflection = summary.get("last_reflection_at")
        if last_reflection is not None:
            try:
                last_reflection = float(last_reflection)
            except (TypeError, ValueError):
                last_reflection = None

        return Metrics(
            db_path=self.db_path,
            db_size_bytes=db_size,
            db_size_mb=round(db_size / 1024 / 1024, 2),
            journal_mode=str(journal_mode),
            busy_timeout_ms=int(busy_timeout),
            vector_backend=self.vector_backend.name,
            namespace=getattr(self.config, "namespace", ""),
            embedding_model=self.embedding_model_name,
            episodes=counts.get("episodes", {}).get("total", 0),
            facts=counts.get("facts", {}).get("total", 0),
            active_facts=active_facts,
            journal_entries=counts.get("journal", {}).get("total", 0),
            active_journal=active_journal,
            contradictions=contradictions,
            unresolved_contradictions=summary.get("unresolved_contradictions", 0),
            user_corrections=user_corrections,
            sessions=counts.get("sessions", {}).get("total", 0),
            agent_runs=counts.get("agent_runs", {}).get("total", 0),
            triples=counts.get("temporal_triples", {}).get("total", 0),
            entities=counts.get("entities", {}).get("total", 0),
            embedded_episodes=counts.get("episodes", {}).get("embedded", 0),
            embedded_facts=counts.get("facts", {}).get("embedded", 0),
            embedded_journal=counts.get("journal", {}).get("embedded", 0),
            pending_embeddings=pending,
            embedding_health=health,
            embedding_cache_entries=health.get("embedding_cache", {}).get("entries", 0),
            embedding_cache_bytes=health.get("embedding_cache", {}).get("vector_json_bytes", 0),
            total_reflections=summary.get("total_reflections", 0),
            last_reflection_at=last_reflection,
            avg_keep_rate=summary.get("avg_keep_rate"),
            avg_confidence=summary.get("avg_confidence"),
            total_observations_kept=summary.get("total_obs_kept", 0),
            total_hypotheses_kept=summary.get("total_hyp_kept", 0),
            total_revisions_kept=summary.get("total_rev_kept", 0),
            total_contradictions_found=summary.get("total_contradictions", 0),
            total_retired=summary.get("total_retired", 0),
            total_merged=summary.get("total_merged", 0),
            recall=recall,
            migration=self.migration_status(),
        )

