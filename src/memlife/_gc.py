"""Prune stale memory rows and expose diagnostics.

Extracted from store.py as part of the mixin refactor.
"""

from __future__ import annotations

import logging
import time
import uuid


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

        # Old episodes and their tool index entries (MF-009).
        cutoff_episodes = now - (episodes_days * 86400)
        cur = self.conn.execute(
            "DELETE FROM episodes WHERE created_at < ?",
            (cutoff_episodes,),
        )
        pruned["episodes"] = cur.rowcount
        cur_tools = self.conn.execute(
            "DELETE FROM episode_tools WHERE episode_id NOT IN "
            "(SELECT id FROM episodes)"
        )
        pruned["episode_tools"] = cur_tools.rowcount

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

        self.conn.commit()

        # MF-006: VACUUM is now a separate method — it needs an exclusive
        # lock and can stall active MCP turns. Callers should use
        # run_vacuum() separately when the store is idle.
        pruned["total_pruned"] = sum(
            v for k, v in pruned.items()
            if isinstance(v, int) and k not in ("db_size_before_mb", "db_size_after_mb")
        )
        return pruned

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

