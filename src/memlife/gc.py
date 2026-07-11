"""Garbage collection for the memory store."""

from __future__ import annotations

from memlife.store import MemoryStore


def run_gc(
    store: MemoryStore,
    *,
    superseded_facts_days: int | None = None,
    superseded_journal_days: int | None = None,
    completed_runs_days: int | None = None,
    metrics_days: int | None = None,
    reflected_queue_days: int | None = None,
    episodes_days: int | None = None,
    closed_triples_days: int | None = None,
) -> dict:
    """Run garbage collection on old/superseded data.

    Delegates to MemoryStore.run_gc(). When parameters are None, falls
    back to MemoryConfig values on the store (MF-016: was using hardcoded
    defaults that ignored config). Most callers should use
    ``store.run_gc()`` directly.
    """
    cfg = store.config
    return store.run_gc(
        superseded_facts_days=superseded_facts_days if superseded_facts_days is not None else cfg.gc_superseded_facts_days,
        superseded_journal_days=superseded_journal_days if superseded_journal_days is not None else cfg.gc_superseded_journal_days,
        completed_runs_days=completed_runs_days if completed_runs_days is not None else cfg.gc_completed_runs_days,
        metrics_days=metrics_days if metrics_days is not None else cfg.gc_metrics_days,
        reflected_queue_days=reflected_queue_days if reflected_queue_days is not None else cfg.gc_reflected_queue_days,
        episodes_days=episodes_days if episodes_days is not None else cfg.gc_episodes_days,
        closed_triples_days=closed_triples_days if closed_triples_days is not None else cfg.gc_closed_triples_days,
    )
