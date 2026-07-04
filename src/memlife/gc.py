"""Garbage collection for the memory store."""

from __future__ import annotations

from memlife.store import MemoryStore


def run_gc(
    store: MemoryStore,
    *,
    superseded_facts_days: int = 90,
    superseded_journal_days: int = 90,
    completed_runs_days: int = 60,
    metrics_days: int = 30,
    reflected_queue_days: int = 30,
    episodes_days: int = 180,
) -> dict:
    """Run garbage collection on old/superseded data.

    Delegates to MemoryStore.run_gc() — this wrapper is the public entry
    point. Most callers should use ``store.run_gc()`` directly.
    """
    return store.run_gc(
        superseded_facts_days=superseded_facts_days,
        superseded_journal_days=superseded_journal_days,
        completed_runs_days=completed_runs_days,
        metrics_days=metrics_days,
        reflected_queue_days=reflected_queue_days,
        episodes_days=episodes_days,
    )
