"""Tests for MemoryStore.metrics() snapshot."""

from __future__ import annotations

import pytest

from memlife import MemoryConfig, MemoryStore, Metrics, SyncMemoryStore


pytestmark = pytest.mark.anyio


@pytest.fixture
def store(tmp_path):
    return MemoryStore(
        config=MemoryConfig(
            db_path=str(tmp_path / "mem.db"),
            embedding_model="test-model",
        )
    )


async def test_metrics_empty_store(store):
    m = store.metrics()
    assert isinstance(m, Metrics)
    assert m.episodes == 0
    assert m.facts == 0
    assert m.active_facts == 0
    assert m.journal_entries == 0
    assert m.active_journal == 0
    assert m.unresolved_contradictions == 0
    assert m.total_reflections == 0
    assert m.pending_embeddings == 0
    assert m.recall == store.recall_stats()
    assert m.vector_backend == "json"
    assert m.namespace == "_default"
    assert m.embedding_model == "test-model"


async def test_metrics_counts_after_writes(store):
    store.remember("task one", "success", summary="summary")
    await store.store_fact("sky is blue", confidence=0.8, embed=False)
    await store.store_fact("grass is green", confidence=0.7, embed=False)
    store.add_journal_entry("observation", "something happened", 0.6)

    m = store.metrics()
    assert m.episodes == 1
    assert m.facts == 2
    assert m.active_facts == 2
    assert m.journal_entries == 1
    assert m.active_journal == 1


async def test_metrics_reflection_aggregates(store):
    store.record_reflection_metrics(
        {
            "episodes_considered": 10,
            "observations_proposed": 4,
            "observations_kept": 3,
            "hypotheses_proposed": 2,
            "hypotheses_kept": 1,
            "revisions_proposed": 1,
            "revisions_kept": 1,
            "contradictions_found": 2,
            "avg_confidence": 0.75,
            "keep_rate": 0.6,
            "consolidated_retired": 1,
            "consolidated_merged": 1,
            "total_journal_entries": 5,
            "total_facts": 7,
            "total_episodes": 12,
        }
    )
    m = store.metrics()
    assert m.total_reflections == 1
    assert m.total_observations_kept == 3
    assert m.total_hypotheses_kept == 1
    assert m.total_revisions_kept == 1
    assert m.total_contradictions_found == 2
    assert m.total_retired == 1
    assert m.total_merged == 1
    assert m.avg_keep_rate == pytest.approx(0.6)
    assert m.avg_confidence == pytest.approx(0.75)
    assert m.last_reflection_at is not None


async def test_metrics_pending_embeddings(store):
    store.remember("task", "success")
    await store.store_fact("fact without embedding", embed=False)
    store.add_journal_entry("observation", "journal without embedding")

    m = store.metrics()
    assert m.pending_embeddings == 3
    assert m.embedded_episodes == 0
    assert m.embedded_facts == 0
    assert m.embedded_journal == 0


async def test_metrics_includes_recall_counters(store):
    before = store.recall_stats()
    store._recall_counters["retrieve_calls"] += 1
    store._recall_counters["facts_considered"] += 5
    m = store.metrics()
    assert m.recall["retrieve_calls"] == before["retrieve_calls"] + 1
    assert m.recall["facts_considered"] == before["facts_considered"] + 5


def test_sync_memory_store_metrics(tmp_path):
    store = SyncMemoryStore(
        config=MemoryConfig(db_path=str(tmp_path / "sync.db"))
    )
    store.remember("sync task", "success")
    m = store.metrics()
    assert isinstance(m, Metrics)
    assert m.episodes == 1


def test_migration_status_healthy_empty_db(tmp_path):
    store = MemoryStore(
        config=MemoryConfig(db_path=str(tmp_path / "migrate.db"))
    )
    status = store.migration_status()
    assert status["healthy"] is True
    assert status["missing_tables"] == []
    assert status["missing_columns"] == []
    assert status["tables_expected"] == status["tables_present"]
    store.close()


def test_migration_status_sync_wrapper(tmp_path):
    store = SyncMemoryStore(
        config=MemoryConfig(db_path=str(tmp_path / "sync_migrate.db"))
    )
    status = store.migration_status()
    assert status["healthy"] is True
    assert status["missing_tables"] == []
    assert status["missing_columns"] == []
    store.close()
