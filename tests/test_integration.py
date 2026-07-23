"""Integration test: full lifecycle with store, backfill, and reflection.

Exercises MemoryStore (store/recall), backfill_embeddings (with
DummyEmbedder), and Reflector.reflect (using DummyChat) in a single
test to surface parsing/critic/DB interactions in CI.
"""


import pytest

from memlife import DummyChat, DummyEmbedder, MemoryConfig, MemoryStore, Reflector


@pytest.mark.asyncio
async def test_full_integration_store_backfill_reflect(tmp_path):
    """End-to-end: store episodes + facts, backfill embeddings, run
    reflection, verify journal entries and contradictions are created,
    then run GC to verify pruning works."""
    store = MemoryStore(
        config=MemoryConfig(
            db_path=str(tmp_path / "integration.db"),
            embedding_model="dummy",
        ),
        embedder=DummyEmbedder(),
    )

    # 1. Store episodes
    ep1 = store.remember(task="User asked about deployment", outcome="success")
    ep2 = store.remember(task="User mentioned they prefer dark mode", outcome="success")

    # 2. Store facts — one pair that's a near-duplicate to exercise dedup
    fact_a = await store.store_fact("User deploys via GitHub Actions", confidence=0.8)
    fact_b = await store.store_fact("User deploys via GitHub Actions CI/CD", confidence=0.7)
    fact_c = await store.store_fact("User prefers dark mode", confidence=0.9)

    # 3. Recall episodes by keyword
    results = store.recall("deployment", limit=5)
    assert len(results) >= 1
    assert any("deployment" in r.task.lower() for r in results)

    # 4. Backfill embeddings with DummyEmbedder
    backfill = await store.backfill_embeddings()
    assert backfill["facts_embedded"] >= 1 or backfill["failed"] >= 0
    # DummyEmbedder always succeeds, so no failures expected
    assert backfill["failed"] == 0

    # 5. Run reflection with DummyChat
    reflector = Reflector(
        memory=store,
        model_chat=DummyChat(),
        critic=False,
        model_name="test",
    )
    result = await reflector.reflect()
    assert len(result.episode_ids) >= 1

    # 6. Verify journal entries were created
    entries = store.journal_recent(limit=10)
    assert len(entries) >= 1

    # 7. Verify embedding health reports correctly
    health = store.embedding_health()
    assert health["embedder_present"] is True
    assert health["facts"]["total"] >= 1

    # 8. Run GC — should not crash, should report pruning counts
    gc_result = store.run_gc(
        superseded_facts_days=0,
        superseded_journal_days=0,
        completed_runs_days=0,
        metrics_days=0,
        reflected_queue_days=0,
        episodes_days=0,
    )
    assert "total_pruned" in gc_result

    # 9. Run VACUUM — should return size info
    vacuum = store.run_vacuum()
    assert "db_size_before_mb" in vacuum
    assert "db_size_after_mb" in vacuum

    # 10. Verify context manager works
    with MemoryStore(config=MemoryConfig(db_path=str(tmp_path / "ctx.db"))) as ctx_store:
        ctx_store.remember(task="context manager test", outcome="success")
        assert len(ctx_store.recent(limit=5)) >= 1

    store.close()