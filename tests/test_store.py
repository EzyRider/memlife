"""Tests for the core memory store."""

import pytest

from memlife import MemoryStore, DummyEmbedder


@pytest.mark.asyncio
async def test_store_and_retrieve_episode(store):
    """An episode can be stored and recalled."""
    ep_id = store.remember(
        task="User asked about deployment",
        outcome="success",
        summary="Explained deploy process",
    )
    assert ep_id.startswith("ep_")
    episodes = store.recent(limit=5)
    assert len(episodes) >= 1
    assert episodes[0].task == "User asked about deployment"


@pytest.mark.asyncio
async def test_store_and_retrieve_fact(store):
    """A fact can be stored and recalled."""
    fact_id = await store.store_fact(
        "User prefers pytest",
        confidence=0.8,
    )
    assert fact_id.startswith("fact_")
    facts = await store.recall_facts("pytest", limit=5)
    assert len(facts) >= 1
    assert any("pytest" in f.content for f in facts)


@pytest.mark.asyncio
async def test_fact_dedup_exact(store):
    """Storing the same fact twice returns the same ID."""
    id1 = await store.store_fact("User likes coffee", confidence=0.7)
    id2 = await store.store_fact("User likes coffee", confidence=0.7)
    assert id1 == id2


@pytest.mark.asyncio
async def test_fact_confidence_ceiling(store):
    """Confidence is capped below 1.0."""
    fact_id = await store.store_fact("Test fact", confidence=1.0)
    fact = store.fact_by_id(fact_id)
    assert fact.confidence < 1.0
    assert fact.confidence <= 0.99


@pytest.mark.asyncio
async def test_episode_tool_index(store):
    """Tool calls are indexed when an episode is stored."""
    store.remember(
        task="ran a command",
        outcome="success",
        tool_calls=[
            {"tool": "read_file", "params": {}},
            {"tool": "run_shell", "params": {}},
        ],
    )
    eps = store.search_episodes_by_tool("read_file", limit=5)
    assert len(eps) >= 1
    eps = store.search_episodes_by_tool("run_shell", limit=5)
    assert len(eps) >= 1


@pytest.mark.asyncio
async def test_search_episodes_by_keyword(store):
    """Keyword search finds episodes by task text."""
    store.remember(task="deployed to production", outcome="success")
    store.remember(task="fixed a bug", outcome="success")
    results = store.search_episodes_by_keyword("production", limit=5)
    assert len(results) >= 1
    assert any("production" in ep.task for ep in results)


@pytest.mark.asyncio
async def test_gc_prunes_nothing_when_empty(store):
    """GC on a fresh store prunes nothing."""
    result = store.run_gc()
    assert result["total_pruned"] == 0


@pytest.mark.asyncio
async def test_gc_prunes_superseded(store):
    """GC prunes superseded facts and journal entries."""
    # Create a superseded fact
    old_id = await store.store_fact("Old fact", confidence=0.5)
    new_id = await store.revise_fact(old_id, "Updated fact", confidence=0.8)
    assert old_id != new_id
    # The old fact should be superseded
    old = store.fact_by_id(old_id)
    assert old.superseded_by != ""
    # Run GC with 0-day retention to prune immediately
    result = store.run_gc(superseded_facts_days=0)
    assert result["superseded_facts"] >= 1


@pytest.mark.asyncio
async def test_embedding_health(store):
    """Embedding health reports correct counts."""
    await store.store_fact("Test fact with embedding", confidence=0.7)
    health = store.embedding_health()
    assert health["facts"]["total"] >= 1
    assert health["facts"]["with_embeddings"] >= 1
    assert health["embedding_model"] == "dummy"


@pytest.mark.asyncio
async def test_embedding_versioning_detects_stale(store):
    """Embedding health detects stale vectors when model changes."""
    await store.store_fact("Fact with dummy model", confidence=0.7)
    # Change the model name
    store.embedding_model_name = "different-model"
    health = store.embedding_health()
    assert health["facts"]["stale"] >= 1


@pytest.mark.asyncio
async def test_import_export_jsonl(store, tmp_path):
    """Export and import round-trip works."""
    store.remember(task="test episode", outcome="success")
    await store.store_fact("test fact", confidence=0.7)
    export_path = str(tmp_path / "export.jsonl")
    from memlife import MemoryConfig, export_jsonl, import_jsonl
    result = export_jsonl(store, export_path)
    assert result["episodes"] >= 1
    assert result["facts"] >= 1
    # Import into a new store
    new_db = str(tmp_path / "import.db")
    new_store = MemoryStore(config=MemoryConfig(db_path=new_db), embedder=DummyEmbedder())
    import_result = import_jsonl(new_store, export_path)
    assert import_result["episodes"] >= 1
    assert import_result["facts"] >= 1
    # Verify data
    eps = new_store.recent(limit=5)
    assert any("test episode" in e.task for e in eps)
    new_store.close()