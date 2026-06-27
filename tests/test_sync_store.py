"""Tests for the SyncMemoryStore wrapper."""

import pytest

from memlife import SyncMemoryStore, MemoryConfig, DummyEmbedder


@pytest.fixture
def sync_store(tmp_path):
    """A SyncMemoryStore with DummyEmbedder."""
    config = MemoryConfig(
        db_path=str(tmp_path / "test.db"),
        embedding_model="dummy",
    )
    store = SyncMemoryStore(config=config, embedder=DummyEmbedder())
    yield store
    store.close()


def test_sync_store_remember(sync_store):
    """SyncMemoryStore.remember works without async."""
    ep_id = sync_store.remember(task="test", outcome="success")
    assert ep_id.startswith("ep_")
    episodes = sync_store.recent(limit=5)
    assert len(episodes) >= 1


def test_sync_store_store_fact(sync_store):
    """SyncMemoryStore.store_fact works without explicit async."""
    fact_id = sync_store.store_fact("Test fact", confidence=0.7)
    assert fact_id.startswith("fact_")
    facts = sync_store.recall_facts("test", limit=5)
    assert len(facts) >= 1


def test_sync_store_retrieve(sync_store):
    """SyncMemoryStore.retrieve works without explicit async."""
    sync_store.remember(task="deployed the app", outcome="success")
    sync_store.store_fact("User deploys via CI/CD", confidence=0.8)
    context = sync_store.retrieve("deploy")
    assert isinstance(context, str)
    assert len(context) > 0


def test_sync_store_run_gc(sync_store):
    """SyncMemoryStore.run_gc works (it's sync already)."""
    result = sync_store.run_gc()
    assert result["total_pruned"] == 0


def test_sync_store_search_by_tool(sync_store):
    """SyncMemoryStore.search_episodes_by_tool works."""
    sync_store.remember(
        task="ran a command",
        outcome="success",
        tool_calls=[{"tool": "read_file", "params": {}}],
    )
    eps = sync_store.search_episodes_by_tool("read_file", limit=5)
    assert len(eps) >= 1


def test_sync_store_embedding_health(sync_store):
    """SyncMemoryStore.embedding_health works."""
    sync_store.store_fact("Test fact", confidence=0.7)
    health = sync_store.embedding_health()
    assert health["facts"]["with_embeddings"] >= 1