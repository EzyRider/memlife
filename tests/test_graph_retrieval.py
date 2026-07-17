"""Regression tests for graph-integrated retrieval (0.6.2).

These tests exercise the fixes that did not make it into 0.6.1:
  * mention-triple graph boost
  * outgoing and incoming relationship traversal
  * case-insensitive entity canonicalisation for manual triples
  * superseded facts are filtered from graph candidates
  * closed (expired) relationship triples are not followed
  * SyncMemoryStore.retrieve supports debug output
"""

import time

import pytest

from memlife import MemoryConfig, MemoryStore, DummyEmbedder


@pytest.fixture
def graph_store(tmp_path):
    cfg = MemoryConfig(
        db_path=str(tmp_path / "graph_retrieval.db"),
        use_graph_retrieval=True,
        graph_retrieval_weight=0.5,
        recall_vector_weight=0.0,
        recall_text_weight=1.0,
        recall_source_weight=0.0,
        recall_veracity_weight=0.0,
        embedding_model="dummy",
    )
    store = MemoryStore(config=cfg, embedder=DummyEmbedder())
    yield store
    store.close()


@pytest.mark.asyncio
async def test_mention_triple_boosts_episode(graph_store):
    """An episode linked to a queried entity via a mention triple is boosted."""
    store = graph_store
    ep_id = store.remember(task="Met with James about the roadmap", outcome="success")
    store.store_mention_triple("episode", ep_id, "James", confidence=0.7)

    result = await store.retrieve("James", debug=True)
    assert isinstance(result, dict)
    by_id = {c["id"]: c for c in result["candidates"]}
    assert ep_id in by_id
    assert by_id[ep_id]["graph_signal"] > 0.0
    assert by_id[ep_id]["graph_triples"]
    assert any(t["predicate"] == "mentions" for t in by_id[ep_id]["graph_triples"])


@pytest.mark.asyncio
async def test_outgoing_relationship_traversal(graph_store):
    """Alice -> knows -> Bob -> mentions -> episode surfaces when querying Alice."""
    store = graph_store
    store.store_triple("Alice", "knows", "Bob", confidence=0.9)
    ep_id = store.remember(task="Bob reviewed the patch", outcome="success")
    store.store_mention_triple("episode", ep_id, "Bob", confidence=0.6)

    result = await store.retrieve("Alice", debug=True)
    by_id = {c["id"]: c for c in result["candidates"]}
    assert ep_id in by_id
    assert by_id[ep_id]["graph_signal"] > 0.0


@pytest.mark.asyncio
async def test_incoming_relationship_traversal(graph_store):
    """Alice -> knows -> Bob; episode mentions Alice; querying Bob finds it."""
    store = graph_store
    store.store_triple("Alice", "knows", "Bob", confidence=0.9)
    ep_id = store.remember(task="Alice wrote the design doc", outcome="success")
    store.store_mention_triple("episode", ep_id, "Alice", confidence=0.6)

    result = await store.retrieve("Bob", debug=True)
    by_id = {c["id"]: c for c in result["candidates"]}
    assert ep_id in by_id
    assert by_id[ep_id]["graph_signal"] > 0.0


@pytest.mark.asyncio
async def test_superseded_fact_not_graph_boosted(graph_store):
    """A superseded fact stops contributing graph candidates."""
    store = graph_store
    fact_id = await store.store_fact("James prefers light mode", confidence=0.8)
    store.store_mention_triple("fact", fact_id, "James", confidence=0.7)
    await store.revise_fact(fact_id, "James prefers dark mode", confidence=0.9)

    result = await store.retrieve("James", debug=True)
    by_id = {c["id"]: c for c in result["candidates"] if c["kind"] == "fact"}
    assert fact_id not in by_id


@pytest.mark.asyncio
async def test_closed_relationship_not_followed(graph_store):
    """An expired relationship edge is ignored during graph expansion."""
    store = graph_store
    triple_id = store.store_triple("Alice", "knows", "Bob", confidence=0.9)
    ep_id = store.remember(task="Bob reviewed the patch", outcome="success")
    store.store_mention_triple("episode", ep_id, "Bob", confidence=0.6)

    # Close the relationship triple.
    store.conn.execute(
        "UPDATE temporal_triples SET valid_until = ? WHERE id = ?",
        (time.time(), triple_id),
    )
    store.conn.commit()

    result = await store.retrieve("Alice", debug=True)
    by_id = {c["id"]: c for c in result["candidates"]}
    assert ep_id not in by_id or by_id[ep_id]["graph_signal"] == 0.0


def test_manual_triple_reuses_auto_canonical_entity(graph_store):
    """Manual store_triple with original casing reuses a lowercase canonical."""
    store = graph_store
    # Simulate auto-extraction: canonical lowercase entity with an alias.
    store.add_entity_alias("james", "James")

    # Manual triple uses the original casing.
    tid = store.store_triple("James", "works_at", "Acme", confidence=0.8)

    # The triple should be stored against the canonical lowercase entity.
    triples = store.triples_about("james")
    assert any(t["id"] == tid and t["subject"] == "james" for t in triples)
    # No separate "James" canonical entity was created.
    assert store.resolve_entity("James") == "james"


def test_sync_store_retrieve_debug(tmp_path):
    """SyncMemoryStore.retrieve supports the debug flag."""
    from memlife import SyncMemoryStore

    cfg = MemoryConfig(
        db_path=str(tmp_path / "sync_debug.db"),
        use_graph_retrieval=True,
        embedding_model="dummy",
    )
    sync = SyncMemoryStore(config=cfg, embedder=DummyEmbedder())
    sync.remember(task="deployed the app", outcome="success")
    result = sync.retrieve("deploy", debug=True)
    assert isinstance(result, dict)
    assert "candidates" in result
    sync.close()
