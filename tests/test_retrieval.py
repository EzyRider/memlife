"""Tests for the unified retrieval system."""

import pytest
import time

from memlife import MemoryConfig, MemoryStore, retrieve


@pytest.mark.asyncio
async def test_retrieve_returns_formatted_context(store, config):
    """Retrieve returns structured context with sections."""
    store.remember(task="deployed the app", outcome="success", summary="v1.0 deployed")
    await store.store_fact("User deploys via GitHub Actions", confidence=0.8)
    context = await retrieve(store, "deploy", config)
    assert isinstance(context, str)
    assert len(context) > 0


@pytest.mark.asyncio
async def test_retrieve_includes_facts(store, config):
    """Retrieve includes matching facts in the context."""
    await store.store_fact("User prefers dark mode", confidence=0.9)
    context = await retrieve(store, "dark mode", config)
    assert "dark mode" in context.lower()


@pytest.mark.asyncio
async def test_retrieve_includes_episodes(store, config):
    """Retrieve includes matching episodes in the context."""
    store.remember(task="configured the server", outcome="success")
    context = await retrieve(store, "server", config)
    assert "server" in context.lower()


@pytest.mark.asyncio
async def test_retrieve_empty_store(store, config):
    """Retrieve on an empty store returns empty string."""
    context = await retrieve(store, "nothing here", config)
    assert context == ""


@pytest.mark.asyncio
async def test_retrieve_respects_max_context_chars(store, config):
    """Context is truncated when it exceeds max_context_chars."""
    config.max_context_chars = 50
    for i in range(20):
        await store.store_fact(f"Fact number {i} with some text", confidence=0.7)
    context = await retrieve(store, "fact", config)
    assert "[...context truncated]" in context
    assert len(context) <= 100  # truncated well below full output


@pytest.mark.asyncio
async def test_retrieve_debug_returns_candidate_breakdown(store, config):
    """With debug=True retrieve returns a dict of context + candidates."""
    await store.store_fact("User prefers dark mode", confidence=0.9)
    result = await retrieve(store, "dark mode", config, debug=True)
    assert isinstance(result, dict)
    assert "context" in result
    assert "candidates" in result
    assert len(result["candidates"]) >= 1
    first = result["candidates"][0]
    for key in ("kind", "id", "vector_sim", "text_score", "source_weight",
                "confidence", "recency", "relevance", "score", "text"):
        assert key in first


@pytest.mark.asyncio
async def test_retrieve_text_only_mode(temp_db):
    """Without an embedder, retrieval still works using text signals."""
    config = MemoryConfig(db_path=temp_db)
    store = MemoryStore(config=config)
    try:
        await store.store_fact("User prefers dark mode", confidence=0.9)
        store.remember(task="configured the server", outcome="success")
        context = await retrieve(store, "dark mode", config)
        assert "dark mode" in context.lower()
    finally:
        store.close()


@pytest.mark.asyncio
async def test_retrieve_vector_only_mode(store, config):
    """With text weight zero, ranking is driven by vector similarity."""
    config.recall_vector_weight = 1.0
    config.recall_text_weight = 0.0
    config.recall_source_weight = 0.0
    await store.store_fact("User prefers dark mode", confidence=0.9)
    result = await retrieve(store, "dark mode", config, debug=True)
    assert isinstance(result, dict)
    assert len(result["candidates"]) >= 1
    for c in result["candidates"]:
        assert c["relevance"] <= 1.0
        assert c["score"] <= 1.0


@pytest.mark.asyncio
async def test_retrieve_blended_ranking_prefers_matching_layer(store, config):
    """A fact that matches the query should outrank a generic episode."""
    store.remember(task="bought coffee", outcome="success", summary="routine morning task")
    await store.store_fact("User prefers dark mode", confidence=0.9)
    result = await retrieve(store, "dark mode", config, debug=True)
    facts = [c for c in result["candidates"] if c["kind"] == "fact"]
    episodes = [c for c in result["candidates"] if c["kind"] == "episode"]
    if facts and episodes:
        assert facts[0]["score"] >= episodes[0]["score"]


@pytest.mark.asyncio
async def test_retrieve_layer_aware_decay(store, config):
    """Facts decay slower than episodes under layer-aware halflifes."""
    config.fact_decay_halflife_days = 365.0
    config.episode_decay_halflife_days = 1.0
    # Two equally relevant items, one fact and one episode, both old.
    old_ts = time.time() - 10 * 86400
    store.remember(task="old routine task", outcome="success")
    await store.store_fact("old important fact", confidence=0.9)
    # Patch timestamps in DB directly since remember/store_fact default to now.
    with store.transaction():
        store.conn.execute("UPDATE episodes SET created_at = ? WHERE task = ?", (old_ts, "old routine task"))
        store.conn.execute("UPDATE facts SET created_at = ?, updated_at = ? WHERE content = ?", (old_ts, old_ts, "old important fact"))
    result = await retrieve(store, "old important fact", config, debug=True)
    facts = [c for c in result["candidates"] if c["kind"] == "fact"]
    episodes = [c for c in result["candidates"] if c["kind"] == "episode"]
    if facts and episodes:
        assert facts[0]["recency"] > episodes[0]["recency"]
