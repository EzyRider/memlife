"""Tests for MV2-006 recall diagnostics."""

import time

import pytest

from memlife import MemoryConfig, MemoryStore, retrieve


@pytest.fixture
def diag_store(tmp_path):
    db = tmp_path / "diag.db"
    cfg = MemoryConfig(
        db_path=str(db),
        recall_vector_weight=0.0,
        recall_text_weight=1.0,
        recall_source_weight=0.0,
        recall_veracity_weight=0.0,
    )
    store = MemoryStore(cfg)
    yield store
    store.close()


@pytest.mark.asyncio
async def test_debug_includes_why_field(diag_store):
    """Every selected candidate in debug mode carries a non-empty why."""
    store = diag_store
    now = time.time()
    store.conn.execute(
        "INSERT INTO facts (id, content, source, confidence, embedding_json, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("f1", "User likes pizza", "user", 0.9, "", now, now),
    )
    store.conn.commit()

    result = await retrieve(store, "pizza", debug=True)
    assert isinstance(result, dict)
    candidates = result["candidates"]
    assert candidates
    for c in candidates:
        assert "why" in c
        assert c["why"]
        assert isinstance(c["why"], str)


@pytest.mark.asyncio
async def test_why_reflects_high_confidence_fact(diag_store):
    """A high-confidence fact explains itself as high confidence + relevant."""
    store = diag_store
    now = time.time()
    store.conn.execute(
        "INSERT INTO facts (id, content, source, confidence, embedding_json, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("f1", "User likes pizza", "user", 0.9, "", now, now),
    )
    store.conn.commit()

    result = await retrieve(store, "pizza", debug=True)
    c = result["candidates"][0]  # type: ignore[index]
    assert "high confidence" in c["why"]


@pytest.mark.asyncio
async def test_why_mentions_veracity_when_enabled(tmp_path):
    """With veracity weight on, a supported fact is marked as well supported."""
    db = tmp_path / "diag_veracity.db"
    cfg = MemoryConfig(
        db_path=str(db),
        recall_vector_weight=0.0,
        recall_text_weight=1.0,
        recall_source_weight=0.0,
        recall_veracity_weight=1.0,
    )
    store = MemoryStore(cfg)
    try:
        now = time.time()
        store.conn.execute(
            "INSERT INTO facts (id, content, source, confidence, embedding_json, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("f1", "User likes pizza", "user", 0.5, "", now, now),
        )
        store.conn.commit()
        store.store_fact_triple("f1", "user", "likes", "pizza", confidence=0.95)

        result = await retrieve(store, "pizza", debug=True)
        c = result["candidates"][0]  # type: ignore[index]
        assert "well supported" in c["why"]
    finally:
        store.close()


@pytest.mark.asyncio
async def test_why_says_old_for_aged_episode(diag_store):
    """An old episode is diagnosed as old."""
    store = diag_store
    old = time.time() - 100 * 86400
    store.conn.execute(
        "INSERT INTO episodes (id, task, outcome, summary, tool_calls_json, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("ep_old", "something happened", "success", "", "[]", old),
    )
    store.conn.commit()

    result = await retrieve(store, "something happened", debug=True)
    by_id = {c["id"]: c for c in result["candidates"]}  # type: ignore[index]
    assert "ep_old" in by_id
    assert "old" in by_id["ep_old"]["why"]
