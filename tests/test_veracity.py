"""Tests for MV2-005 veracity-weighted recall."""

import time

import pytest

from memlife import MemoryConfig, MemoryStore, retrieve


@pytest.fixture
def veracity_store(tmp_path):
    db = tmp_path / "veracity.db"
    cfg = MemoryConfig(
        db_path=str(db),
        recall_vector_weight=0.0,
        recall_text_weight=1.0,
        recall_source_weight=0.0,
        recall_veracity_weight=0.5,
    )
    store = MemoryStore(cfg)
    yield store
    store.close()


@pytest.mark.asyncio
async def test_veracity_boosts_triple_supported_fact(veracity_store):
    """A fact backed by a current triple ranks above a similar low-conf fact."""
    store = veracity_store
    now = time.time()

    # Same query text-score, different dedup text so dedup keeps both.
    store.conn.execute(
        "INSERT INTO facts (id, content, source, confidence, embedding_json, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("fact_low", "User prefers dark mode", "agent", 0.3, "", now, now),
    )
    store.conn.execute(
        "INSERT INTO facts (id, content, source, confidence, embedding_json, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("fact_high", "Client prefers dark mode", "agent", 0.3, "", now, now),
    )
    store.conn.commit()
    store.store_fact_triple("fact_high", "user", "prefers", "dark mode", confidence=0.95)

    result = await retrieve(store, "dark mode preference", debug=True)
    assert isinstance(result, dict)
    candidates = result["candidates"]
    by_id = {c["id"]: c for c in candidates}
    assert "fact_high" in by_id and "fact_low" in by_id
    assert by_id["fact_high"]["score"] > by_id["fact_low"]["score"]
    assert by_id["fact_high"]["veracity"] > by_id["fact_low"]["veracity"]
    assert by_id["fact_high"]["text_score"] == pytest.approx(by_id["fact_low"]["text_score"], abs=1e-4)


@pytest.mark.asyncio
async def test_veracity_disabled_when_weight_zero(tmp_path):
    """With veracity weight zero, similar facts tie on relevance."""
    db = tmp_path / "no_veracity.db"
    cfg = MemoryConfig(
        db_path=str(db),
        recall_vector_weight=0.0,
        recall_text_weight=1.0,
        recall_source_weight=0.0,
        recall_veracity_weight=0.0,
    )
    store = MemoryStore(cfg)
    try:
        now = time.time()
        store.conn.execute(
            "INSERT INTO facts (id, content, source, confidence, embedding_json, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("fact_a", "User prefers dark mode", "agent", 0.3, "", now, now),
        )
        store.conn.execute(
            "INSERT INTO facts (id, content, source, confidence, embedding_json, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("fact_b", "Client prefers dark mode", "agent", 0.3, "", now, now),
        )
        store.conn.commit()
        store.store_fact_triple("fact_b", "user", "prefers", "dark mode", confidence=0.95)

        result = await retrieve(store, "dark mode preference", debug=True)
        candidates = result["candidates"]  # type: ignore[index]
        by_id = {c["id"]: c for c in candidates}
        assert "fact_a" in by_id and "fact_b" in by_id
        assert by_id["fact_a"]["relevance"] == pytest.approx(
            by_id["fact_b"]["relevance"], abs=1e-4
        )
        assert by_id["fact_b"]["veracity"] > by_id["fact_a"]["veracity"]
    finally:
        store.close()


@pytest.mark.asyncio
async def test_veracity_uses_current_triples_only(tmp_path):
    """Expired triples do not boost veracity."""
    db = tmp_path / "expired.db"
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
            ("fact_expired", "User prefers dark mode", "agent", 0.5, "", now, now),
        )
        store.conn.commit()
        store.store_fact_triple(
            "fact_expired", "user", "prefers", "dark mode",
            confidence=0.95, valid_until=now - 1.0,
        )

        result = await retrieve(store, "dark mode preference", debug=True)
        candidates = result["candidates"]  # type: ignore[index]
        expired = next(c for c in candidates if c["id"] == "fact_expired")
        assert expired["veracity"] == pytest.approx(0.5, abs=1e-4)
    finally:
        store.close()
