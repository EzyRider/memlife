"""Tests for MV2-004 annotations / veracity flags."""

import time

import pytest

from memlife import MemoryConfig, MemoryStore, retrieve


@pytest.fixture
def anno_store(tmp_path):
    db = tmp_path / "anno.db"
    cfg = MemoryConfig(
        db_path=str(db),
        recall_vector_weight=0.0,
        recall_text_weight=1.0,
        recall_source_weight=0.0,
        recall_veracity_weight=0.0,
    )
    store = MemoryStore(cfg)
    try:
        yield store
    finally:
        store.conn.close()


@pytest.mark.asyncio
async def test_annotate_fact_persists(anno_store):
    store = anno_store
    now = time.time()
    store.conn.execute(
        "INSERT INTO facts (id, content, source, confidence, embedding_json, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("fact_1", "Ottawa is the capital of Canada", "agent", 0.8, "", now, now),
    )
    store.conn.commit()

    assert store.annotate_fact("fact_1", "confirmed") is True
    assert store.annotate_fact("fact_1", "confirmed", dedupe=True) is False
    assert store.annotate_fact("fact_1", "confirmed", dedupe=False) is True

    assert store.fact_by_id("fact_1").annotations == ["confirmed", "confirmed"]


@pytest.mark.asyncio
async def test_annotate_journal_persists(anno_store):
    store = anno_store
    store.add_journal_entry(
        content="It rained heavily last Tuesday",
        type="observation",
        confidence=0.9,
    )
    rows = store.conn.execute(
        "SELECT id FROM journal WHERE type = 'observation'"
    ).fetchall()
    jid = rows[0][0]

    assert store.annotate_journal(jid, "verified") is True
    assert store.annotate_journal(jid, "verified") is False

    entry = store.journal_recent(limit=1)[0]
    assert entry.annotations == ["verified"]


@pytest.mark.asyncio
async def test_annotations_appear_in_debug_candidates(anno_store):
    store = anno_store
    now = time.time()
    store.conn.execute(
        "INSERT INTO facts (id, content, source, confidence, embedding_json, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("fact_2", "User prefers metric units", "agent", 0.7, "", now, now),
    )
    store.conn.commit()
    store.annotate_fact("fact_2", "confirmed")

    result = await retrieve(store, "metric units", debug=True)
    candidates = result["candidates"]  # type: ignore[index]
    fact = next(c for c in candidates if c["id"] == "fact_2")
    assert fact["annotations"] == ["confirmed"]


@pytest.mark.asyncio
async def test_annotate_missing_row_returns_false(anno_store):
    store = anno_store
    assert store.annotate_fact("missing", "confirmed") is False
    assert store.annotate_journal("missing", "verified") is False


@pytest.mark.asyncio
async def test_blank_annotation_rejected(anno_store):
    store = anno_store
    now = time.time()
    store.conn.execute(
        "INSERT INTO facts (id, content, source, confidence, embedding_json, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("fact_3", "A thing", "agent", 0.5, "", now, now),
    )
    store.conn.commit()
    assert store.annotate_fact("fact_3", "") is False
    assert store.annotate_fact("fact_3", "   ") is False
    assert store.fact_by_id("fact_3").annotations == []


def test_annotations_property_on_unset_json():
    """Models with default annotations_json '[]' return empty list."""
    from memlife.models import Fact, JournalEntry

    assert Fact(id="x", content="c").annotations == []
    assert JournalEntry(id="y", type="observation", content="c").annotations == []
