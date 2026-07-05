"""Tests for MV2-009 journal belief network."""

import pytest

from memlife import MemoryConfig, MemoryStore, retrieve


@pytest.fixture
def belief_store(tmp_path):
    db = tmp_path / "belief.db"
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


def test_link_journal_entries(belief_store):
    store = belief_store
    a = store.add_journal_entry(content="It rained on Tuesday", type="observation", confidence=0.9)
    b = store.add_journal_entry(content="It rained on Wednesday", type="observation", confidence=0.8)

    assert store.link_journal_entries(a, b, "related", strength=0.7) is True
    entries = store.journal_by_type("observation", limit=2)
    entry = next(e for e in entries if e.id == a)
    assert entry.links == [{"target": b, "relation": "related", "strength": 0.7}]


def test_link_replaces_existing_to_same_target(belief_store):
    store = belief_store
    a = store.add_journal_entry(content="A", type="observation", confidence=0.9)
    b = store.add_journal_entry(content="B", type="observation", confidence=0.8)

    store.link_journal_entries(a, b, "supports", strength=0.5)
    store.link_journal_entries(a, b, "undermines", strength=0.9)

    entry = next(e for e in store.journal_by_type("observation", limit=2) if e.id == a)
    assert len(entry.links) == 1
    assert entry.links[0]["relation"] == "undermines"
    assert entry.links[0]["strength"] == pytest.approx(0.9)


def test_link_missing_from_returns_false(belief_store):
    store = belief_store
    b = store.add_journal_entry(content="B", type="observation", confidence=0.8)
    assert store.link_journal_entries("missing", b, "supports") is False


def test_link_self_rejected(belief_store):
    store = belief_store
    a = store.add_journal_entry(content="A", type="observation", confidence=0.9)
    assert store.link_journal_entries(a, a, "related") is False


def test_link_unsupported_relation_raises(belief_store):
    store = belief_store
    a = store.add_journal_entry(content="A", type="observation", confidence=0.9)
    b = store.add_journal_entry(content="B", type="observation", confidence=0.8)
    with pytest.raises(ValueError, match="unsupported relation"):
        store.link_journal_entries(a, b, "likes")


def test_link_strength_clamped(belief_store):
    store = belief_store
    a = store.add_journal_entry(content="A", type="observation", confidence=0.9)
    b = store.add_journal_entry(content="B", type="observation", confidence=0.8)
    store.link_journal_entries(a, b, "supports", strength=1.5)
    entry = next(e for e in store.journal_by_type("observation", limit=2) if e.id == a)
    assert entry.links[0]["strength"] == pytest.approx(1.0)

    store.link_journal_entries(a, b, "supports", strength=-0.3)
    entry = next(e for e in store.journal_by_type("observation", limit=2) if e.id == a)
    assert entry.links[0]["strength"] == pytest.approx(0.0)


def test_bidirectional_link(belief_store):
    store = belief_store
    a = store.add_journal_entry(content="A", type="observation", confidence=0.9)
    b = store.add_journal_entry(content="B", type="observation", confidence=0.8)
    store.link_journal_entries(a, b, "supports", bidirectional=True)

    a_entry = next(e for e in store.journal_by_type("observation", limit=2) if e.id == a)
    b_entry = next(e for e in store.journal_by_type("observation", limit=2) if e.id == b)
    assert a_entry.links == [{"target": b, "relation": "supports", "strength": 1.0}]
    assert b_entry.links == [{"target": a, "relation": "related", "strength": 1.0}]


@pytest.mark.asyncio
async def test_links_appear_in_debug_candidates(belief_store):
    store = belief_store
    a = store.add_journal_entry(content="User likes spicy food", type="observation", confidence=0.9)
    b = store.add_journal_entry(content="User ordered hot wings", type="observation", confidence=0.8)
    store.link_journal_entries(a, b, "supports")

    result = await retrieve(store, "spicy food", debug=True)
    candidates = result["candidates"]  # type: ignore[index]
    entry = next(c for c in candidates if c["id"] == a)
    assert entry["links"] == [{"target": b, "relation": "supports", "strength": 1.0}]


def test_default_links_empty():
    from memlife.models import JournalEntry

    entry = JournalEntry(id="x", type="observation", content="c")
    assert entry.links == []
