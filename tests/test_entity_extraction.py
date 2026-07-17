"""Tests for deterministic automatic entity extraction (0.6.0)."""

from __future__ import annotations

import time

import pytest

from memlife import MemoryConfig, MemoryStore
from memlife.entity_extractor import DEFAULT_ALLOWLIST, extract_entities


@pytest.fixture
def store(tmp_path):
    cfg = MemoryConfig(
        db_path=str(tmp_path / "test.db"),
        auto_entity_extraction=True,
        auto_entity_mentions=True,
        auto_entity_confidence=0.6,
        embedding_model="dummy",
    )
    ms = MemoryStore(cfg)
    yield ms
    ms.close()


@pytest.fixture
def disabled_store(tmp_path):
    cfg = MemoryConfig(
        db_path=str(tmp_path / "disabled.db"),
        auto_entity_extraction=False,
        embedding_model="dummy",
    )
    ms = MemoryStore(cfg)
    yield ms
    ms.close()


def test_extract_entities_finds_proper_nouns():
    text = "James and Julie went to Melbourne with memlife."
    got = extract_entities(text)
    canonicals = {c for c, _ in got}
    assert "james" in canonicals
    assert "julie" in canonicals
    assert "melbourne" in canonicals
    assert "memlife" in canonicals


def test_extract_entities_uses_allowlist():
    text = "The quick brown fox uses openclaw and hermes."
    got = extract_entities(text, allowlist={"openclaw", "hermes"})
    canonicals = {c for c, _ in got}
    assert "openclaw" in canonicals
    assert "hermes" in canonicals


def test_extract_entities_respects_blocklist():
    text = "James went to The Store."
    got = extract_entities(text, blocklist={"store"})
    canonicals = {c for c, _ in got}
    assert "james" in canonicals
    assert "store" not in canonicals


def test_extract_entities_dedupes():
    text = "James saw James and james."
    got = extract_entities(text)
    assert len(got) == 1
    assert got[0][0] == "james"


def test_default_allowlist_contains_project_terms():
    assert "memlife" in DEFAULT_ALLOWLIST
    assert "ingrid" in DEFAULT_ALLOWLIST


@pytest.mark.asyncio
async def test_store_fact_creates_entities_and_mentions(store):
    fid = await store.store_fact("James is building memlife with Ingrid.")
    assert store.resolve_entity("james") == "james"
    assert store.resolve_entity("memlife") == "memlife"
    assert store.resolve_entity("ingrid") == "ingrid"

    # Aliases preserve casing.
    assert store.resolve_entity("Ingrid") == "ingrid"

    triples = store.triples_about("ingrid", predicate="mentions")
    assert any(t["subject"] == fid for t in triples)
    triples = store.triples_about("memlife", predicate="mentions")
    assert any(t["subject"] == fid for t in triples)
    triples = store.triples_about("james", predicate="mentions")
    assert any(t["subject"] == fid for t in triples)


@pytest.mark.asyncio
async def test_remember_creates_entities_and_mentions(store):
    ep_id = store.remember(
        task="Plan memlife 0.6.0 with James",
        outcome="success",
        summary="He sketched the entity graph.",
    )
    assert store.resolve_entity("james") == "james"
    assert store.resolve_entity("memlife") == "memlife"

    triples = store.triples_about("james", predicate="mentions")
    assert any(t["subject"] == ep_id and t["object"] == "james" for t in triples)


def test_add_journal_entry_creates_entities_and_mentions(store):
    jid = store.add_journal_entry(
        type="observation",
        content="Ingrid thinks memlife should auto-extract entities.",
    )
    assert store.resolve_entity("ingrid") == "ingrid"
    assert store.resolve_entity("memlife") == "memlife"

    triples = store.triples_about("memlife", predicate="mentions")
    assert any(t["subject"] == jid and t["object"] == "memlife" for t in triples)


@pytest.mark.asyncio
async def test_entity_extraction_disabled_by_default(disabled_store):
    await disabled_store.store_fact("James is building memlife.")
    assert disabled_store.resolve_entity("james") is None
    assert disabled_store.resolve_entity("memlife") is None


@pytest.mark.asyncio
async def test_gc_cleans_mention_triples_when_source_pruned(tmp_path):
    cfg = MemoryConfig(
        db_path=str(tmp_path / "gc.db"),
        auto_entity_extraction=True,
        auto_entity_mentions=True,
        embedding_model="dummy",
        gc_episodes_days=0,  # prune immediately
    )
    store = MemoryStore(cfg)

    # Create an episode, then make it old enough for GC.
    ep_id = store.remember(
        task="James likes memlife.",
        outcome="success",
    )
    assert store.resolve_entity("james") == "james"
    old_triples = store.triples_about("james", predicate="mentions")
    assert any(t["subject"] == ep_id for t in old_triples)

    store.conn.execute(
        "UPDATE episodes SET created_at = ? WHERE id = ?",
        (time.time() - 86400 * 365, ep_id),
    )
    store.conn.commit()

    # Run GC with zero retention.
    result = store.run_gc()
    assert result["episodes"] >= 1
    assert result.get("mention_triples_for_deleted_sources", 0) >= 1

    # The mention triples tied to the deleted episode are gone.
    remaining = store.triples_about("james", predicate="mentions")
    assert not any(t["subject"] == ep_id for t in remaining)

    store.close()


@pytest.mark.asyncio
async def test_gc_removes_orphan_entities_after_source_pruned(tmp_path):
    cfg = MemoryConfig(
        db_path=str(tmp_path / "orphan.db"),
        auto_entity_extraction=True,
        auto_entity_mentions=True,
        embedding_model="dummy",
        gc_episodes_days=0,
    )
    store = MemoryStore(cfg)

    ep_id = store.remember(
        task="Single mention of OrphanEntityX",
        outcome="success",
    )
    assert store.resolve_entity("orphanentityx") == "orphanentityx"

    # Make the episode old enough to prune.
    store.conn.execute(
        "UPDATE episodes SET created_at = ? WHERE id = ?",
        (time.time() - 86400 * 365, ep_id),
    )
    store.conn.commit()

    result = store.run_gc()
    assert result["episodes"] >= 1
    assert result.get("mention_triples_for_deleted_sources", 0) >= 1
    assert result.get("orphan_entities", 0) >= 1
    assert store.resolve_entity("orphanentityx") is None

    store.close()
