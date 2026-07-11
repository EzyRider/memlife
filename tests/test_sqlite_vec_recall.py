"""Tests for sqlite-vec-backed vector recall.

These require sqlite-vec to be installed and an interpreter that supports
SQLite extension loading. They are skipped otherwise.
"""

from __future__ import annotations

import sqlite3

import pytest

from memlife import MemoryConfig, MemoryStore, DummyEmbedder, vec_backend


def _extension_loading_available() -> bool:
    try:
        conn = sqlite3.connect(":memory:")
        return hasattr(conn, "enable_load_extension") and vec_backend.available()
    except Exception:
        return False


@pytest.fixture
def sqlite_vec_enabled() -> bool:
    return _extension_loading_available()


@pytest.mark.asyncio
async def test_sqlite_vec_episode_recall(tmp_path):
    if not _extension_loading_available():
        pytest.skip("sqlite-vec or extension loading not available")
    db = tmp_path / "sv_ep.db"
    cfg = MemoryConfig(
        db_path=str(db),
        use_sqlite_vec=True,
        embedding_model="dummy",
    )
    store = MemoryStore(config=cfg, embedder=DummyEmbedder())
    ep_id = store.remember("hello world episode", "success", summary="greeting")
    await store.embed_episode(ep_id)

    q = (await store.embed_texts(["hello world"]))[0]
    eps = await store.recall_episodes_vector(q, limit=5)
    assert len(eps) == 1
    assert eps[0].id == ep_id
    assert getattr(eps[0], "_score", 0) > 0
    store.close()


@pytest.mark.asyncio
async def test_sqlite_vec_fact_recall(tmp_path):
    if not _extension_loading_available():
        pytest.skip("sqlite-vec or extension loading not available")
    db = tmp_path / "sv_fact.db"
    cfg = MemoryConfig(
        db_path=str(db),
        use_sqlite_vec=True,
        embedding_model="dummy",
    )
    store = MemoryStore(config=cfg, embedder=DummyEmbedder())
    fid = await store.store_fact("hello world fact", confidence=0.8)

    q = (await store.embed_texts(["hello world"]))[0]
    facts = await store.recall_facts("", query_vector=q, limit=5)
    assert len(facts) == 1
    assert facts[0].id == fid
    assert getattr(facts[0], "_score", 0) > 0
    store.close()


@pytest.mark.asyncio
async def test_sqlite_vec_journal_recall(tmp_path):
    if not _extension_loading_available():
        pytest.skip("sqlite-vec or extension loading not available")
    db = tmp_path / "sv_jrn.db"
    cfg = MemoryConfig(
        db_path=str(db),
        use_sqlite_vec=True,
        embedding_model="dummy",
    )
    store = MemoryStore(config=cfg, embedder=DummyEmbedder())
    jid = store.add_journal_entry("observation", "hello world observation")
    await store.embed_journal_entry(jid)

    q = (await store.embed_texts(["hello world"]))[0]
    entries = await store.recall_journal_vector(q, limit=5)
    assert len(entries) == 1
    assert entries[0].id == jid
    assert getattr(entries[0], "_score", 0) > 0
    store.close()


@pytest.mark.asyncio
async def test_sqlite_vec_kinds_are_isolated(tmp_path):
    if not _extension_loading_available():
        pytest.skip("sqlite-vec or extension loading not available")
    db = tmp_path / "sv_isolated.db"
    cfg = MemoryConfig(
        db_path=str(db),
        use_sqlite_vec=True,
        embedding_model="dummy",
    )
    store = MemoryStore(config=cfg, embedder=DummyEmbedder())
    ep_id = store.remember("hello world episode", "success", summary="greeting")
    await store.embed_episode(ep_id)
    fid = await store.store_fact("hello world fact", confidence=0.8)

    q = (await store.embed_texts(["hello world"]))[0]
    facts = await store.recall_facts("", query_vector=q, limit=5)
    eps = await store.recall_episodes_vector(q, limit=5)
    assert [f.id for f in facts] == [fid]
    assert [e.id for e in eps] == [ep_id]
    store.close()
