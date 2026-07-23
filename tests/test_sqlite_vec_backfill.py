"""Tests for sqlite-vec backfill from JSON embeddings.

These require sqlite-vec to be installed and an interpreter that supports
SQLite extension loading. They are skipped otherwise.
"""

from __future__ import annotations

import sqlite3

import pytest

from memlife import DummyEmbedder, MemoryConfig, MemoryStore, vec_backend


def _extension_loading_available() -> bool:
    try:
        # Match the store's driver selection: pysqlite3 may support
        # extensions when the stdlib sqlite3 module does not.
        import pysqlite3.dbapi2 as pysqlite3_sqlite3  # type: ignore[import-not-found]

        conn = pysqlite3_sqlite3.connect(":memory:")
    except Exception:
        conn = sqlite3.connect(":memory:")
    return hasattr(conn, "enable_load_extension") and vec_backend.available()


@pytest.mark.asyncio
async def test_sqlite_vec_backfill_facts(tmp_path):
    if not _extension_loading_available():
        pytest.skip("sqlite-vec or extension loading not available")
    db = tmp_path / "sv_bf_fact.db"

    # Phase 1: create store with sqlite-vec disabled so embeddings land in JSON.
    cfg = MemoryConfig(
        db_path=str(db),
        use_sqlite_vec=False,
        embedding_model="dummy",
    )
    store = MemoryStore(config=cfg, embedder=DummyEmbedder())
    fid = await store.store_fact("hello world fact", confidence=0.8)
    store.close()

    # Phase 2: reopen with sqlite-vec enabled and backfill.
    cfg2 = MemoryConfig(
        db_path=str(db),
        use_sqlite_vec=True,
        embedding_model="other",
    )
    store2 = MemoryStore(config=cfg2, embedder=DummyEmbedder())
    result = await store2.backfill_embeddings(batch_size=20)
    assert result["facts_embedded"] == 1

    q = (await store2.embed_texts(["hello world"]))[0]
    facts = await store2.recall_facts("", query_vector=q, limit=5)
    assert [f.id for f in facts] == [fid]
    store2.close()


@pytest.mark.asyncio
async def test_sqlite_vec_backfill_journal(tmp_path):
    if not _extension_loading_available():
        pytest.skip("sqlite-vec or extension loading not available")
    db = tmp_path / "sv_bf_jrn.db"
    cfg = MemoryConfig(
        db_path=str(db),
        use_sqlite_vec=False,
        embedding_model="dummy",
    )
    store = MemoryStore(config=cfg, embedder=DummyEmbedder())
    jid = store.add_journal_entry("observation", "hello world observation")
    await store.embed_journal_entry(jid)
    store.close()

    cfg2 = MemoryConfig(
        db_path=str(db),
        use_sqlite_vec=True,
        embedding_model="other",
    )
    store2 = MemoryStore(config=cfg2, embedder=DummyEmbedder())
    result = await store2.backfill_embeddings(batch_size=20)
    assert result["journal_embedded"] == 1

    q = (await store2.embed_texts(["hello world"]))[0]
    entries = await store2.recall_journal_vector(q, limit=5)
    assert [e.id for e in entries] == [jid]
    store2.close()


@pytest.mark.asyncio
async def test_sqlite_vec_backfill_episodes(tmp_path):
    if not _extension_loading_available():
        pytest.skip("sqlite-vec or extension loading not available")
    db = tmp_path / "sv_bf_ep.db"
    cfg = MemoryConfig(
        db_path=str(db),
        use_sqlite_vec=False,
        embedding_model="dummy",
    )
    store = MemoryStore(config=cfg, embedder=DummyEmbedder())
    ep_id = store.remember("hello world episode", "success", summary="greeting")
    await store.embed_episode(ep_id)
    store.close()

    cfg2 = MemoryConfig(
        db_path=str(db),
        use_sqlite_vec=True,
        embedding_model="other",
    )
    store2 = MemoryStore(config=cfg2, embedder=DummyEmbedder())
    result = await store2.backfill_embeddings(batch_size=20)
    assert result["episodes_embedded"] == 1

    q = (await store2.embed_texts(["hello world"]))[0]
    eps = await store2.recall_episodes_vector(q, limit=5)
    assert [e.id for e in eps] == [ep_id]
    store2.close()
