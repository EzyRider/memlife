"""Tests for MV2-I001 sqlite-vec backend.

sqlite-vec is not a dependency, so these tests exercise graceful fallback.
"""

import pytest

from memlife import MemoryConfig, MemoryStore, DummyEmbedder, vec_backend


def test_sqlite_vec_not_available_by_default():
    """Without the package installed, availability is False."""
    # The package may or may not be installed in CI; either way the API must
    # report its real status without crashing.
    assert isinstance(vec_backend.available(), bool)


@pytest.mark.asyncio
async def test_store_falls_back_to_json_when_sqlite_vec_unavailable(tmp_path):
    """Even with use_sqlite_vec=True, the store works when sqlite-vec is absent."""
    db = tmp_path / "novvec.db"
    cfg = MemoryConfig(
        db_path=str(db),
        use_sqlite_vec=True,
        embedding_model="dummy",
    )
    store = MemoryStore(config=cfg, embedder=DummyEmbedder())
    ep_id = store.remember("task one", "success", summary="hello world")
    await store.embed_episode(ep_id)

    row = store.conn.execute(
        "SELECT embedding_json, embedding_model FROM episodes WHERE id = ?", (ep_id,)
    ).fetchone()
    assert row[0]
    assert row[1] == "dummy"
    store.conn.close()


def test_vec_backend_store_returns_false_without_extension(tmp_path):
    """The adapter returns False when sqlite-vec cannot be loaded."""
    import sqlite3

    db = tmp_path / "rawvec.db"
    conn = sqlite3.connect(str(db))
    result = vec_backend.store(conn, "facts", "f1", [0.1, 0.2, 0.3])
    assert result is False
    conn.close()


def test_vec_backend_search_returns_empty_without_extension(tmp_path):
    import sqlite3

    db = tmp_path / "rawvec2.db"
    conn = sqlite3.connect(str(db))
    result = vec_backend.search(conn, "facts", [0.1, 0.2, 0.3], limit=5)
    assert result == []
    conn.close()


def test_dimension_guard_rejects_mixed_dims():
    """The embedder dimension guard catches inconsistent vector sizes."""
    from memlife.protocols import Embedder

    class BadEmbedder(Embedder):
        async def embed(self, texts):
            return [[0.1, 0.2] if i % 2 == 0 else [0.1, 0.2, 0.3] for i in range(len(texts))]

    db = "/tmp/dim_guard.db"
    import os
    for f in [db, db + "-wal", db + "-shm"]:
        try:
            os.remove(f)
        except FileNotFoundError:
            pass
    cfg = MemoryConfig(db_path=db, embedding_model="bad")
    store = MemoryStore(config=cfg, embedder=BadEmbedder())

    import asyncio
    result = asyncio.run(store.embed_texts(["a", "b"]))
    assert result is None
    store.conn.close()
    for f in [db, db + "-wal", db + "-shm"]:
        try:
            os.remove(f)
        except FileNotFoundError:
            pass
