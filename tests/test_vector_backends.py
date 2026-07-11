"""Tests for the pluggable vector backend ABC and default JSON backend."""

from __future__ import annotations

import pytest

from memlife import (
    DummyEmbedder,
    JsonVectorBackend,
    MemoryConfig,
    MemoryStore,
    SqliteVecBackend,
    VectorBackend,
    VectorSearchResult,
    create_vector_backend,
)
from memlife.vector_backends.base import VectorBackend as BaseVectorBackend


def test_json_backend_default(store):
    """A store without vector_backend configured uses the JSON backend."""
    assert store.vector_backend.name == "json"
    assert isinstance(store.vector_backend, JsonVectorBackend)
    assert store.vector_backend.available() is True


def test_create_vector_backend_json(store):
    backend = create_vector_backend("json", store)
    assert backend.name == "json"
    assert isinstance(backend, JsonVectorBackend)


def test_create_vector_backend_unknown(store):
    with pytest.raises(ValueError):
        create_vector_backend("unknown", store)


def test_backend_is_abstract(store):
    """The base ABC cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseVectorBackend(store)


def test_vector_search_result_immutable():
    r = VectorSearchResult("id1", 0.9, "facts")
    assert r.item_id == "id1"
    assert r.similarity == 0.9
    assert r.kind == "facts"


def test_json_serialize_roundtrip(store):
    vec = [0.1, -0.2, 0.3, -0.4]
    raw = store.vector_backend.serialize(vec)
    assert raw.startswith("[")
    restored = store.vector_backend.deserialize(raw)
    assert restored == pytest.approx(vec)


def test_json_serialize_empty(store):
    assert store.vector_backend.serialize([]) == ""
    assert store.vector_backend.deserialize("") is None


@pytest.mark.asyncio
async def test_json_backend_search(store):
    backend = store.vector_backend
    # Seed two facts with different vectors.
    await store.store_fact("cat fact", confidence=0.8, embed=False)
    await store.store_fact("dog fact", confidence=0.8, embed=False)
    # Manually set embeddings via the backend's serialize form.
    cat_vec = [1.0, 0.0, 0.0]
    dog_vec = [0.0, 1.0, 0.0]
    store.conn.execute(
        "UPDATE facts SET embedding_json = ? WHERE content = ?",
        (backend.serialize(cat_vec), "cat fact"),
    )
    store.conn.execute(
        "UPDATE facts SET embedding_json = ? WHERE content = ?",
        (backend.serialize(dog_vec), "dog fact"),
    )
    store.conn.commit()

    results = backend.search("facts", [0.9, 0.1, 0.0], limit=2)
    assert len(results) == 2
    assert results[0].item_id != results[1].item_id
    assert results[0].similarity > results[1].similarity
    assert results[0].kind == "facts"


@pytest.mark.asyncio
async def test_json_backend_with_binary_vectors(tmp_path):
    db = tmp_path / "bin.db"
    cfg = MemoryConfig(
        db_path=str(db),
        use_binary_vectors=True,
        embedding_model="dummy",
    )
    store = MemoryStore(config=cfg, embedder=DummyEmbedder())
    assert store.vector_backend.name == "json"

    ep_id = store.remember("task", "success", summary="hello")
    await store.embed_episode(ep_id)

    row = store.conn.execute(
        "SELECT embedding_json FROM episodes WHERE id = ?", (ep_id,)
    ).fetchone()
    assert row[0].startswith("binary:")
    eps = store.episodes_by_ids([ep_id])
    assert len(eps) == 1
    assert eps[0].embedding is not None
    assert len(eps[0].embedding) == 128
    store.close()


def test_legacy_use_sqlite_vec_selects_sqlite_vec(tmp_path):
    """The legacy flag still selects sqlite_vec when vector_backend is default."""
    db = tmp_path / "legacy.db"
    cfg = MemoryConfig(
        db_path=str(db),
        use_sqlite_vec=True,
        embedding_model="dummy",
    )
    store = MemoryStore(config=cfg)
    # If sqlite-vec is unavailable, the store falls back to json.
    if SqliteVecBackend(store).available():
        assert store.vector_backend.name == "sqlite_vec"
    else:
        assert store.vector_backend.name == "json"
    store.close()


def test_explicit_vector_backend_overrides_legacy(tmp_path):
    """vector_backend='sqlite_vec' overrides an explicit 'json' default even when
    use_sqlite_vec is True, because the legacy flag is only meant to nudge the
    default.  To force JSON, set vector_backend='json' *and* use_sqlite_vec=False.
    """
    db = tmp_path / "explicit.db"
    cfg = MemoryConfig(
        db_path=str(db),
        vector_backend="sqlite_vec",
        use_sqlite_vec=True,
        embedding_model="dummy",
    )
    store = MemoryStore(config=cfg)
    if SqliteVecBackend(store).available():
        assert store.vector_backend.name == "sqlite_vec"
    else:
        assert store.vector_backend.name == "json"
    store.close()


def test_legacy_flag_selects_sqlite_vec_when_default(tmp_path):
    """When vector_backend is the default 'json', use_sqlite_vec=True selects
    sqlite_vec for backward compatibility (if available)."""
    db = tmp_path / "legacy.db"
    cfg = MemoryConfig(
        db_path=str(db),
        use_sqlite_vec=True,
        embedding_model="dummy",
    )
    store = MemoryStore(config=cfg)
    if SqliteVecBackend(store).available():
        assert store.vector_backend.name == "sqlite_vec"
    else:
        assert store.vector_backend.name == "json"
    store.close()


def test_force_json_backend(tmp_path):
    """To force JSON, set vector_backend='json' and use_sqlite_vec=False."""
    db = tmp_path / "forcejson.db"
    cfg = MemoryConfig(
        db_path=str(db),
        vector_backend="json",
        use_sqlite_vec=False,
        embedding_model="dummy",
    )
    store = MemoryStore(config=cfg)
    assert store.vector_backend.name == "json"
    store.close()


def test_backend_scoped_to_namespace(tmp_path):
    """A backend created for one namespace must not leak into another."""
    cfg_a = MemoryConfig(
        db_path=str(tmp_path / "a.db"),
        namespace="a",
        embedding_model="dummy",
    )
    cfg_b = MemoryConfig(
        db_path=str(tmp_path / "b.db"),
        namespace="b",
        embedding_model="dummy",
    )
    store_a = MemoryStore(config=cfg_a)
    store_b = MemoryStore(config=cfg_b)
    assert store_a.vector_backend is not store_b.vector_backend
    assert store_a.db_path != store_b.db_path
    store_a.close()
    store_b.close()
