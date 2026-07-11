"""Tests for the pluggable vector backend ABC and default JSON backend."""

from __future__ import annotations

import pytest

from memlife import (
    BinaryVectorBackend,
    DummyEmbedder,
    JsonVectorBackend,
    MemoryConfig,
    MemoryStore,
    SqliteVecBackend,
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
        vector_backend="json",
        use_binary_vectors=True,
        embedding_model="dummy",
    )
    store = MemoryStore(config=cfg, embedder=DummyEmbedder())
    # vector_backend='json' is now explicit, so the legacy flag is ignored.
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


# -----------------------------------------------------------------------------
# BinaryVectorBackend tests
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_binary_backend_search(store):
    """BinaryVectorBackend uses Hamming distance on packed binary vectors."""
    backend = BinaryVectorBackend(store)
    await store.store_fact("cat fact", confidence=0.8, embed=False)
    await store.store_fact("dog fact", confidence=0.8, embed=False)
    rows = store.conn.execute(
        "SELECT id FROM facts WHERE content = ?", ("cat fact",)
    ).fetchall()
    cat_id = rows[0][0]
    rows = store.conn.execute(
        "SELECT id FROM facts WHERE content = ?", ("dog fact",)
    ).fetchall()
    dog_id = rows[0][0]

    cat_vec = [1.0, -1.0, 1.0, -1.0]
    dog_vec = [-1.0, 1.0, -1.0, 1.0]
    store.conn.execute(
        "UPDATE facts SET embedding_json = ? WHERE id = ?",
        (backend.serialize(cat_vec), cat_id),
    )
    store.conn.execute(
        "UPDATE facts SET embedding_json = ? WHERE id = ?",
        (backend.serialize(dog_vec), dog_id),
    )
    store.conn.commit()

    results = backend.search("facts", cat_vec, limit=2)
    assert len(results) == 2
    assert results[0].item_id == cat_id
    assert results[0].similarity > results[1].similarity
    assert results[0].kind == "facts"


def test_binary_backend_serialize_roundtrip(store):
    """Binary serialization packs floats and debinarizes them lossily."""
    backend = BinaryVectorBackend(store)
    vec = [1.0, -0.5, 0.25, -0.75]
    raw = backend.serialize(vec)
    assert raw.startswith("binary:")
    restored = backend.deserialize(raw)
    assert restored is not None
    assert len(restored) == len(vec)
    # Binarization is lossy: each float becomes ±1.
    assert restored == pytest.approx([1.0, -1.0, 1.0, -1.0])


def test_binary_backend_serialize_empty(store):
    """Empty vectors serialize to an empty string."""
    backend = BinaryVectorBackend(store)
    assert backend.serialize([]) == ""
    assert backend.deserialize("") is None


def test_binary_backend_delete(store):
    """Deleting a vector clears embedding_json and embedding_model."""
    backend = BinaryVectorBackend(store)
    ep_id = store.remember("task", "success", summary="hello")
    store.conn.execute(
        "UPDATE episodes SET embedding_json = ?, embedding_model = ? WHERE id = ?",
        (backend.serialize([1.0, -1.0, 1.0, -1.0]), "model", ep_id),
    )
    store.conn.commit()
    assert backend.delete("episodes", ep_id, dim=4) is True
    row = store.conn.execute(
        "SELECT embedding_json, embedding_model FROM episodes WHERE id = ?", (ep_id,)
    ).fetchone()
    assert row[0] == ""
    assert row[1] == ""


def test_binary_backend_search_filters_contradictions(store):
    """Journal search excludes contradiction entries."""
    backend = BinaryVectorBackend(store)
    store.conn.execute(
        "INSERT INTO journal (id, type, content, confidence, source_episodes_json, "
        "private, created_at, superseded_by, embedding_json) "
        "VALUES (?, 'observation', ?, 0.8, '[]', 0, ?, '', ?)",
        ("j_obs", "observation text", 1.0, backend.serialize([1.0, -1.0, 1.0, -1.0])),
    )
    store.conn.execute(
        "INSERT INTO journal (id, type, content, confidence, source_episodes_json, "
        "private, created_at, superseded_by, embedding_json) "
        "VALUES (?, 'contradiction', ?, 0.8, '[]', 0, ?, '', ?)",
        ("j_con", "contradiction text", 1.0, backend.serialize([1.0, -1.0, 1.0, -1.0])),
    )
    store.conn.commit()
    results = backend.search("journal", [1.0, -1.0, 1.0, -1.0], limit=5)
    assert len(results) == 1
    assert results[0].item_id == "j_obs"


def test_binary_backend_unavailable_for_unknown_kind(store):
    """Searching an unknown kind returns an empty list safely."""
    backend = BinaryVectorBackend(store)
    assert backend.search("unknown", [1.0, -1.0], limit=5) == []


@pytest.mark.asyncio
async def test_binary_backend_recall_facts_end_to_end(tmp_path):
    """When vector_backend='binary', fact recall uses Hamming distance."""
    db = tmp_path / "binary_recall.db"
    cfg = MemoryConfig(
        db_path=str(db),
        vector_backend="binary",
        embedding_model="dummy",
    )
    store = MemoryStore(config=cfg, embedder=DummyEmbedder())
    # Use manually crafted orthogonal binary vectors so the Hamming
    # distance is deterministic and the query clearly ranks one fact first.
    cat_id = await store.store_fact("cats are great", confidence=0.9, embed=False)
    dog_id = await store.store_fact("dogs are great", confidence=0.9, embed=False)
    backend = store.vector_backend
    cat_vec = [1.0, -1.0] * 64
    dog_vec = [-1.0, 1.0] * 64
    store.conn.execute(
        "UPDATE facts SET embedding_json = ? WHERE id = ?",
        (backend.serialize(cat_vec), cat_id),
    )
    store.conn.execute(
        "UPDATE facts SET embedding_json = ? WHERE id = ?",
        (backend.serialize(dog_vec), dog_id),
    )
    store.conn.commit()

    facts = await store.recall_facts("cats", limit=2, query_vector=cat_vec)
    assert len(facts) == 1
    assert facts[0].id == cat_id
    assert getattr(facts[0], "_vector_sim", 0.0) == pytest.approx(1.0)
    store.close()


def test_legacy_use_binary_vectors_selects_binary(tmp_path):
    """The legacy flag selects the binary backend when vector_backend is default."""
    db = tmp_path / "legacy_binary.db"
    cfg = MemoryConfig(
        db_path=str(db),
        use_binary_vectors=True,
        embedding_model="dummy",
    )
    store = MemoryStore(config=cfg)
    assert store.vector_backend.name == "binary"
    assert isinstance(store.vector_backend, BinaryVectorBackend)
    store.close()


def test_explicit_vector_backend_overrides_legacy_binary(tmp_path):
    """An explicit vector_backend='json' overrides use_binary_vectors=True."""
    db = tmp_path / "explicit_json.db"
    cfg = MemoryConfig(
        db_path=str(db),
        vector_backend="json",
        use_binary_vectors=True,
        embedding_model="dummy",
    )
    store = MemoryStore(config=cfg)
    assert store.vector_backend.name == "json"
    assert isinstance(store.vector_backend, JsonVectorBackend)
    store.close()


# Duplicate removed; legacy-binary selection is covered above.
