"""Tests for MV2-I002 binary vector compression."""

import pytest

from memlife import MemoryConfig, MemoryStore, DummyEmbedder
from memlife.binary_vectors import (
    binarize,
    debinarize,
    hamming_distance,
    hamming_similarity,
    cosine_from_binary,
)


def test_binarize_roundtrip():
    vec = [0.5, -0.2, 0.0, -1.0, 0.9, -0.1, 0.3, -0.8]
    packed = binarize(vec)
    assert len(packed) == 1
    reconstructed = debinarize(packed, len(vec))
    assert reconstructed == [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]


def test_hamming_distance_identical():
    a = binarize([0.1] * 8)
    b = binarize([0.2] * 8)
    assert hamming_distance(a, b) == 0


def test_hamming_distance_opposite():
    a = binarize([0.1] * 8)
    b = binarize([-0.1] * 8)
    assert hamming_distance(a, b) == 8


def test_hamming_similarity():
    a = binarize([0.1] * 8)
    b = binarize([0.1] * 4 + [-0.1] * 4)
    assert hamming_similarity(a, b, 8) == 0.5


def test_cosine_from_binary():
    a = binarize([0.1] * 8)
    b = binarize([0.1] * 4 + [-0.1] * 4)
    assert cosine_from_binary(a, b, 8) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.asyncio
async def test_binary_vector_storage(tmp_path):
    db = tmp_path / "binvec.db"
    cfg = MemoryConfig(
        db_path=str(db),
        use_binary_vectors=True,
        embedding_model="dummy",
    )
    store = MemoryStore(config=cfg, embedder=DummyEmbedder())
    ep_id = store.remember("binary task", "success", summary="hello world")
    await store.embed_episode(ep_id)

    row = store.conn.execute(
        "SELECT embedding_json FROM episodes WHERE id = ?", (ep_id,)
    ).fetchone()
    assert row[0].startswith("binary:")
    eps = store.episodes_by_ids([ep_id])
    assert len(eps) == 1
    assert eps[0].embedding is not None
    assert len(eps[0].embedding) == 128
    store.conn.close()


def test_binary_vectors_smaller_than_json():
    vec = [0.1 if i % 2 == 0 else -0.1 for i in range(128)]
    packed = binarize(vec)
    json_form = str(vec)
    assert len(packed) < len(json_form)
