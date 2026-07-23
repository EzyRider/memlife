"""Tests for the embedding cache (memlife 0.6.0)."""

from __future__ import annotations

import hashlib

import pytest

from memlife import DummyEmbedder, MemoryConfig, MemoryStore


@pytest.fixture
def store(tmp_path):
    cfg = MemoryConfig(
        db_path=str(tmp_path / "cache.db"),
        embedding_model="dummy",
        embedding_cache_enabled=True,
        embedding_cache_max_mb=1,
    )
    s = MemoryStore(config=cfg, embedder=DummyEmbedder())
    yield s
    s.close()


@pytest.mark.asyncio
async def test_cache_stores_vectors_after_first_embed(store):
    text = "User prefers dark mode for all interfaces"
    vecs = await store.embed_texts([text])
    assert vecs and len(vecs) == 1
    assert vecs[0]

    health = store.embedding_health()
    assert health["embedding_cache"]["enabled"] is True
    assert health["embedding_cache"]["entries"] == 1


@pytest.mark.asyncio
async def test_cache_hit_avoids_embedder_call(store, monkeypatch):
    text = "James likes concise answers"
    first = await store.embed_texts([text])
    assert first and first[0]

    # Monkeypatch embedder to blow up if called again.
    async def boom(texts):
        raise AssertionError("embedder should not be called on cache hit")

    monkeypatch.setattr(store.embedder, "embed", boom)

    second = await store.embed_texts([text])
    assert second and second[0] == first[0]


@pytest.mark.asyncio
async def test_cache_disabled_bypasses_cache(store, monkeypatch):
    cfg = MemoryConfig(
        db_path=store.db_path,
        embedding_model="dummy",
        embedding_cache_enabled=False,
    )
    # Re-open with cache disabled by creating a new store pointing at same DB.
    no_cache = MemoryStore(config=cfg, embedder=DummyEmbedder())
    text = "A unique phrase for disabled cache"
    calls = []

    async def capture(texts):
        calls.append(texts)
        return await DummyEmbedder().embed(texts)

    monkeypatch.setattr(no_cache.embedder, "embed", capture)

    await no_cache.embed_texts([text])
    await no_cache.embed_texts([text])
    assert len(calls) == 2
    no_cache.close()


@pytest.mark.asyncio
async def test_cache_model_isolation(store):
    text = "Model isolation test"
    await store.embed_texts([text])

    cfg2 = MemoryConfig(
        db_path=store.db_path,
        embedding_model="other-model",
        embedding_cache_enabled=True,
    )
    store2 = MemoryStore(config=cfg2, embedder=DummyEmbedder())
    await store2.embed_texts([text])

    rows = store2.conn.execute(
        "SELECT COUNT(*) FROM embedding_cache"
    ).fetchone()[0]
    assert rows == 2
    store2.close()


@pytest.mark.asyncio
async def test_backfill_primes_cache(store):
    ep_id = store.remember(task="User asked about caching", outcome="success")
    # Store a fact without embedding so backfill has work to do.
    fact_id = await store.store_fact(
        "Embedding caches save API calls", confidence=0.8, embed=False
    )

    result = await store.backfill_embeddings()
    assert result["failed"] == 0
    assert result["facts_embedded"] >= 1
    assert result["episodes_embedded"] >= 1

    health = store.embedding_health()
    assert health["embedding_cache"]["entries"] >= 2


@pytest.mark.asyncio
async def test_gc_sweeps_unreferenced_cache_entries(store):
    text = "Soon to be orphaned"
    await store.embed_texts([text])
    assert store.embedding_health()["embedding_cache"]["entries"] == 1

    # No fact/episode/journal references this text, so GC should drop it.
    pruned = store.run_gc(
        superseded_facts_days=0,
        superseded_journal_days=0,
        completed_runs_days=0,
        metrics_days=0,
        reflected_queue_days=0,
        episodes_days=0,
    )
    assert pruned.get("embedding_cache_unreferenced", 0) == 1
    assert store.embedding_health()["embedding_cache"]["entries"] == 0


@pytest.mark.asyncio
async def test_gc_keeps_referenced_cache_entries(store):
    text = "Referenced cache entry"
    await store.store_fact(text, confidence=0.8)

    pruned = store.run_gc(
        superseded_facts_days=0,
        superseded_journal_days=0,
        completed_runs_days=0,
        metrics_days=0,
        reflected_queue_days=0,
        episodes_days=0,
    )
    assert pruned.get("embedding_cache_unreferenced", 0) == 0
    assert store.embedding_health()["embedding_cache"]["entries"] == 1


def test_cache_table_appears_in_migration_status(store):
    status = store.migration_status()
    assert "embedding_cache" in status["missing_tables"] or status["healthy"]
    assert "embedding_cache" not in status["missing_tables"]


def test_config_rejects_negative_cache_size():
    cfg = MemoryConfig(embedding_cache_max_mb=-1)
    with pytest.raises(ValueError, match="embedding_cache_max_mb"):
        cfg.validate()


@pytest.mark.asyncio
async def test_cache_entries_are_canonical_floats(store):
    text = "Canonical float storage"
    await store.embed_texts([text])

    row = store.conn.execute(
        "SELECT model_name, text_hash, vector_json FROM embedding_cache"
    ).fetchone()
    assert row[0] == "dummy"
    assert row[1] == hashlib.sha256(text.encode("utf-8")).hexdigest()
    vec = __import__("json").loads(row[2])
    assert isinstance(vec, list)
    assert all(isinstance(x, (int, float)) for x in vec)


def test_gc_does_not_loop_when_first_batch_is_all_referenced(store):
    """Regression for the 0.6.4 audit: _prune_unreferenced_embedding_cache
    used LIMIT without an offset/keyset, so when the first batch_size rows
    were all referenced it fetched the same rows forever.
    """
    import signal
    import time

    model = store.embedding_model_name or "dummy"
    now = time.time()
    n = 1001  # larger than the 1000-row batch size

    for i in range(n):
        text = f"referenced cache row {i:04d}"
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        cache_key = f"{model}:{text_hash}"
        store.conn.execute(
            "INSERT OR IGNORE INTO embedding_cache "
            "(cache_key, model_name, text_hash, vector_json, created_at, last_used_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (cache_key, model, text_hash, "[1.0, 2.0]", now, now),
        )
        store.conn.execute(
            "INSERT INTO facts (id, content, source, confidence, created_at, "
            "updated_at, embedding_model, embedding_json, superseded_by) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (f"fact_{i}", text, "agent", 0.8, now, now, model, "[1.0, 2.0]", ""),
        )
    store.conn.commit()

    def _timeout(_signum, _frame):
        raise AssertionError("run_gc hung on all-referenced embedding cache rows")

    old_handler = signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(5)
    try:
        pruned = store.run_gc(
            superseded_facts_days=0,
            superseded_journal_days=0,
            completed_runs_days=0,
            metrics_days=0,
            reflected_queue_days=0,
            episodes_days=0,
        )
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    assert pruned.get("embedding_cache_unreferenced", 0) == 0
    assert store.embedding_health()["embedding_cache"]["entries"] == n
