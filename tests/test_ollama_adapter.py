"""Tests for the Ollama adapter.

These are integration tests — they require a running Ollama instance.
Skipped automatically if Ollama is not reachable.
"""

import os

import pytest

from memlife import MemoryConfig, MemoryStore
from memlife.adapters.ollama import OllamaChat, OllamaEmbedder


def _ollama_reachable():
    """Check if Ollama is running locally."""
    import urllib.request
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _ollama_reachable(),
    reason="Ollama not reachable at localhost:11434",
)


@pytest.mark.asyncio
async def test_ollama_embedder_embeds_text():
    """OllamaEmbedder produces vectors."""
    embedder = OllamaEmbedder(model="mxbai-embed-large:latest")
    try:
        vecs = await embedder.embed(["hello world"])
        assert vecs is not None
        assert len(vecs) == 1
        assert len(vecs[0]) > 0
    finally:
        await embedder.close()


@pytest.mark.asyncio
async def test_ollama_embedder_embeds_batch():
    """OllamaEmbedder handles batch embedding."""
    embedder = OllamaEmbedder(model="mxbai-embed-large:latest")
    try:
        vecs = await embedder.embed(["hello", "world", "test"])
        assert vecs is not None
        assert len(vecs) == 3
    finally:
        await embedder.close()


@pytest.mark.asyncio
async def test_ollama_embedder_returns_none_on_bad_model():
    """OllamaEmbedder returns None (not raises) for a nonexistent model."""
    embedder = OllamaEmbedder(model="nonexistent-model:latest")
    try:
        vecs = await embedder.embed(["test"])
        assert vecs is None
    finally:
        await embedder.close()


@pytest.mark.asyncio
async def test_ollama_chat_returns_text():
    """OllamaChat returns text content from the model."""
    chat = OllamaChat(
        model=os.getenv("INGRID_MODEL", "qwen3.5:cloud"),
        max_retries=1,
        timeout=30.0,
    )
    try:
        response = await chat.chat(
            [{"role": "user", "content": "Say 'hello' and nothing else."}],
            model=os.getenv("INGRID_MODEL", "qwen3.5:cloud"),
        )
        assert isinstance(response, str)
        assert len(response) > 0
    finally:
        await chat.close()


@pytest.mark.asyncio
async def test_ollama_embedder_with_memlife_store(tmp_path):
    """Full integration: OllamaEmbedder works with MemoryStore."""
    config = MemoryConfig(
        db_path=str(tmp_path / "test.db"),
        embedding_model="mxbai-embed-large:latest",
    )
    embedder = OllamaEmbedder(model="mxbai-embed-large:latest")
    store = MemoryStore(config=config, embedder=embedder)
    try:
        # Store a fact with real embeddings
        fact_id = await store.store_fact(
            "The user deploys via GitHub Actions",
            confidence=0.8,
        )
        assert fact_id.startswith("fact_")

        # Retrieve with a real query
        context = await store.retrieve("deployment")
        assert "GitHub Actions" in context or "deploys" in context

        # Check embedding health
        health = store.embedding_health()
        assert health["facts"]["with_embeddings"] >= 1
        assert health["embedding_model"] == "mxbai-embed-large:latest"
    finally:
        store.close()
        await embedder.close()