"""Tests for the unified retrieval system."""

import asyncio
import pytest

from memlife import MemoryConfig, MemoryStore, DummyEmbedder, retrieve


@pytest.mark.asyncio
async def test_retrieve_returns_formatted_context(store, config):
    """Retrieve returns structured context with sections."""
    store.remember(task="deployed the app", outcome="success", summary="v1.0 deployed")
    await store.store_fact("User deploys via GitHub Actions", confidence=0.8)
    context = await retrieve(store, "deploy", config)
    assert isinstance(context, str)
    assert len(context) > 0


@pytest.mark.asyncio
async def test_retrieve_includes_facts(store, config):
    """Retrieve includes matching facts in the context."""
    await store.store_fact("User prefers dark mode", confidence=0.9)
    context = await retrieve(store, "dark mode", config)
    assert "dark mode" in context.lower()


@pytest.mark.asyncio
async def test_retrieve_includes_episodes(store, config):
    """Retrieve includes matching episodes in the context."""
    store.remember(task="configured the server", outcome="success")
    context = await retrieve(store, "server", config)
    assert "server" in context.lower()


@pytest.mark.asyncio
async def test_retrieve_empty_store(store, config):
    """Retrieve on an empty store returns empty string."""
    context = await retrieve(store, "nothing here", config)
    assert context == ""


@pytest.mark.asyncio
async def test_retrieve_respects_max_context_chars(store, config):
    """Context is truncated when it exceeds max_context_chars."""
    config.max_context_chars = 50
    for i in range(20):
        await store.store_fact(f"Fact number {i} with some text", confidence=0.7)
    context = await retrieve(store, "fact", config)
    assert "[...context truncated]" in context
    assert len(context) <= 100  # truncated well below full output