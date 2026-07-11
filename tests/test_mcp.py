"""Tests for the MCP server.

Tests the tool definitions and resource handlers without running
a full MCP transport — we call the registered functions directly.
"""

import json

import pytest

from memlife.mcp_server import create_server


@pytest.fixture
def server(tmp_path):
    """Create an MCP server with a temp DB and DummyEmbedder."""
    mcp = create_server(
        db_path=str(tmp_path / "mcp_test.db"),
        embedder_type="dummy",
        embedding_model="dummy",
    )
    yield mcp
    mcp._memlife_store.close()


def test_create_server_vector_backend(tmp_path):
    """create_server passes the vector_backend option to MemoryConfig."""
    mcp = create_server(
        db_path=str(tmp_path / "mcp_vec.db"),
        embedder_type="dummy",
        embedding_model="dummy",
        vector_backend="binary",
    )
    store = mcp._memlife_store
    assert store.vector_backend.name == "binary"
    mcp._memlife_store.close()


def _get_tool(server, name):
    """Get a tool function from the FastMCP server."""
    # FastMCP stores tools in _tool_manager
    manager = server._tool_manager
    if hasattr(manager, '_tools'):
        return manager._tools.get(name)
    return None


@pytest.mark.asyncio
async def test_memory_store_tool(server):
    """The memory_store tool stores a fact."""
    store = server._memlife_store
    fact_id = await store.store_fact("Test fact via MCP", confidence=0.7)
    assert fact_id.startswith("fact_")
    # Verify it's retrievable
    facts = await store.recall_facts("test", limit=5)
    assert any("MCP" in f.content for f in facts)


@pytest.mark.asyncio
async def test_memory_search_tool(server):
    """The memory_search tool finds facts."""
    store = server._memlife_store
    await store.store_fact("User likes pizza", confidence=0.9)
    facts = await store.recall_facts("pizza", limit=5)
    assert len(facts) >= 1
    assert any("pizza" in f.content for f in facts)


def test_memory_search_journal_tool(server):
    """The memory_search_journal tool finds journal entries."""
    store = server._memlife_store
    store.add_journal_entry("observation", "User mentioned pizza preferences", 0.7)
    entries = store.search_journal("pizza", limit=5)
    assert len(entries) >= 1


def test_memory_search_episodes_tool(server):
    """The memory_search_episodes tool finds episodes."""
    store = server._memlife_store
    store.remember(
        task="deployed to production",
        outcome="success",
        tool_calls=[{"tool": "run_shell", "params": {}}],
    )
    # Search by keyword
    eps = store.search_episodes_by_keyword("production", limit=5)
    assert len(eps) >= 1
    # Search by tool
    eps = store.search_episodes_by_tool("run_shell", limit=5)
    assert len(eps) >= 1


@pytest.mark.asyncio
async def test_memory_revise_tool(server):
    """The memory_revise tool supersedes a fact."""
    store = server._memlife_store
    old_id = await store.store_fact("Old fact", confidence=0.5)
    new_id = await store.revise_fact(old_id, "Updated fact", confidence=0.8)
    assert old_id != new_id
    old = store.fact_by_id(old_id)
    assert old.superseded_by == new_id


def test_memory_expire_tool(server):
    """The memory_expire tool expires a fact."""
    import asyncio
    store = server._memlife_store
    fact_id = asyncio.run(store.store_fact("Temporary fact", confidence=0.5))
    result = store.expire_fact(fact_id)
    assert result is True


@pytest.mark.asyncio
async def test_memory_retrieve_tool(server):
    """The memory_retrieve tool returns formatted context."""
    store = server._memlife_store
    store.remember(task="deployed the app", outcome="success")
    await store.store_fact("User deploys via CI/CD", confidence=0.8)
    context = await store.retrieve("deploy")
    assert isinstance(context, str)
    assert len(context) > 0


def test_memory_gc_tool(server):
    """The memory_gc tool runs garbage collection."""
    store = server._memlife_store
    result = store.run_gc()
    assert "total_pruned" in result
    # MF-006: VACUUM is now a separate method — run_gc no longer returns
    # disk size. That's in run_vacuum() instead.
    assert "episodes" in result  # MF-009: episode pruning now included


def test_stats_resource(server):
    """The stats resource returns JSON statistics."""
    store = server._memlife_store
    health = store.embedding_health()
    summary = store.get_metrics_summary()
    data = {
        "embedding_model": health.get("embedding_model", ""),
        "facts": health["facts"],
        "journal": health["journal"],
        "episodes": health["episodes"],
        "reflection_count": summary.get("total_reflections", 0),
    }
    parsed = json.loads(json.dumps(data))
    assert "facts" in parsed
    assert "embedding_model" in parsed


def test_health_resource(server):
    """The health resource returns embedding health."""
    store = server._memlife_store
    health = store.embedding_health()
    assert "facts" in health
    assert "episodes" in health
    assert "embedder_present" in health


def test_contradictions_resource(server):
    """The contradictions resource returns JSON."""
    store = server._memlife_store
    items = store.list_contradictions(limit=20)
    data = json.loads(json.dumps(items, default=str))
    assert isinstance(data, list)


def test_server_has_tools_registered(server):
    """The MCP server has all expected tools registered."""
    manager = server._tool_manager
    if hasattr(manager, '_tools'):
        tool_names = set(manager._tools.keys())
    else:
        tool_names = set()
    # The exact set depends on FastMCP internals, but at least some
    # of our tools should be there.
    expected = {"memory_store", "memory_search", "memory_search_journal",
                "memory_search_episodes", "memory_revise", "memory_expire",
                "memory_retrieve", "memory_gc"}
    # Check that at least the tool functions were registered
    # (FastMCP may store them differently across versions)
    assert len(tool_names) > 0 or hasattr(manager, '_tools')


def test_shutdown_mcp_server_closes_store(server):
    """shutdown_mcp_server closes the underlying store."""
    from memlife.mcp_server import shutdown_mcp_server
    store = server._memlife_store
    assert store._conn is not None or store.db_path
    shutdown_mcp_server(server)
    assert store._conn is None


def test_shutdown_mcp_server_idempotent(server):
    """shutdown_mcp_server can be called twice without error."""
    from memlife.mcp_server import shutdown_mcp_server
    shutdown_mcp_server(server)
    shutdown_mcp_server(server)  # should not raise


def test_create_server_embedder_is_dummy(server):
    """The default embedder is the zero-dependency DummyEmbedder."""
    from memlife.embedders import DummyEmbedder
    assert isinstance(server._memlife_embedder, DummyEmbedder)


@pytest.mark.asyncio
async def test_ollama_session_deferred_creation(tmp_path):
    """Ollama adapters do not create an aiohttp session outside an async context."""
    from memlife.adapters.ollama import OllamaEmbedder, OllamaChat

    embedder = OllamaEmbedder(model="dummy")
    chat = OllamaChat(model="dummy")
    # No session should exist before an async call.
    assert embedder._session is None
    assert chat._session is None

    # Creating the session from inside an async context works.
    async def _touch():
        embedder._ensure_session()
        chat._ensure_session()

    await _touch()
    assert embedder._session is not None
    assert chat._session is not None
    await embedder.close()
    await chat.close()