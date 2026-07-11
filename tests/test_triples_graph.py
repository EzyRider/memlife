"""Tests for graph/triples layer (entity normalization, provenance, traversal)."""

import pytest

from memlife import MemoryConfig, MemoryStore


@pytest.fixture
def graph_store(tmp_path):
    db = tmp_path / "graph.db"
    store = MemoryStore(MemoryConfig(db_path=str(db)))
    yield store
    store.close()


def test_store_triple_creates_entities(graph_store):
    store = graph_store
    tid = store.store_triple("Alice", "knows", "Bob", confidence=0.9)
    assert tid.startswith("triple_")
    assert store.resolve_entity("Alice") == "Alice"
    assert store.resolve_entity("Bob") == "Bob"


def test_add_entity_alias(graph_store):
    store = graph_store
    store.add_entity_alias("Alice", "Alicia")
    assert store.resolve_entity("Alicia") == "Alice"
    # canonical still resolves to itself
    assert store.resolve_entity("Alice") == "Alice"


def test_store_triple_resolves_aliases(graph_store):
    store = graph_store
    store.add_entity_alias("Alice", "Alicia")
    tid = store.store_triple("Alicia", "knows", "Bob", confidence=0.8)
    triples = store.triples_about("Alice")
    assert any(t["id"] == tid and t["subject"] == "Alice" for t in triples)


def test_triples_about(graph_store):
    store = graph_store
    store.store_triple("Alice", "knows", "Bob")
    store.store_triple("Bob", "likes", "pizza")
    store.store_triple("Carol", "knows", "Alice")
    about = store.triples_about("Alice")
    assert len(about) == 2
    predicates = {t["predicate"] for t in about}
    assert "knows" in predicates


def test_triples_from_and_to(graph_store):
    store = graph_store
    store.store_triple("Alice", "knows", "Bob")
    store.store_triple("Alice", "likes", "pizza")
    store.store_triple("Carol", "knows", "Alice")
    out_ = store.triples_from("Alice")
    assert {t["object"] for t in out_} == {"Bob", "pizza"}
    into = store.triples_to("Alice")
    assert {t["subject"] for t in into} == {"Carol"}


def test_entity_neighbors_depth_one(graph_store):
    store = graph_store
    store.store_triple("Alice", "knows", "Bob")
    store.store_triple("Bob", "likes", "pizza")
    store.store_triple("Carol", "knows", "Alice")
    neighbors = store.entity_neighbors("Alice", depth=1)
    names = {n["entity"] for n in neighbors}
    assert names == {"Bob", "Carol"}


def test_entity_neighbors_depth_two(graph_store):
    store = graph_store
    store.store_triple("Alice", "knows", "Bob")
    store.store_triple("Bob", "likes", "pizza")
    neighbors = store.entity_neighbors("Alice", depth=2)
    names = {n["entity"] for n in neighbors}
    assert "pizza" in names


def test_triple_provenance(graph_store):
    store = graph_store
    tid = store.store_triple("Alice", "knows", "Bob", provenance=[{"kind": "episode", "id": "ep_123"}])
    triples = store.triples_about("Alice")
    t = next(x for x in triples if x["id"] == tid)
    assert t["provenance"] == [{"kind": "episode", "id": "ep_123"}]


def test_store_fact_triple_creates_provenance(graph_store):
    store = graph_store
    tid = store.store_fact_triple("fact_abc", "user", "lives_in", "Melbourne", confidence=0.9)
    triples = store.triples_about("user")
    t = next(x for x in triples if x["id"] == tid)
    assert any(p["kind"] == "fact" and p["id"] == "fact_abc" for p in t["provenance"])


def test_current_truth_still_works(graph_store):
    store = graph_store
    store.store_fact_triple("fact_1", "user", "prefers", "dark mode", confidence=0.95)
    obj, conf, _ = store.current_truth("user", "prefers")
    assert obj == "dark mode"
    assert conf == pytest.approx(0.95)


def test_mcp_server_registers_triple_tools(tmp_path):
    """The MCP server exposes the new graph/triples tools."""
    from memlife.mcp_server import create_server

    mcp = create_server(
        db_path=str(tmp_path / "mcp_graph.db"),
        embedder_type="dummy",
        embedding_model="dummy",
    )
    manager = mcp._tool_manager
    tool_names = set(manager._tools.keys()) if hasattr(manager, "_tools") else set()
    assert "memory_store_triple" in tool_names
    assert "memory_search_triples" in tool_names
    assert "memory_entity_neighbors" in tool_names
    mcp._memlife_store.close()
