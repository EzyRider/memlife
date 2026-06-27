"""Quickstart — 30-second demo with zero external dependencies.

No Ollama, no OpenAI, no API key. Just DummyEmbedder.
"""
import asyncio

from memlife import (
    MemoryConfig,
    MemoryStore,
    DummyEmbedder,
    retrieve,
)


async def main():
    # Create a store with the dummy embedder (no external dependencies)
    config = MemoryConfig(db_path="/tmp/memlife_quickstart.db")
    store = MemoryStore(config=config, embedder=DummyEmbedder())
    store.embedding_model_name = "dummy"

    # Store an episode (something that happened)
    ep_id = store.remember(
        task="User asked about deployment process",
        outcome="success",
        summary="Explained GitHub Actions workflow",
        tool_calls=[{"tool": "read_file", "params": {"path": "deploy.yml"}}],
    )
    print(f"Stored episode: {ep_id}")

    # Store a fact (durable truth)
    fact_id = await store.store_fact(
        "User deploys via GitHub Actions",
        confidence=0.8,
    )
    print(f"Stored fact: {fact_id}")

    # Store another fact
    fact_id2 = await store.store_fact(
        "User prefers dark mode",
        confidence=0.9,
    )
    print(f"Stored fact: {fact_id2}")

    # Retrieve relevant memories (unified scoring across all layers)
    context = await retrieve(store, "deployment", config)
    print(f"\n--- Retrieved context ---\n{context}")

    # Search episodes by tool name
    eps = store.search_episodes_by_tool("read_file", limit=5)
    print(f"\n--- Episodes using read_file: {len(eps)} ---")
    for ep in eps:
        print(f"  [{ep.id}] {ep.task}")

    # Check embedding health
    health = store.embedding_health()
    print(f"\n--- Embedding health ---")
    print(f"  model: {health['embedding_model']}")
    for layer in ("facts", "journal", "episodes"):
        h = health[layer]
        print(f"  {layer:12s}  {h['with_embeddings']}/{h['total']}")

    # Run garbage collection
    gc_result = store.run_gc(
        superseded_facts_days=0,  # aggressive for demo
        superseded_journal_days=0,
        completed_runs_days=0,
        metrics_days=0,
        reflected_queue_days=0,
    )
    print(f"\n--- GC result ---")
    print(f"  pruned: {gc_result['total_pruned']} rows")
    print(f"  DB: {gc_result['db_size_before_mb']}MB → {gc_result['db_size_after_mb']}MB")

    store.close()
    print("\nDone. memlife works with zero external dependencies.")


if __name__ == "__main__":
    asyncio.run(main())