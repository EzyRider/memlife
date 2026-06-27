# memlife

Memory that degrades gracefully. Not another pile that grows forever.

## What

memlife is a four-tier lifecycle memory system for AI agents:

- **Episodes** — raw events (what happened)
- **Facts** — durable truths (what I know)
- **Journal** — reflected beliefs (what I believe)
- **Decay/Prune** — confidence fades, stale entries retire, GC cleans up

Every memory has a lifecycle. Facts decay through confidence erosion. Journal entries retire when they fall below the floor. Superseded data is pruned after a retention period. Nothing accumulates forever.

## Why

Every other memory system accumulates. Facts never expire. Confidence never decays. Stale conventions become unquestioned truths. Recall quality degrades over time.

memlife solves this. Memory should be like human memory — it fades, it gets revised, it gets pruned. Not a database that grows until it breaks.

## Quickstart

```bash
pip install memlife
```

```python
import asyncio
from memlife import MemoryStore, MemoryConfig, DummyEmbedder

async def main():
    store = MemoryStore(
        config=MemoryConfig(db_path="./mem.db"),
        embedder=DummyEmbedder(),  # zero external dependencies
    )

    # Store an episode (something happened)
    store.remember(task="User asked about deployment", outcome="success")

    # Store a fact (durable truth)
    await store.store_fact("User deploys via GitHub Actions", confidence=0.8)

    # Retrieve relevant memories (unified scoring across all layers)
    context = await store.retrieve("deployment")

    print(context)
    store.close()

asyncio.run(main())
```

No Ollama, no OpenAI, no API key. The DummyEmbedder uses hash-based vectors. The full lifecycle — store, retrieve, decay, GC — works without any LLM.

## The Lifecycle

```
┌───────────┐     reflection      ┌───────────┐
│  EPISODE  │ ──────────────────▶ │  JOURNAL  │
│  (event)  │   LLM synthesises    │ (belief)  │
└─────┬─────┘   observations &    └─────┬─────┘
      │         hypotheses              │
      │                                  │
      │ store_fact()                    │ confidence decay
      ▼                                  │ (30d halflife)
┌───────────┐    recall bumps     ┌─────▼─────┐
│   FACT    │ ◀────────────────   │  RETIRE   │
│  (truth)  │   confidence +0.05  │ (floor)   │
└─────┬─────┘                     └─────┬─────┘
      │                                 │
      │ revise / supersede              │ GC prunes
      ▼                                 ▼
┌───────────┐                      ┌───────────┐
│ SUPERSEDED│   90 days retention  │  PRUNED   │
│ (replaced)│ ───────────────────▶ │ (deleted) │
└───────────┘                      └───────────┘

UNIFIED SCORE = relevance × confidence × recency
Applied across ALL layers before every response.
```

## No-LLM Mode

The store, retrieval, decay, GC, and embedding versioning all work without any LLM. Only the reflection loop needs a model. If you just want durable, decaying memory:

```python
store = MemoryStore(config=MemoryConfig(db_path="./mem.db"))
store.remember(task="something happened", outcome="success")
context = await store.retrieve("something")
```

## With Reflection

```python
from memlife import MemoryStore, MemoryConfig, Reflector, DummyEmbedder, DummyChat

store = MemoryStore(
    config=MemoryConfig(db_path="./mem.db"),
    embedder=DummyEmbedder(),
)
reflector = Reflector(
    memory=store,
    model_chat=DummyChat(),
    critic=False,
)
result = await reflector.reflect()
```

For real LLMs, implement the `Embedder` and `ChatCallable` protocols, or use an adapter (Phase 2).

## Features

- Four-tier lifecycle: Episode → Fact → Journal → Decay/Prune
- Unified scoring: relevance × confidence × recency across all layers
- Confidence ceiling (0.99) — facts are never immutable
- Confidence decay with 30-day halflife — journal entries fade
- GC with configurable retention (90 days for superseded facts, etc.)
- Embedding versioning — detect stale vectors when the model changes
- Episode tool index — search "have I used this tool before?"
- Incremental contradiction detection — O(new × n), not O(n²)
- JSONL import/export for backup and migration
- SQLite-backed, single file, zero external services
- Works with zero dependencies (DummyEmbedder + DummyChat)

## Status

**v0.1.0-beta.** The API may change before v1.0.

## License

MIT