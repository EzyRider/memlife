# memlife

Memory that degrades gracefully. Not another pile that grows forever.

[![PyPI](https://img.shields.io/pypi/v/memlife.svg)](https://pypi.org/project/memlife/)
[![Python](https://img.shields.io/pypi/pyversions/memlife.svg)](https://pypi.org/project/memlife/)
[![License](https://img.shields.io/pypi/l/memlife.svg)](https://github.com/EzyRider/memlife/blob/main/LICENSE)

## What

memlife is a four-tier lifecycle memory system for AI agents. Instead of treating memory as a monotonically growing database, every entry has a lifecycle — facts decay, journal entries retire, superseded data is pruned, and nothing accumulates forever.

The four tiers:

- **Episodes** — raw events (what happened)
- **Facts** — durable truths (what I know)
- **Journal** — reflected beliefs (what I believe)
- **Decay/Prune** — confidence fades, stale entries retire, GC cleans up

## Why

Every other memory system accumulates. Facts never expire. Confidence never decays. Stale conventions become unquestioned truths. Recall quality degrades over time.

memlife solves this. Memory should be like human memory — it fades, it gets revised, it gets pruned. Not a database that grows until it breaks.

## Install

```bash
pip install memlife --pre
```

With adapters (optional):

```bash
pip install memlife[ollama] --pre       # Ollama embedder + chat
pip install memlife[openai] --pre       # OpenAI embedder + chat
pip install memlife[sentence-transformers] --pre  # Local embeddings
pip install memlife[mcp] --pre          # MCP server
```

## Quickstart (30 seconds, zero dependencies)

```python
import asyncio
from memlife import MemoryStore, MemoryConfig, DummyEmbedder

async def main():
    store = MemoryStore(
        config=MemoryConfig(db_path="./mem.db", embedding_model="dummy"),
        embedder=DummyEmbedder(),
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

No Ollama, no OpenAI, no API key. The DummyEmbedder uses bag-of-words vectors — similar sentences get positive cosine similarity. The full lifecycle — store, retrieve, decay, GC — works without any LLM.

## The Lifecycle

```
┌───────────┐     reflection      ┌───────────┐
│  EPISODE  │ ──────────────────▶│  JOURNAL  │
│  (event)  │   LLM synthesises   │ (belief)  │
└─────┬─────┘   observations &   └─────┬─────┘
      │         hypotheses             │
      │                                 │
      │ store_fact()                   │ confidence decay
      ▼                                 │ (30d halflife)
┌───────────┐    recall bumps    ┌─────▼─────┐
│   FACT    │ ◀────────────────  │  RETIRE   │
│  (truth)  │   confidence +0.05 │ (floor)   │
└─────┬─────┘                    └─────┬─────┘
      │                                │
      │ revise / supersede             │ GC prunes
      ▼                                ▼
┌───────────┐                   ┌───────────┐
│ SUPERSEDED│  90 days retention │  PRUNED   │
│ (replaced)│ ──────────────────▶│ (deleted) │
└───────────┘                   └───────────┘

UNIFIED SCORE = relevance × confidence × recency
Applied across ALL layers before every response.

NO-LLM MODE: store + retrieve + decay + GC work
without any model. Only reflection needs an LLM.
```

## No-LLM Mode

The store, retrieval, decay, GC, and embedding versioning all work without any LLM. Only the reflection loop needs a model.

```python
from memlife import MemoryStore, MemoryConfig

store = MemoryStore(config=MemoryConfig(db_path="./mem.db"))
store.remember(task="something happened", outcome="success")

# retrieve() is async — use SyncMemoryStore or asyncio.run():
import asyncio
context = asyncio.run(store.retrieve("something"))
```

## With an Embedder

```python
import asyncio
from memlife import MemoryStore, MemoryConfig
from memlife.adapters.ollama import OllamaEmbedder

async def main():
    store = MemoryStore(
        config=MemoryConfig(db_path="./mem.db", embedding_model="mxbai-embed-large:latest"),
        embedder=OllamaEmbedder(model="mxbai-embed-large:latest"),
    )
    await store.store_fact("User prefers dark mode", confidence=0.9)
    context = await store.retrieve("dark mode")
    store.close()

asyncio.run(main())
```

Also available: `OpenAIEmbedder` (`pip install memlife[openai]`) and `STEmbedder` for local Sentence Transformers (`pip install memlife[sentence-transformers]`).

## With Reflection

```python
import asyncio
from memlife import MemoryStore, MemoryConfig, Reflector, DummyEmbedder, DummyChat

async def main():
    store = MemoryStore(
        config=MemoryConfig(db_path="./mem.db", embedding_model="dummy"),
        embedder=DummyEmbedder(),
    )
    reflector = Reflector(
        memory=store,
        model_chat=DummyChat(),
        critic=False,
    )
    result = await reflector.reflect()
    store.close()

asyncio.run(main())
```

For real LLMs, use an adapter:

```python
from memlife.adapters.ollama import OllamaChat

# Provide your own model name — memlife doesn't ship deployment-specific defaults.
chat = OllamaChat(model="your-model-name")
reflector = Reflector(memory=store, model_chat=chat, agent_name="my-agent")
```

## Sync API

For non-async codebases:

```python
from memlife import SyncMemoryStore, MemoryConfig, DummyEmbedder

store = SyncMemoryStore(
    config=MemoryConfig(db_path="./mem.db", embedding_model="dummy"),
    embedder=DummyEmbedder(),
)
store.remember(task="hello", outcome="success")
fact_id = store.store_fact("Test fact", confidence=0.7)
context = store.retrieve("test")
```

## MCP Server

Expose memlife to any MCP-compatible agent (Claude Desktop, Cursor, etc.):

```bash
memlife-mcp-server --db ./mem.db --embedder ollama --embedding-model mxbai-embed-large:latest
```

Claude Desktop config:

**macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`

**Linux:** `~/.config/Claude/claude_desktop_config.json`

**Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "memlife": {
      "command": "memlife-mcp-server",
      "args": ["--db", "/path/to/mem.db", "--embedder", "ollama", "--embedding-model", "mxbai-embed-large:latest"]
    }
  }
}
```

**Tools exposed:**

| Tool | Description |
|------|-------------|
| `memory_store` | Store a durable fact |
| `memory_search` | Search facts by query |
| `memory_search_journal` | Search journal entries |
| `memory_search_episodes` | Search episodes by keyword or tool name |
| `memory_revise` | Revise an existing fact |
| `memory_expire` | Mark a fact as expired |
| `memory_retrieve` | Unified cross-layer retrieval |
| `memory_gc` | Run garbage collection |

**Resources:**

| Resource | Description |
|----------|-------------|
| `memlife://stats` | Memory statistics |
| `memlife://health` | Embedding health report |
| `memlife://contradictions` | Detected contradictions |

## Features

- **Four-tier lifecycle:** Episode → Fact → Journal → Decay/Prune
- **Unified scoring:** relevance × confidence × recency across all layers
- **Confidence ceiling (0.99):** facts are never immutable
- **Confidence decay:** 30-day halflife, floored at 0.15 — journal entries fade
- **GC with configurable retention:** 90 days for superseded facts, 180 for episodes, 60 for runs, 30 for metrics
- **Embedding versioning:** detect stale vectors when the model changes, backfill automatically
- **Episode tool index:** search "have I used this tool before?"
- **Incremental contradiction detection:** O(new × n), not O(n²)
- **Reflection loop:** LLM synthesises observations, hypotheses, and revisions with a critic gate
- **JSONL import/export:** backup and migration
- **MCP server:** plug into Claude, Cursor, or any MCP client
- **Adapters:** Ollama, OpenAI, Sentence Transformers
- **Sync wrapper:** for non-async codebases
- **SQLite-backed:** single file, zero external services
- **Zero dependencies:** works out of the box with DummyEmbedder + DummyChat

## Comparison

| | memlife | Mem0 | MemPalace | Graphiti |
|---|---|---|---|---|
| **Lifecycle/decay** | Yes — core feature | No | No | No |
| **Confidence erosion** | Yes (30d halflife) | No | No | No |
| **GC + pruning** | Yes (configurable) | No | No | No |
| **Reflection loop** | Yes (LLM + critic) | No | No | No |
| **Embedding versioning** | Yes | No | No | No |
| **Zero-dependency mode** | Yes (DummyEmbedder) | No | No | No |
| **MCP server** | Yes | No | No | No |
| **Backend** | SQLite (single file) | Various | SQLite | Neo4j |
| **Multi-user** | No (single-agent) | Yes | Yes (by wing) | Yes |
| **Graph reasoning** | No | No | No | Yes |
| **Self-hosted/local** | Yes | Yes | Yes | Requires Neo4j |

memlife wins on lifecycle, decay, and zero-dependency quickstart. It doesn't pretend to beat everyone at everything — Mem0 has multi-user, Graphiti has graph reasoning. If you want memory that degrades gracefully instead of accumulating forever, memlife is the one.

## Status

**v0.3.7-beta.** The API may change before v1.0. Not recommended for production yet.

## License

MIT