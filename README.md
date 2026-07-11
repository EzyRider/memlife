# memlife

Memory that degrades gracefully. Not another pile that grows forever.

[![PyPI](https://img.shields.io/pypi/v/memlife.svg)](https://pypi.org/project/memlife/)
[![Python](https://img.shields.io/pypi/pyversions/memlife.svg)](https://pypi.org/project/memlife/)
[![License](https://img.shields.io/pypi/l/memlife.svg)](https://github.com/EzyRider/memlife/blob/main/LICENSE)

**Current version: 0.5.4**

## What's new in 0.5.4

- **MCP reflection timeout controls.** `memlife-mcp-server` now exposes
  `--reflection-timeout` and `--reflection-total-timeout` so reflection stays
  within the MCP client's window on slower LLM endpoints.
- **README tool list refreshed.** The MCP server tool list now matches the
  actually implemented tools, including `memory_recall`, `memory_revise`, and
  `memory_expire`.
- **Memorias and polyphonic feature flags.** `--memorias-extraction` and
  `--polyphonic-recall` can now be enabled from the MCP server CLI.
- **Episode search by tool name.** `memory_search_episodes` now accepts a
  `tool_name` filter, making it easier to audit how a specific tool was used.
- **Vector backend default cleanup.** `--vector-backend` now defaults to `json`
  explicitly, with clearer precedence when legacy flags are also supplied.

## What's new in 0.5.3

- **Config validation regression fixed.** `MemoryConfig.validate()` now runs all
  invariant checks even when `vector_backend` is left at its default auto value.
  Previously the default of `None` caused validation to return early, skipping
  decay and threshold checks.
- **Binary backend no longer loses JSON-stored vectors.** `BinaryVectorBackend`
  deserialisation now falls back to JSON format for rows stored before the
  backend was switched, so changing from `json` to `binary` on an existing store
  does not make old vectors invisible to search.

## What's new in 0.5.2

- **Binary vector backend is now used for retrieval.** Previously the backend
  was selectable but recall still used inline cosine. Fact, episode, and journal
  vector recall now route through the configured backend, so `binary` uses
  Hamming distance search and `sqlite_vec` uses sqlite-vec KNN.
- **Vector backend precedence is centralised.** `MemoryConfig.resolved_vector_backend()`
  gives a clear order: explicit `vector_backend` > `use_binary_vectors` >
  `use_sqlite_vec` > `json` default. Both legacy flags at once now resolve to
  `binary` with a warning, instead of silently depending on if/else order.
- **MCP server shutdown is clean.** The shutdown path now closes the store,
  embedder, chat adapter, and reflector, and SIGTERM exits the process after
  cleanup instead of hanging.
- **Ollama adapters no longer create sessions before a loop exists.** Sessions
  are created lazily on first async call, avoiding `RuntimeError` when the
  embedder/chat is instantiated outside an event loop.

## What's new in 0.5.1

- `memlife-mcp-server` now accepts `--vector-backend {json,sqlite_vec,binary}`.

- **Pluggable vector backends.** Swap between JSON, binary, and `sqlite-vec`
  backends with a single config change. The new binary backend stores embeddings
  as bit-packed vectors for ~32x smaller storage, while sqlite-vec provides
  fast approximate nearest neighbours when the extension is available.
- **Public `Metrics` snapshot.** `store.metrics()` returns a structured
  snapshot of counts, embedding coverage, reflection aggregates, recall
  counters, DB metadata, and schema migration health. Exposed via the
  `memlife://stats` MCP resource.
- **Schema migration status.** `store.migration_status()` checks whether the
  database has all expected tables and columns, flags missing items, and
  reports SQLite version and page stats. Surfaces in `Metrics` so operators
  can detect a stale schema before it breaks.
- **pysqlite3 fallback for sqlite-vec.** On interpreters whose stdlib `sqlite3`
  is compiled without extension loading, memlife transparently falls back to
  `pysqlite3` so sqlite-vec can still load.
- **Reflection audit trail.** Reflection passes are recorded with proposed,
  kept, and dropped items, plus timing and model metadata. Paginated audit
  queries help debug what the reflection loop is actually doing.
- **Namespace default and validation hardening.** The default namespace is now
  `_default`, names are validated/normalized before any DB is opened, and the
  config fails fast on unknown vector backends or unsafe paths.

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
pip install memlife
```

With adapters (optional):

```bash
pip install memlife[ollama]       # Ollama embedder + chat
pip install memlife[openai]       # OpenAI embedder + chat
pip install memlife[sentence-transformers]  # Local embeddings
pip install memlife[mcp]          # MCP server
```

## Quickstart (30 seconds, zero dependencies)

```python
import asyncio
from memlife import MemoryStore, MemoryConfig, DummyEmbedder

async def main():
    store = MemoryStore(
        config=MemoryConfig(db_path="./memlife.db", embedding_model="dummy"),
        embedder=DummyEmbedder(),
    )

    # Store an episode (something happened)
    store.remember(task="User asked about deployment", outcome="success")

    # Store a fact (durable truth)
    await store.store_fact("User deploys via GitHub Actions", confidence=0.8)

    # Store an entity relationship (fact-like but structured)
    store.store_triple("User", "deploys_via", "GitHub Actions", confidence=0.8)

    # Retrieve relevant memories (unified scoring across all layers)
    context = await store.retrieve("deployment")
    print(context)

    store.close()

asyncio.run(main())
```

No Ollama, no OpenAI, no API key. The DummyEmbedder uses bag-of-words vectors — similar sentences get positive cosine similarity. The full lifecycle — store, retrieve, decay, GC, and entity graph — works without any LLM. Only structured extraction and reflection need a model.

## The Lifecycle

```
┌───────────┐     reflection      ┌───────────┐
│  EPISODE  │ ──────────────────▶│  JOURNAL  │
│  (event)  │   LLM synthesises   │ (belief)  │
└─────┬─────┘   observations &   └─────┬─────┘
│  extract triples   │
      │                                 │
      │ store_fact() / store_triple()  │ confidence decay
      ▼                                 │ (configurable)
┌───────────┐    recall bumps    ┌─────▼─────┐
│   FACT    │ ◀────────────────  │  RETIRE   │
│  (truth)  │   confidence +0.05 │ (floor)   │
└─────┬─────┘                    └─────┬─────┘
│    │
│    │ entity graph (triples)
│    ▼
┌───────────────┐
│ TRIPLE / GRAPH│
│(subject-pred- │
│ object + prov)│
└───────┬───────┘
      │
      │ revise / supersede             │ GC prunes
      ▼                                ▼
┌───────────┐                   ┌───────────┐
│ SUPERSEDED│  configurable     │  PRUNED   │
│ (replaced)│ ──────────────────▶│ (deleted) │
└───────────┘   retention       └───────────┘

UNIFIED SCORE = relevance × confidence × recency
Applied across ALL layers before every response.

NO-LLM MODE: store + retrieve + decay + GC + entity graph work
without any model. Only reflection and structured extraction need an LLM.
```

## No-LLM Mode

The store, retrieval, decay, GC, entity graph, and embedding versioning all work without any LLM. Only the reflection loop and structured extraction need a model.

```python
from memlife import MemoryStore, MemoryConfig

store = MemoryStore(config=MemoryConfig(db_path="./memlife.db"))
store.remember(task="something happened", outcome="success")

# retrieve() is async — use SyncMemoryStore or asyncio.run():
import asyncio
context = asyncio.run(store.retrieve("something"))
store.close()
```

## With an Embedder

```python
import asyncio
from memlife import MemoryStore, MemoryConfig
from memlife.adapters.ollama import OllamaEmbedder

async def main():
    store = MemoryStore(
        config=MemoryConfig(db_path="./memlife.db", embedding_model="mxbai-embed-large:latest"),
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
        config=MemoryConfig(db_path="./memlife.db", embedding_model="dummy"),
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
    config=MemoryConfig(db_path="./memlife.db", embedding_model="dummy"),
    embedder=DummyEmbedder(),
)
store.remember(task="hello", outcome="success")
fact_id = store.store_fact("Test fact", confidence=0.7)
context = store.retrieve("test")
```

## MCP Server

Expose memlife to any MCP-compatible agent (Claude Desktop, Cursor, etc.):

```bash
memlife-mcp-server --db ./memlife.db --embedder ollama --embedding-model mxbai-embed-large:latest
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
      "args": ["--db", "/path/to/memlife.db", "--embedder", "ollama", "--embedding-model", "mxbai-embed-large:latest"]
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
| `memory_store_triple` | Store an entity relationship |
| `memory_search_triples` | Search triples connected to an entity |
| `memory_entity_neighbors` | Traverse the entity graph |
| `memory_revise` | Revise an existing fact |
| `memory_expire` | Mark a fact as expired |
| `memory_retrieve` | Unified cross-layer retrieval |
| `memory_gc` | Run garbage collection |
| `memory_vacuum` | Reclaim disk space from the SQLite database |
| `memory_reflect` | Run reflection pass (synthesise episodes into journal) |

**Resources:**

| Resource | Description |
|----------|-------------|
| `memlife://stats` | Memory statistics |
| `memlife://health` | Embedding health report |
| `memlife://contradictions` | Detected contradictions |

## Features

- **Four-tier lifecycle:** Episode → Fact → Journal → Decay/Prune
- **Entity graph:** normalized entities, aliases, and temporal triples with provenance
- **Graph traversal:** BFS entity neighbors exposed via MCP, no external graph DB
- **Triple lifecycle:** closed triples and orphan entities/aliases are GC'd like everything else
- **Confidence decay:** facts decay with a configurable halflife; triples inherit the same decay
- **Unified scoring:** relevance × confidence × recency across all layers
- **Confidence ceiling (0.99):** facts are never immutable
- **GC with configurable retention:** superseded facts, episodes, runs, metrics, and closed triples
- **Embedding versioning:** detect stale vectors when the model changes, backfill automatically
- **Episode tool index:** search "have I used this tool before?"
- **Incremental contradiction detection:** O(new × n), not O(n²)
- **Reflection loop:** LLM synthesises observations, hypotheses, and revisions with a critic gate
- **Structured extraction:** optional MEMORIA extraction turns reflection output into attributable triples
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
| **Confidence erosion** | Yes (configurable halflife) | No | No | No |
| **GC + pruning** | Yes (configurable, includes triples) | No | No | No |
| **Reflection loop** | Yes (LLM + critic) | No | No | No |
| **Embedding versioning** | Yes | No | No | No |
| **Entity graph / triples** | Yes (SQLite-native) | No | No | Yes |
| **Graph lifecycle** | Yes (decay + GC) | No | No | No |
| **Zero-dependency mode** | Yes (DummyEmbedder) | No | No | No |
| **MCP server** | Yes | No | No | No |
| **Backend** | SQLite (single file) | Various | SQLite | Neo4j |
| **Multi-user** | Namespaces (isolated DBs) | Yes | Yes (by wing) | Yes |
| **Self-hosted/local** | Yes | Yes | Yes | Requires Neo4j |

memlife wins on lifecycle, decay, and zero-dependency quickstart. It doesn't pretend to beat everyone at everything — Mem0 has multi-user, Graphiti has deep graph analytics. If you want memory that degrades gracefully instead of accumulating forever, memlife is the one.

## Status

**v0.5.2.** The API may change before v1.0. Not recommended for production yet.

## License

MIT