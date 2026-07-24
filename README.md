# memlife

Memory that degrades gracefully. Not another pile that grows forever.

[![PyPI](https://img.shields.io/pypi/v/memlife.svg)](https://pypi.org/project/memlife/)
[![Python](https://img.shields.io/pypi/pyversions/memlife.svg)](https://pypi.org/project/memlife/)
[![License](https://img.shields.io/pypi/l/memlife.svg)](https://github.com/EzyRider/memlife/blob/main/LICENSE)

**Current version: 0.6.12**

memlife is a four-tier lifecycle memory system for AI agents. Instead of treating memory as a monotonically growing database, every entry has a lifecycle — facts decay, journal entries retire, superseded data is pruned, and nothing accumulates forever.

The four tiers:

- **Episodes** — raw events (what happened)
- **Facts** — durable truths (what I know)
- **Journal** — reflected beliefs (what I believe)
- **Decay/Prune** — confidence fades, stale entries retire, GC cleans up

## Why memlife

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

No Ollama, no OpenAI, no API key. The `DummyEmbedder` uses bag-of-words vectors — similar sentences get positive cosine similarity. The full lifecycle — store, retrieve, decay, GC, and entity graph — works without any LLM. Only structured extraction and reflection need a model.

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

This makes memlife useful for testing, offline agents, and lightweight deployments where you don't want to manage model access.

## Core Concepts

### Episodes

Episodes are raw events. They're cheap to store and useful for recall, but the system doesn't treat them as durable truth. Over time, reflection can promote observations from episodes into facts or journal entries.

```python
store.remember(
    task="User asked about deployment",
    outcome="success",
    confidence=0.9,
)
```

### Facts

Facts are durable truths with explicit confidence. They decay on disuse, get bumped on recall, and can be revised or superseded.

```python
await store.store_fact("User deploys via GitHub Actions", confidence=0.8)
```

### Entity Graph (Triples)

Structured relationships between entities. Triples behave like facts but participate in graph traversal.

```python
store.store_triple("User", "deploys_via", "GitHub Actions", confidence=0.8)
```

### Journal

The journal holds reflected beliefs — higher-level observations the system synthesises from episodes and facts during reflection.

```python
await store.reflect()  # or schedule it
```

### Decay, Retirement, and GC

Confidence decays over time. Entries that fall below a floor retire. Superseded entries age out. Garbage collection prunes them according to retention policy. Everything is configurable.

## Retrieval

`retrieve()` scores across all four tiers using a unified function:

```
score = relevance × confidence × recency
```

This means a highly relevant but low-confidence memory won't drown out a moderately relevant, high-confidence, recent one. The result is a blended context string ready to feed into a prompt.

```python
context = await store.retrieve("deployment")
```

## Configuration

```python
from memlife import MemoryConfig

config = MemoryConfig(
    db_path="./memlife.db",
    embedding_model="dummy",          # or "nomic-embed-text", etc.
    vector_backend="json",              # json | binary | sqlite_vec
    reflection_timeout=120.0,             # per-LLM-call timeout during reflection
    journal_decay_halflife_days=30.0,
    journal_decay_floor=0.15,
    namespace="default",
)
```

See `MemoryConfig` for the full set of options.

## MCP Server

memlife ships with an MCP server so any MCP-compatible client can use it:

```bash
memlife-mcp-server --db-path ./memlife.db --vector-backend binary
```

Tools exposed include `memory_store`, `memory_search`, `memory_search_journal`, `memory_search_episodes`, `memory_retrieve`, `memory_revise`, `memory_expire`, `memory_reflect`, `memory_gc`, `memory_vacuum`, `memory_store_triple`, `memory_search_triples`, and `memory_entity_neighbors`.

Resources include `memlife://stats`, `memlife://health`, and `memlife://contradictions`.
>
> **Note: PyPI vs GitHub `main`** — The `--log-tool-calls` flag ships in the
> current PyPI release (0.5.5 and later). If you are still on 0.5.4, omit
> `--log-tool-calls` from your MCP server command.
>
> **Windows / cloud-sync caveat:** Do not point `data_dir` at a OneDrive,
> Dropbox, Google Drive, iCloud, or other cloud-sync folder, or at a network
> share. SQLite WAL mode keeps `-wal` and `-shm` sidecar files next to the
> database that are constantly rewritten; sync clients and real-time antivirus
> scanners can lock or corrupt those files, causing "database is locked" or
> checksum errors. Use a local, non-synced directory. If you still see locking
> errors on Windows, set `sqlite_journal_mode="DELETE"` in `MemoryConfig` to
> disable WAL mode.

## What's new in 0.6.12

- `MemoryConfig.from_env()` validates environment configuration before returning.
- `retrieve()` logs and counts recall-path failures instead of silently swallowing them.
- Vector backend `delete()` rejects unknown table kinds via an explicit allowlist.
- MCP server supports `--chat-adapter {ollama,openai}` for non-Ollama reflection endpoints.
- Polyphonic recall counters now report source attribution per fused candidate.

## What's new in 0.6.11

- GitHub Actions now tests Python 3.11 and 3.12 only; the obsolete Python 3.10 job has been removed.

## What's new in 0.6.10

- **Minimum Python version is now 3.11.** memlife no longer supports Python 3.10, so the `typing_extensions` compatibility shim added in 0.6.8 has been removed.

## What's new in 0.6.9

- (Superseded by 0.6.10; the Python 3.11 minimum is now documented above.)

## What's new in 0.6.8

- Hardened `_LockedConn` thread-safety contract: cursors are now returned as context-managed `_LockedCursor` proxies that hold the store lock for their entire lifetime, and `row_factory`, `isolation_level`, and `text_factory` access is serialised through the same lock.
- Added PRAGMA allowlist validation in `MemoryConfig` and `MemoryStore._set_pragma()` so `journal_mode` and other PRAGMA values are checked before any SQL interpolation.
- Audited remaining cursor-iteration sites in `_gc.py`, `_triples.py`, and `_schema.py` to use explicit cursors or `.fetchall()` so cursors never outlive the lock.

## What's new in 0.6.7

- Fixed a silent gap in graph-integrated retrieval: lowercase queries now match stored entities and aliases. Previously `retrieve("james")` would not follow graph links for an entity stored as "James" because the query parser only recognised capitalised proper nouns. The extractor is still used for discovering new entities, but retrieval now also scans the query against known entity names and aliases case-insensitively.
- Added a regression test for lowercase entity queries in graph retrieval.

## What's new in 0.6.6

- Fixed a case-sensitivity bug in triple queries: `triples_about`, `triples_from`, `triples_to`, `current_truth`, `truth_as_of`, and `entity_neighbors` now resolve entities case-insensitively and via aliases, matching the insertion path. Querying "james" returns triples stored under "James" or its alias "Jimmy".
- Added a regression test covering mixed-case entity queries and alias resolution.

## What's new in 0.6.5

- Fixed an infinite-loop bug in embedding-cache GC: when the first batch of cache rows were all still referenced, the old `LIMIT`/`OFFSET`-less scan would fetch the same rows forever. GC now uses keyset pagination by `cache_key`.
- Embedding-cache lookup and storage are now batched, cutting round-trips from O(N) per text to 1–2 per batch.
- Auto-extracted entity mention triples and aliases are now committed in a single transaction instead of one commit per entity.
- GC output from the MCP `memory_gc` tool now includes mention-triple pruning, embedding-cache unreferenced rows, and LRU eviction counts.
- Tool-call dedup eviction in the MCP server is now guarded by a lock to avoid a latent race under concurrent tool threads.

## What's new in 0.6.4

- Synchronised `main` branch with the 0.6.2 and 0.6.3 release tags so the GitHub source tree and README match PyPI.

## What's new in 0.6.3

- README now reports the current version and includes a consolidated "What's New" section.
- `memlife.__version__` and `pyproject.toml` version metadata are kept in sync with releases.

## What's new in 0.6.2

- Graph relationship traversal now follows **incoming** edges as well as outgoing edges, so querying an entity that appears as the object of a relationship discovers related sources.
- Entity canonicalisation is case-insensitive when creating or ensuring entities. Manual `store_triple("James", ...)` reuses an auto-extracted canonical entity "james" instead of creating a duplicate "James" node.
- `MemoryStore.retrieve()` and `SyncMemoryStore.retrieve()` now accept `debug=True` and return the structured debug dict.
- `SyncMemoryStore.store_mention_triple()` added for parity with the async `MemoryStore` API.
- Regression test suite for graph-integrated retrieval added (`tests/test_graph_retrieval.py`).

## What's new in 0.6.1

- `_schema._migrate()` now re-reads journal columns before adding `annotations_json` / `links_json`, making migration idempotent on partially-migrated databases.
- `FactStore.check_conflicts()` keyword fallback now respects `fact_conflict_threshold` instead of hardcoding 0.5.
- Graph-integrated retrieval now scales `graph_signal` by the confidence of the strongest currently-valid linking triple, rather than giving every linked candidate a flat 1.0 boost.
- Graph retrieval no longer surfaces superseded facts, retired/superseded journal entries, or contradiction rows.
- `retrieve()` now skips vector-recalled episodes that were already added via graph expansion, avoiding duplicate episode candidates.
- Graph expansion failures are caught and logged instead of crashing the whole retrieval call.
- Polyphonic recall now runs RRF only on candidates that passed the score cutoff, so cutoff configuration remains meaningful.
- Debug output now includes the actual triples that produced a graph link in `graph_triples`.

## What's new in 0.6.0

- **Embedding cache** — content-addressable, LRU-capped embedding cache.
- **Automatic entity extraction** — deterministic, zero-LLM extraction of entities from facts, episodes, and journal entries.
- **Graph-integrated retrieval** — `retrieve()` boosts candidates linked to entities mentioned in the query.

## What's new in 0.5.5

- **Version consistency.** `memlife.__version__` now matches `pyproject.toml` (0.5.5).
- **README accuracy.** The MCP server tool list now lists only implemented
  tools and clarifies that `memlife://contradictions` is a resource, not a tool.
  The `MemoryConfig` example now uses real fields (`reflection_timeout`,
  `journal_decay_halflife_days`, `journal_decay_floor`).
- **Bounded tool-call logging cache.** The in-memory dedup dict used by
  `--log-tool-calls` is capped so long-running MCP servers cannot grow it
  without bound.
- **Idempotent MCP shutdown.** `shutdown_mcp_server()` now guards against
  double runs so SIGTERM and `atexit` cannot both attempt cleanup.
- **Windows namespace case-collision hardening.** `list_namespaces()` now
  normalizes directory names to lowercase and warns/ignores mixed-case
  duplicates, preventing two directories that resolve to the same database from
  appearing as separate namespaces on case-insensitive filesystems.
- **Cloud-sync path warning.** memlife now logs a warning when `data_dir`
  resolves under a known cloud-sync folder (OneDrive, Dropbox, Google Drive,
  iCloud, etc.) because sync clients can lock or corrupt SQLite WAL sidecar
  files.
- **Windows sqlite-vec fallback verified.** The store already falls back from
  `sqlite_vec` to `json` gracefully when extension loading is unavailable. On
  Windows, where `pysqlite3-binary` is not installed, this fallback uses the
  stdlib `sqlite3` module and continues to work.

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

## License

MIT
