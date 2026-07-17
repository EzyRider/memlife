# memlife 0.6.0 Roadmap

> Clean audit of `main` and `CHANGELOG.md` as of 2026-07-11.
> Stale backlog items that are already shipped have been removed.
> Only genuinely pending work, deferred scale items, and packaging/docs gaps remain.
>
> Last updated: 2026-07-18

## 1. State of `main` (what is already true)

The following items were listed as pending in the previous draft but are
already shipped and tested:

- **MF-012** — `import_jsonl` column-name SQL injection fix (whitelist per
  table). Shipped in 0.3.2b0 (`d482b2d`).
- **MF-014** — `DummyEmbedder` bag-of-words vectors. Shipped in 0.3.2b0
  (`d482b2d`).
- **MF-005** — Stale contradiction embedding backfill on model swap. Shipped
  in 0.3.3b0 (`a3088f7`); contradictions with existing embeddings and a stale
  model are re-embedded, while contradictions with no embedding remain skipped.
- **MF-016 cleanup items** — `SyncMemoryStore._run()` concurrency fix,
  `OpenAIChat.chat()` empty-choices guard, `sentence_transformers`
  `get_running_loop()` migration, README no-LLM `await` fix, corrupt checkpoint
  handling, `MemoryStore` context manager, `limit` validation, unified journal
  scoring, batched `consolidate_journal()`, broader `_normalize()` punctuation
  stripping. All shipped in 0.3.4b0 (`f5c3931`).
- **`source_episodes_json` overload** — Documented in schema comment as
  intentional (holds episode IDs for observations/hypotheses/revisions and fact
  IDs for contradictions). Shipped in 0.3.4b0 (`f5c3931`).
- **`Reflector` config "duplication"** — Intentional separation of concerns;
  `Reflector` takes a small set of tuning parameters that may differ from
  `MemoryConfig`. Not a bug.
- **Namespace isolation** — Separate DB files per namespace, validation,
  case-normalisation, `list_namespaces()`, `switch_namespace()`. Shipped in
  0.5.0 (`f319425`, `345b555`, `4a98792`).
- **Pluggable vector backends** — `VectorBackend` ABC with JSON, binary, and
  sqlite-vec implementations; `MemoryConfig.vector_backend`; legacy flag
  migration. Shipped in 0.4.5–0.5.2 (`ba82e5d`, `b55af46`, `0635556`,
  `0861e7e`, `6eca99c`, `1f02f91`, `00fc3ba`, `b232a15`).
- **Entity graph substrate** — `entities`, `entity_aliases`, `temporal_triples`,
  `triple_provenance` tables; `store_triple`, `store_fact_triple`,
  `resolve_entity`, `add_entity_alias`, `triples_about`, `triples_from`,
  `triples_to`, `triples_for_fact`, `current_truth`, `truth_as_of`,
  `entity_neighbors`; MCP tools `memory_store_triple`,
  `memory_search_triples`, `memory_entity_neighbors`. Shipped in 0.4.3
  (`9a2b7d2`).
- **MEMORIA structured extraction** — Regex-based extraction of facts,
  preferences, instructions, timelines, and KG triples; wired into
  `Reflector.reflect()` when `config.memorias_extraction` is enabled. Shipped
  in 0.4.3 (`7a44e40`).
- **Polyphonic recall** — RRF fusion across retrieval voices; opt-in via
  `config.use_polyphonic_recall` and the MCP `--polyphonic-recall` flag.
  Shipped in 0.4.0b0 and exposed on the MCP server in 0.5.4 (`daf7f87`).
- **Incremental contradiction detection** — `Reflector._detect_contradictions`
  compares only new/updated facts against the full active set, giving
  O(new × n) steady-state cost, not O(n²). Shipped in 0.4.x/0.5.x.
- **Tool-call episode indexing** — `episode_tools` table,
  `search_episodes_by_tool(outcome=...)`, and `memory_search_episodes` with
  `tool_name`/`outcome` filters. Shipped in 0.5.x.
- **Confidence ceiling** — `MAX_FACT_CONFIDENCE = 0.99`; no fact is immutable.
  Shipped in V2.
- **Contradiction surfacing** — `memlife://contradictions` resource and
  `Metrics.unresolved_contradictions`. Shipped in 0.5.0.
- **GC / retention** — `run_gc()` with per-tier retention days and separate
  `run_vacuum()`. Shipped across 0.3.x–0.5.x.
- **MCP server cleanup** — `shutdown_mcp_server()` closes store, embedder,
  chat adapter, and reflector; registered with `atexit` and SIGTERM handler.
  Shipped in 0.5.2 (`1f02f91` / `f5c3931` follow-ups).
- **Reflection audit / correction propagation** — Reflection passes persisted
  with proposed/kept/dropped items, model metadata, and timing; user
  corrections stored as superseding journal entries. Shipped in 0.5.0
  (`345b555`).
- **Metrics / migration status** — `MemoryStore.metrics()`, public `Metrics`
  snapshot, `migration_status()`, `memlife://stats`. Shipped in 0.5.0
  (`9041f1d`, `a76e445`).

## 2. Genuinely pending work

### 2.1 Embedding cache / model-versioned vectors (lead 0.6.0 item)

**Status: shipped in `0fed63d`.**

A content-addressable embedding cache is now implemented. Vectors are keyed
on `(model_name, sha256(text))` and stored as canonical JSON floats so switching
`vector_backend` never leaves cache rows unreadable.

**What works:**
- `embedding_cache` table with `cache_key`, `model_name`, `text_hash`,
  `vector_json`, `created_at`, `last_used_at`.
- Cache read/write wrapped into `EmbedMixin.embed_texts()` — only cache misses
  hit the embedder.
- `MemoryConfig.embedding_cache_enabled: bool = True` and
  `embedding_cache_max_mb: int = 512` (env: `MEMLIFE_EMBEDDING_CACHE_ENABLED`,
  `MEMLIFE_EMBEDDING_CACHE_MAX_MB`).
- `backfill_embeddings()` primes the cache as it goes.
- `run_gc()` sweeps unreferenced cache rows and enforces the LRU size cap.
- `embedding_health()` and `Metrics` expose cache entry/size stats.
- Migration support for existing databases.

**Remaining follow-ups:**
- Stress-test the cache under model swaps and namespace switches.
- Decide whether `store_triple` should cache any embedded triple text once
  triples carry embeddings.

### 2.2 Automatic entity extraction from free-form text

**Status: shipped.**

The graph storage exists, but entities were only created when a caller
explicitly invoked `store_triple`/`store_fact_triple` or when MEMORIA
reflection emitted a labelled KG triple. Automatic extraction from arbitrary
facts, episodes, and journal entries is now implemented.

**What works:**
- `MemoryConfig.auto_entity_extraction: bool = False` (opt-in).
- `MemoryConfig.auto_entity_mentions: bool = True` creates `mentions` triples
  linking each source row to the entities it contains.
- `MemoryConfig.auto_entity_confidence: float = 0.6` for generated mention
  triples.
- `entity_extraction_allowlist` / `entity_extraction_blocklist` for tuning.
- Heuristic, zero-LLM extractor in `memlife.entity_extractor`:
  - Capitalised phrases (proper nouns).
  - Known terms from an allowlist.
  - Short uppercase acronyms.
  - Deduplication and blocklist filtering.
- Hooked into `store_fact()`, `remember()`, and `add_journal_entry()`.
- Mention triples are GC'd automatically when their source row is pruned.
- Orphan entities/aliases are removed by the existing entity/alias GC.

**Env vars:**
- `MEMLIFE_AUTO_ENTITY_EXTRACTION`
- `MEMLIFE_AUTO_ENTITY_MENTIONS`
- `MEMLIFE_AUTO_ENTITY_CONFIDENCE`

**Remaining follow-ups:**
- Graph-integrated retrieval (2.3) now consumes these `mentions` triples.

### 2.3 Graph-integrated retrieval

**Status: shipped.**

`retrieve()` now boosts candidates that are linked to entities mentioned in
the query. A query like "what projects is James working on?" can follow
`James --works_on--> Project` triples and surface related facts and episodes
even when vector/text scores are low.

**What works:**
- `MemoryConfig.use_graph_retrieval: bool = False` (opt-in).
- `MemoryConfig.graph_retrieval_weight: float = 0.25` scales the graph
  signal against the standard relevance score.
- Entity extraction from the query uses the same deterministic extractor as
  storage, honouring `entity_extraction_allowlist` / `entity_extraction_blocklist`.
- Canonical entity resolution is case-insensitive via `entity_aliases`.
- Graph expansion loads sources linked by `mentions` triples and follows one-hop
  relationship triples to neighbouring entities.
- Graph signal scales by triple confidence and source recency.
- Debug output exposes `graph_signal` and the expanding triples per candidate.

**Remaining follow-ups:**
- Evaluate whether multi-hop expansion is worth the extra latency.
- Add dedicated tests for graph-only retrieval (`vector/text/source/veracity
  weights` all zero).

### 2.4 Docs / packaging gaps

| Item | File | Status | Notes |
|------|------|--------|-------|
| Vector backend comparison | `docs/vector-backends.md` | Not created | Compare JSON / binary / sqlite-vec |
| Namespace design & migration | `docs/namespaces.md` | Not created | Design, migration, backup/restore |
| Reflection audit & corrections | `docs/reflection-audit.md` | Not created | Transparency and correction usage |
| README MCP tool list accuracy | `README.md` | Stale | Lists `memory_recall` (removed in `f51306a`) and calls `memory_contradictions` a tool; it is a resource (`memlife://contradictions`) |
| README config snippet accuracy | `README.md` | Stale | Uses `reflection_interval`, `decay_half_life_days`, `confidence_floor`, which do not exist in `MemoryConfig` |
| Package version consistency | `src/memlife/__init__.py` | Stale | `__version__` is `"0.5.3"` while `pyproject.toml` and README say `0.5.4` (release commit `3708f92` bumped only `pyproject.toml`) |

## 3. Operational / scale items (deferred)

These are valid but not blocking for 0.6.0:

| Item | Notes |
|------|-------|
| Brute-force vector search replacement | sqlite-vec already provides ANN when available; revisit when JSON/binary latency becomes painful |
| Journal consolidation false-merge mitigation | Latent; needs operational data to design the right guard |
| SQLite backup rotation via cron | Hermes priority #4; operational, not a code change |
| Long-context summarisation of retrieved bundles | 0.8.0 candidate |
| Multi-agent identity / author attribution | Research-level; out of 0.6.0 |
| Sync subsystem as separate package | Out of 0.6.0 |
| HTTP server wrapper for MCP | Out of 0.6.0 |

## 4. Release planning

### 0.5.5 (patch — docs & packaging only)

The security / correctness / concurrency fixes originally slated for 0.5.5
are already in `main`. 0.5.5 should therefore be a small docs/packaging patch:

1. Fix `src/memlife/__init__.py` `__version__` to match `pyproject.toml`
   (`0.5.4`).
2. Fix README MCP tool list: remove non-existent `memory_recall`; clarify that
   `memory_contradictions` is a resource.
3. Fix README `MemoryConfig` example to use real fields (e.g.
   `reflection_timeout`, `journal_decay_halflife_days`, `journal_decay_floor`).
4. Add the missing 0.5.4 CHANGELOG attribution for README/tool-list fixes if
   not already present.

### 0.6.0 (minor — entity graph + embedding cache)

1. **Embedding cache / model-versioned vectors** — build first; everything
   below benefits from it.
2. **Automatic entity extraction** — regex/LLM hybrid, opt-in, zero-dependency
   path preserved.
3. **Graph-integrated retrieval** — gated on (2) producing useful graph data.
4. **Docs split** — create `docs/vector-backends.md`, `docs/namespaces.md`,
   `docs/reflection-audit.md`.

### Beyond 0.6.0

- Multi-agent identity / collaborative attestation.
- Sync subsystem as a separate package.
- HTTP server wrapper for MCP.
- Cross-encoder reranking and long-context summarisation.

## 5. Decision register

| ID | Decision | Status | Notes |
|----|----------|--------|-------|
| R1 | Namespace strategy | Shipped in 0.5.x | Separate DB files per namespace |
| R2 | Vector backend config | Shipped in 0.5.x | `vector_backend` enum/string |
| R2a | Vector backend migration | Shipped in 0.5.x | Explicit opt-in |
| R3 | Multi-agent scope | Out of 0.6.0 | Research track |
| R4 | Corrections as journal entries | Shipped in 0.5.x | Reuse supersession machinery |
| R5 | Sync subsystem | Separate package, later | Unchanged |
| R6 | Reflection pass retention | Shipped in 0.5.x | Count + days |
| R7 | Embedder sharing across namespaces | Shipped in 0.5.x | Opt-in only |
| R8 | Reflection pass API | Shipped in 0.5.x | Internal record, public audit |
| R9 | Next major feature | Entity graphing | Decided 2026-07-11; substrate shipped in 0.4.3 |
| R10 | Incremental conflict detection | Shipped | O(new × n) since 0.4.x/0.5.x |
| R11 | Embedding cache sequencing | First 0.6.0 item | Confirmed 2026-07-11 |
| R12 | Auto entity extraction | Promoted to 0.6.0 | Builds on existing graph layer |
| R13 | Graph-integrated retrieval | 0.6.0 candidate | Gated on R12 |
| R14 | 0.5.5 scope | Docs/packaging only | Security fixes already shipped |

---

*Source files audited: `BACKLOG.md`, `docs/ROADMAP.md`, `CHANGELOG.md`,
`README.md`, `src/memlife/__init__.py`, `src/memlife/config.py`,
`src/memlife/mcp_server.py`, `src/memlife/_schema.py`.*
