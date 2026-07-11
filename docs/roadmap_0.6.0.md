# memlife 0.6.0 Roadmap

> Clean audit of `main` as of 2026-07-11. Stale backlog items that are already
> shipped have been removed. Only genuinely pending work, deferred scale items,
> and packaging/docs gaps remain.
>
> Last updated: 2026-07-11

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

## 2. Genuinely pending work

### 2.1 Embedding cache / model-versioned vectors (lead 0.6.0 item)

There is no content-addressable embedding cache today. Every `store_fact`,
`store_triple` (when triples carry embeddings), journal entry, and episode
re-embeds text even if the same text was embedded moments ago with the same
model.

**Goal:** Cache vectors keyed on `(model_name, sha256(text))` so that:
- Model swaps are cheap (only new/changed text hits the embedder).
- Repeated text (e.g. reflection re-processing the same episode) is instant.
- The cache is namespace-aware or stored inside each SQLite file so sharing
  an embedder across namespaces remains safe.

**Interface sketch:**
- Add `embedding_cache_table` to schema (or reuse a simple SQLite table):
  `cache_key TEXT PRIMARY KEY, model_name TEXT, text_hash TEXT, vector BLOB,
  created_at REAL`.
- Wrap `Embedder.embed()` in `EmbeddingMixin` with a cache read/write path.
- Provide `MemoryConfig.embedding_cache_enabled: bool = True` and
  `embedding_cache_max_mb: int = 512` for eviction.
- `backfill_embeddings()` should prime the cache as it goes.

**Why first:** Entity extraction and reflection both repeatedly embed similar
or identical text. Building the cache before those features makes them cheaper
and more deterministic.

### 2.2 Automatic entity extraction from free-form text

The graph storage exists, but entities are only created when a caller
explicitly invokes `store_triple`/`store_fact_triple` or when MEMORIA
reflection emits a labelled KG triple. There is no automatic extraction from
arbitrary facts, episodes, or journal entries.

**Goal:** Given a fact, episode, or journal entry, extract entity mentions
and (optionally) relations, then link them into the existing triple/entity
layer.

**Approaches to evaluate:**
1. **Zero-dependency regex/NER heuristics** — preserve the no-LLM contract.
2. **LLM-based extraction** — opt-in, gated by `config`, used during reflection.
3. **Hybrid** — regex for entities, LLM only for relation classification.

**Scope for 0.6.0 MVP:**
- Extract canonical entity mentions from fact content and episode task/summary.
- Create `entity_aliases` entries for variants seen in text.
- Optionally create `mentions` triples (e.g. `fact_abc mentions Entity`) so
  retrieval can follow the link.
- Keep it deterministic and reversible (GC should clean up auto-created
  entities when the source fact is pruned).

### 2.3 Graph-integrated retrieval

Today `retrieve()` pools episodes, facts, and journal but does not use the
entity graph. A query like "what projects is James working on?" should be
able to follow `James --works_on--> Project` triples and boost related facts.

**Goal:** Blend graph traversal into the retrieval pipeline without breaking
the unified `score = relevance × confidence × recency` model.

**Scope for 0.6.0:**
- Optional graph expansion step: starting from entities mentioned in the query,
  fetch related triples and the facts/episodes that mention those entities.
- Add a small graph-recency/confidence signal to candidate scoring.
- Expose a debug flag so callers can see which triples expanded the result set.

**Gate:** Only if automatic entity extraction (2.2) is in place; otherwise
there is too little graph data to make retrieval changes meaningful.

### 2.4 Docs / packaging gaps

| Item | File | Status |
|------|------|--------|
| Vector backend comparison | `docs/vector-backends.md` | Not created |
| Namespace design & migration | `docs/namespaces.md` | Not created |
| Reflection audit & corrections | `docs/reflection-audit.md` | Not created |
| README MCP tool list accuracy | `README.md` | `memory_recall` is listed but does not exist; `memory_contradictions` is a resource, not a tool |
| README config snippet accuracy | `README.md` | Uses `reflection_interval`, `decay_half_life_days`, `confidence_floor`, which do not exist in `MemoryConfig` |
| Package version consistency | `src/memlife/__init__.py` | `__version__` is `"0.5.3"` while `pyproject.toml` and README say `0.5.4` |

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
   `fact_decay_halflife_days`, `journal_decay_floor`, `recall_min_score`).
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
`src/memlife/_embeddings.py`, `src/memlife/_triples.py`,
`src/memlife/_episodes.py`, `src/memlife/reflection.py`,
`src/memlife/mcp_server.py`, `src/memlife/_schema.py`.*
