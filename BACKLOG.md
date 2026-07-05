# memlife Backlog

Items to address in the next reroll of memlife. Sourced from Ingrid backend audit
(June 30, 2026) and Hermes operational experience.

## Bug Fixes

### MF-001: Contradiction dedupe in _store()
**Priority:** High  
**Source:** Ingrid backend audit, June 2026

`_store()` in `reflection.py` writes contradictions without checking whether
the same fact-pair contradiction already exists. Result: 3,693 duplicate
contradictions of 47 unique pairs in Ingrid's DB, bloating retrieval and metrics.

**Fix:** Add a composite key guard — before writing a contradiction, check if an
active contradiction for the same fact-pair already exists. If so, skip the
write (or bump its last_detected timestamp instead of creating a new row).

This is a memlife-level bug, not a consumer bug. Every agent using memlife's
contradiction detection will hit this duplication unless it's fixed upstream.

**Status:** Confirmed patched in Ingrid backend (`ingrid/journal/reflection.py::_store()`)
and upstreamed into the standalone memlife package (`src/memlife/reflection.py`)
with `has_active_contradiction()` / `touch_active_contradiction()`. Both test
suites green.

### MF-002: WAL + busy_timeout in MemoryStore.__init__
**Priority:** High  
**Source:** Nano DB corruptions (2x), known since June 2026

MemoryStore.__init__ does not enable WAL mode and busy_timeout by default.
Concurrent writes from multiple processes caused two b-tree corruptions in
NanoBot. Consumers currently have to enable these manually, and most don't.

**Fix:** Enable `PRAGMA journal_mode=WAL` and `PRAGMA busy_timeout=5000` in
`__init__` by default, before any table creation. Make it overridable via
config but on by default.

**Status:** Confirmed patched in Ingrid backend (`ingrid/memory/store.py` and
`ingrid/config.py`); full test suite green (209 passed). Pending upstreaming
into the standalone memlife package.

## API Design

### MF-003: Reflector lifecycle contract
**Priority:** Medium  
**Source:** Ingrid backend audit, June 2026

The Reflector is designed to be created once and reused across reflection
passes, but the API doesn't enforce or document this. Ingrid recreates it on
every pass in `agent.reflect()`, resetting `_last_contradiction_scan` to 0.0
and disabling incremental scanning. Every consumer will make the same mistake.

**Fix options (pick one):**
1. Make `_last_contradiction_scan` a pass-through parameter — caller persists
   it between calls, Reflector stays stateless. Cleanest separation.
2. Make Reflector a singleton via factory method — `Reflector.get_or_create()`
   returns existing instance, prevents recreation.
3. Document the lifecycle contract clearly and raise a warning if Reflector
   is recreated with a fresh timestamp.

Option 1 is the most memlife-idiomatic — the caller owns persistence, the
Reflector owns logic. Aligns with the existing pattern where MemoryStore
persists everything else.

**Status:** Patched in Ingrid backend (`ingrid/journal/reflection.py`):
`Reflector` is now stateless. `reflect()` accepts and returns
`last_contradiction_scan` and `reflection_cycle`; `IngridAgent` persists these
watermarks across passes. Existing tests updated to simulate caller
persistence, plus a new statelessness test. Full suite green (216 passed).
Pending upstreaming into the standalone memlife package.

## Features (core to decay thesis, not feature creep)

### MF-004: Contradiction retirement policy
**Priority:** Medium  
**Source:** Ingrid backend audit, June 2026

Old active contradictions accumulate forever. There's no mechanism to retire
contradictions that haven't been re-detected in N reflection cycles. This is
the same accumulation problem memlife's decay model solves for facts —
contradictions need the same lifecycle.

**Design:** Contradictions should have a confidence or last_detected timestamp.
If a contradiction hasn't been re-detected within N reflection cycles (or N
days), retire it — set status to 'retired' or delete it. N should be
configurable, defaulting to something reasonable (30 cycles? 60 days?).
This is decay applied to contradictions, which is the memlife thesis.

**Status:** Confirmed patched in Ingrid backend (`ingrid/journal/reflection.py`
and `ingrid/memory/store.py`): contradictions are stored as journal entries
with a `last_detected` cycle; re-detected pairs are touched instead of
duplicated; unresolved contradictions are reinforced each pass; and stale
contradictions older than `contradiction_retirement_cycles` (default 14,
configurable via `Config`) are retired. Wired through `IngridAgent` and
covered by targeted tests in `tests/test_journal.py`. Full test suite green
(216 passed). Pending upstreaming into the standalone memlife package.

### MF-005: Embedding versioning backfill on contradiction cleanup
**Priority:** Low  
**Source:** Hermes operational experience (Gary Qdrant incident)

When the embedding model changes, memlife detects stale vectors and backfills
automatically for facts. Contradictions with embeddings should get the same
treatment. If contradiction embeddings become stale after a model swap, they
should be backfilled, not left as orphaned vectors.

### MF-006: Separate GC pruning from VACUUM
**Priority:** Medium  
**Source:** Ingrid backend audit / OpenClaw code review, July 2026

`MemoryStore.run_gc()` currently both deletes stale rows and runs `VACUUM`
inside the same critical section. Even with WAL and `busy_timeout`, `VACUUM`
needs an exclusive database lock to rebuild the file and can stall active MCP
turns long enough to time out. MF-002 fixes routine writer contention but does
not address heavy maintenance on the hot path.

**Fix:** Split `run_gc()` into two operations:

- `run_gc()` — prune superseded facts, journal, runs, metrics, and reflected
  queue entries. Safe to run on a schedule.
- `run_vacuum()` — reclaim disk space via `VACUUM`. Expose as a separate
  maintenance-mode operation and only run it when the store is idle (no
  active MCP turn for N seconds) or via explicit CLI invocation.

This keeps lightweight pruning on the hot path and moves the expensive
file-rebuild operation off it.

**Status:** Confirmed patched in Ingrid backend (`ingrid/memory/store.py`):
`run_gc()` no longer runs `VACUUM`; new `run_vacuum()` method exposed via
CLI `ingrid vacuum`, REPL/TUI `/vacuum`, HTTP `/vacuum`, and daemon task
`vacuum`. Full test suite green (215 passed). Pending upstreaming into the
standalone memlife package.

### MF-007: Weighted containment in `store_fact`
**Priority:** High  
**Source:** OpenClaw onboarding review, July 2026

The second deduplication layer in `store_fact` uses simple substring
containment: if one fact's content is contained in another, the shorter one is
discarded. This conflates string length with semantic value and causes two
failure modes:

1. **Nuance erasure:** a general truth ("The system is stable.") gets swallowed
   by a conditional version ("The system is stable during peak loads."), even
   though both are worth keeping.
2. **Noise absorption:** a concise, high-confidence fact is discarded because
   it is a substring of a longer, noisier fact full of filler or citations.

**Fix:** Replace blind containment with weighted containment. First
iteration:

- Compute the set of non-stop-word tokens in the symmetric difference between
  the two facts.
- If the extra tokens contain meaningful nouns, verbs, or modifiers, skip
  containment and let the semantic-merge layer (cosine similarity) decide.
- If the confidence difference is large, prefer the higher-confidence fact as
  the retained "core" truth rather than always keeping the longer one.

This keeps the fix zero-dependency and surgical. A future iteration may add
`core_fact` / `detailed_fact` metadata, but that is out of scope for the first
pass.

**Status:** Confirmed patched in Ingrid backend (`ingrid/memory/store.py`,
`ingrid/config.py`, `ingrid/agent.py`); full test suite green (214 passed).
Pending upstreaming into the standalone memlife package.

## Notes

### MF-008: `fact_conflict_threshold` not initialized in `MemoryStore.__init__`
**Priority:** Critical
**Source:** Second-agent code review, July 2026

`self.fact_conflict_threshold` is referenced in `check_conflicts()` but is
never initialized in `MemoryStore.__init__`. Calling `store.check_conflicts()`
raises `AttributeError`. This is exposed through the public API including
`SyncMemoryStore.check_conflicts()`.

**Fix:** Initialize both thresholds from `MemoryConfig`:

```python
self.fact_merge_threshold = config.fact_merge_threshold
self.fact_conflict_threshold = config.fact_conflict_threshold
```

Add matching fields to `MemoryConfig` if they do not already exist. This is a
blocking bug for any consumer using contradiction detection.

**Status:** Confirmed patched in Ingrid backend (`ingrid/memory/store.py`);
tests pass. Pending upstreaming into the standalone memlife package.

### MF-009: Episode pruning missing from `run_gc()`
**Priority:** High
**Source:** Second-agent code review, July 2026

`run_gc()` prunes superseded facts, journal, agent runs, checkpoints, metrics,
and reflection queue entries, but it never deletes from the `episodes` table or
the `episode_tools` index. Episodes therefore grow indefinitely, which directly
contradicts memlife's core thesis of graceful degradation.

**Fix:** Add episode pruning to `run_gc()` with a configurable retention
period (e.g. `gc_episodes_days=180` in `MemoryConfig`). Prune old episodes and
clean up orphaned `episode_tools` rows:

```python
cur = self.conn.execute(
    "DELETE FROM episodes WHERE created_at < ?", (cutoff_episodes,))
pruned["episodes"] = cur.rowcount
self.conn.execute(
    "DELETE FROM episode_tools WHERE episode_id NOT IN "
    "(SELECT id FROM episodes)")
```

**Status:** Confirmed patched in Ingrid backend (`ingrid/memory/store.py`,
`ingrid/config.py`, `ingrid/agent.py`); full test suite green (210 passed).
Pending upstreaming into the standalone memlife package.

### MF-010: Hardcoded "Ingrid" agent name in Reflector prompt
**Priority:** Medium
**Source:** Second-agent code review, July 2026

The Reflector system prompt says "You are Ingrid's reflective faculty." This
is a single-agent leftover in what should be a general-purpose library. Any
other consumer of memlife gets reflections framed as Ingrid.

**Fix:** Make the agent name configurable with a sensible default:

```python
def __init__(self, ..., agent_name: str = "the agent"):
    self.agent_name = agent_name
```

Use `f"You are {self.agent_name}'s reflective faculty."` in the prompt.

**Status:** Confirmed patched in Ingrid backend (`ingrid/journal/reflection.py`);
full test suite green (218 passed). Pending upstreaming into the standalone
memlife package.

### MF-011: Hardcoded model names in Reflector and adapters
**Priority:** Medium
**Source:** Second-agent code review, July 2026

`Reflector` and the Ollama adapter hardcode model identifiers such as
`qwen3.5:cloud`, `deepseek-v4-flash:cloud`, and `kimi-k2.7-code:cloud`. These
are deployment-specific names that will not exist for other users, leading to
confusing failures.

**Fix:** Remove model-specific defaults. Require callers to pass a model name,
raise a clear error if unset, and document the requirement in the adapter and
`Reflector` constructors.

**Status:** Confirmed patched in Ingrid backend (`ingrid/journal/reflection.py`
and `ingrid/models/ollama.py`); full test suite green (218 passed). Pending
upstreaming into the standalone memlife package.

### MF-012: SQL injection in `import_jsonl` via column names
**Priority:** High
**Source:** Second-agent code review, July 2026

`import_jsonl()` interpolates JSONL keys directly into the SQL INSERT
statement. A malicious backup/migration file can inject arbitrary SQL via
column names such as `"id) VALUES (1); DROP TABLE facts; --"`.

**Fix:** Whitelist allowed columns per table and reject unknown keys before
building the query. Values are already parameterized; only the column names are
vulnerable.

### MF-013: `_tokenize` drops tokens shorter than 3 characters
**Priority:** Medium
**Source:** Second-agent code review, July 2026

`_tokenize()` filters out tokens with `len < 3`. Important short terms such as
"AI", "ML", "Go", "C", "OS", and "Py" become invisible to keyword search. A
query for "AI deployment" is reduced to just "deployment".

**Fix:** Lower the minimum token length to 2, or replace the length filter
with a proper stop-word list. At minimum, document the behavior.

**Status:** Confirmed patched in Ingrid backend (`ingrid/memory/store.py`);
full test suite green (218 passed). Pending upstreaming into the standalone
memlife package.

### MF-014: `DummyEmbedder` hash vectors produce misleading cosine similarity
**Priority:** Medium
**Source:** Second-agent code review, July 2026

`DummyEmbedder` uses hash-based vectors. Semantically similar sentences can
receive negative cosine similarity, while unrelated sentences get near-zero.
This makes semantic merge and conflict detection behave worse than random when
the dummy embedder is used.

**Fix:** Replace hash vectors with a bag-of-words approach so similar
sentences receive positive cosine similarity:

```python
def _bow_vector(text, dim=128):
    tokens = re.findall(r"[a-z0-9_]+", text.lower())
    vec = [0.0] * dim
    for tok in tokens:
        vec[hash(tok) % dim] += 1.0
    return vec
```

### MF-015: Lexical contradiction threshold too aggressive for small fact sets
**Priority:** Medium
**Source:** Second-agent code review, July 2026

In `reflection.py`, `threshold = max(2, len(facts) * 0.3)`. For 3 facts the
threshold is 2, so any term appearing in all 3 facts is treated as "common"
and filtered out. This causes contradictions to be missed in small fact sets.

**Fix:** Use a higher multiplier or a fixed minimum:

```python
threshold = max(3, len(facts) * 0.5)
```

**Status:** Confirmed patched in Ingrid backend (`ingrid/journal/reflection.py`);
full test suite green (218 passed). Pending upstreaming into the standalone
memlife package.

### MF-016: Robustness and correctness pass
**Priority:** Low
**Source:** Second-agent code review, July 2026

A collection of smaller correctness, API-hygiene, and documentation issues
that do not each need their own backlog item but should be swept in one pass:

- `SyncMemoryStore._run()` swallows the "already running" `RuntimeError` then
calls `asyncio.run()` from inside a running loop; fix the exception path.
- `MemoryStore._lock` only guards connection creation, not subsequent DB
access; document or serialize concurrent access.
- `get_last_checkpoint()` does not handle `json.JSONDecodeError` on corrupt
state.
- `OllamaInterface.session` creates `aiohttp.ClientSession()` outside an async
context.
- `OpenAIChat.chat()` does not handle an empty `response.choices` list.
- `recall_journal_vector()` sets `_score` to raw cosine similarity instead of
the unified `sim × confidence × recency` formula used by other recall
methods.
- Contradictions store fact IDs in `source_episodes_json`; rename or document
this overload.
- `Reflector` duplicates decay/config parameters instead of accepting a
`MemoryConfig`.
- `embedding_model` column is missing from initial `CREATE TABLE`
statements and added only via `ALTER TABLE` in `_migrate()`.
- `consolidate_journal()` commits once per merge instead of batching.
- `search_journal()` and `search_episodes_by_keyword()` load 500 rows then
filter in Python instead of using SQL `LIKE`.
- `store.py` is ~1,800 lines; consider splitting into focused modules.
- README "No-LLM Mode" example uses `await` outside an async function.
- `run_gc()` docstring references a stale `ingrid-db-backup.sh` script.
- `sentence_transformers` adapter uses deprecated `asyncio.get_event_loop()`.
- `MemoryStore` lacks `__enter__` / `__exit__` context manager support.
- MCP server has no shutdown cleanup for store, embedder, or sessions.
- `_normalize()` only strips trailing periods, not other punctuation.
- `recall()`, `recent()`, etc. do not validate `limit`; negative values
return all rows in SQLite.
- `_dedupe_jaccard()` returns a `frozenset` where the `_Candidate` type expects
a string.
- `gc.py run_gc()` wrapper ignores `MemoryConfig` values and uses its own
hardcoded defaults.

**Status:** Partially patched in Ingrid backend (`ingrid/memory/store.py`,
`ingrid/journal/reflection.py`, `ingrid/agent.py`, `ingrid/server.py`,
`ingrid/repl.py`, `ingrid/tui.py`, `ingrid/daemon/tasks.py`,
`ingrid/models/ollama.py`). Completed items include SQL `LIKE` filtering with
preserved match-count ranking for `search_journal()` and
`search_episodes_by_keyword()`, serialized DB access via `_LockedConn` proxy
around `RLock` plus `transaction()` context manager (with `_supersede_fact`
and `trace_event` wrapped for atomicity), `run_gc()` / `run_vacuum()` split,
corrupt checkpoint handling, `MemoryStore` context-manager support, `limit`
validation, unified journal vector scoring, batched `consolidate_journal()`,
`embedding_model` columns in the initial schema, broader `_normalize()`
punctuation stripping, and the stale backup-script docstring fix. Items left
as latent/standalone-only: `OllamaInterface.session` async-context creation
(works in practice), `OpenAIChat` empty choices, `DummyEmbedder`,
`SyncMemoryStore._run()`, `import_jsonl` column whitelisting, sentence-
transformers `get_event_loop()`, MCP server cleanup, `Reflector` config
parameter duplication, `source_episodes_json` naming, `store.py` module split,
and README no-LLM example. Full test suite green (219 passed).

## Notes

- MF-001, MF-002, and MF-008 are bug fixes that should land before any V2
  architecture work. MF-008 is critical and blocks contradiction detection.
- MF-003 is an API design issue that affects every consumer.
- MF-004 is the decay thesis extended to contradictions — it's core, not creep.
- MF-005 is a consistency gap, low priority.
- MF-009 is the decay thesis extended to episodes — also core, not creep.
- All items should be verified against Ingrid and Nano as testbeds before release.

---

# memlife V2 Backlog

Items derived from a critical read of Mnemosyne (github.com/mnemosyne-oss/mnemosyne),
organised into two buckets:

- **Core:** extends the four-layer model without new dependencies. Works in
  zero-dependency / no-LLM mode.
- **Infrastructure:** adds optional capabilities, often with new dependencies.
  These must be opt-in and degrade gracefully.

## V2 Core (no new dependencies)

### MV2-001: Tiered episodic degradation
**Priority:** High

Mnemosyne keeps episodic memories at full fidelity, then compresses them
through degradation tiers over time. memlife currently only has confidence
decay and GC. Add explicit content tiers for episodes:

- Tier 1: full episode text.
- Tier 2 (e.g. 30 days): LLM-summarised episode, 0.5× recall weight.
- Tier 3 (e.g. 180 days): key-signal extraction, 0.25× recall weight.

The actual compression must be optional and fallback-aware: use the injected
`model_chat` when available; otherwise fall back to a keyword/signal extract
so no-LLM mode keeps working.

**Why core:** it extends the Decay/Prune layer, not a new layer.

### MV2-002: Hybrid retrieval scoring
**Priority:** High

Mnemosyne's recall blends vector, FTS, and importance scores. memlife's
unified score is `relevance × confidence × recency`, where relevance is
currently opaque. Decompose relevance into:

- `relevance = 0.5*vector + 0.3*fts + 0.2*source_weight`
- normalised per query.

Expose the formula and weights in `MemoryConfig` so callers can tune them.

**Why core:** improves existing retrieval without new dependencies.

### MV2-003: Temporal triple store
**Priority:** Medium

Mnemosyne separates temporal triples (single current truth) from
append-only annotations. memlife handles truth revision via `memory_revise`
and supersession, but the time axis is implicit. Add an optional
`temporal_triples` table:

- `subject`, `predicate`, `object`, `valid_from`, `valid_until`, `fact_id`.
- `memory_revise` writes a new triple and closes the previous one.
- Queries can ask for current truth or as-of-date.

Default to the existing facts table if unused; no breaking change.

**Why core:** strengthens the Facts layer without new dependencies.

### MV2-004: AnnotationStore for multi-valued metadata
**Priority:** Medium

Mnemosyne uses an append-only `AnnotationStore` for entity mentions,
extracted facts, dates, and sources. memlife episodes store tool calls as
JSON; annotations would generalise this.

Add an `annotations` table: `(memory_id, kind, value, source, confidence)`.
Use it for:

- entity mentions
- extracted dates
- source references
- tool outcomes

Search episodes/facts by annotation. Keep the existing schema intact.

**Why core:** enriches Episodes and Facts; no new dependencies.

### MV2-005: Veracity-weighted recall
**Priority:** Medium

Mnemosyne tags every memory with `stated` / `inferred` / `tool` / `imported`
/ `unknown` and uses it as a recall multiplier. memlife already has `source`
(`user`, `agent`, `journal`) and confidence capping.

Map sources to veracity weights and apply a small multiplier during
retrieval, keeping confidence as the primary signal. For example:

- `user`: 1.0
- `agent`: 0.9
- `tool`: 0.95
- `journal`: 0.85
- `imported`: 0.8

Make weights configurable.

**Why core:** refines confidence handling; no new dependencies.

### MV2-006: Recall path diagnostics
**Priority:** Low

Mnemosyne tracks how often recall falls back to weak substring scanning vs.
FTS/vector. memlife already exposes `/stats` and `/health`; add recall
path counters:

- vector hits
- FTS hits
- fallback (substring) hits
- empty results

Surface in `memlife://stats` or a new `memlife://recall-diagnostics`
resource. Keep it lightweight and optional.

**Why core:** operational visibility; no new dependencies.

### MV2-007: Layer-aware configurable decay
**Priority:** Medium
**Source:** OpenClaw onboarding review, July 2026

The recency term in `vectors.py` uses a single hard-coded decay constant
for all memory layers. Over time this flattens episodic detail and
long-term identity facts into the same forgetting curve, and it lets a
low-confidence new fact outrank a high-confidence old fact simply because
it is recent.

**Fix:** Move decay parameters into `MemoryConfig` and make
`recency_weight()` layer-aware:

- `fact_decay_halflife` — slow (default ~1 year) for identity-grade facts.
- `episode_decay_halflife` — fast (default ~7 days) for transient events.
- `journal_decay_halflife` — medium (default ~30 days) for inferred
  beliefs.
- `confidence_decay_floor` — above this threshold, decay is slowed so
  high-confidence truths do not erode as quickly.

Keep the existing `relevance × confidence × recency` formula; only expose
the decay constants and add the confidence floor. This is a tuning change,
not a new dependency.

**Why core:** refines the existing Decay layer; no new dependencies.

### MV2-008: Temporal gap markers
**Priority:** Low

From Mastra's Observational Memory: insert a lightweight marker when a
meaningful time gap passes between messages in a thread (default ~10
minutes, configurable). The marker is stored as a transient episode entry
and helps both the agent and downstream consumers see that a conversation
resumed after a pause.

Use cases:

- Anchor observations to real-world time ("User asked about deployment
  after a 2-day gap").
- Improve temporal reasoning without building a full timeline parser.
- Give UI consumers a cheap timeline hint.

Keep the implementation minimal: detect on episode insert, write a
synthetic episode with `kind="gap_marker"`, and include it in recall only
when the query has temporal cues.

**Why core:** extends the Episodes layer; no new dependencies.

### MV2-009: Journal as belief/opinion network
**Priority:** Medium

From Hindsight's four-network model: treat the journal layer less as a
private diary and more as a first-class belief network. Each journal entry
represents an inferred model of the user, world, or agent self, with
confidence and provenance.

Design exploration:

- Add optional `belief_type` to journal entries: `user_preference`,
  `world_model`, `agent_self`, `relationship`.
- Track `evidence_episodes` and `evidence_facts` as provenance.
- Allow confidence updates and explicit retirement when evidence shifts.
- Surface beliefs during reflection as structured "what I believe" context,
  separate from raw facts.

This does not replace existing journal entries; it gives them a schema
when callers opt in. The default journal remains free-form.

**Why core:** strengthens the Journal layer without new dependencies.

## V2 Infrastructure (opt-in, may add dependencies)

### MV2-I001: sqlite-vec native vector backend
**Priority:** High

memlife stores embeddings as JSON in SQLite. Mnemosyne uses `sqlite-vec`
virtual tables when available, with JSON/numpy fallbacks. Add an optional
`sqlite-vec` backend:

- Use virtual tables for vector search when the extension is available.
- Fall back to JSON embeddings if not.
- Add dimension guards and backfill on model change.

Package as `memlife[sqlite-vec]` or auto-detect at runtime.

**Why infrastructure:** adds a dependency/extension but is fully optional.

### MV2-I002: Binary vector compression
**Priority:** Medium

Mnemosyne's MIB binarization compresses 384-dim float32 vectors to 48 bytes
and uses Hamming distance. Add an optional `BinaryVectorStore` adapter:

- Use when storage is constrained or `sqlite-vec` is unavailable.
- Keep float32 as the default for accuracy.
- Allow per-store selection via config.

**Why infrastructure:** storage/performance optimisation; optional.

### MV2-I003: Structured MEMORIA-style extraction
**Priority:** Medium

Mnemosyne v3 extracts structured facts into `memoria_facts`,
`memoria_timelines`, `memoria_instructions`, `memoria_preferences`, and
`memoria_kg`. memlife's reflection already produces facts and journal
entries. Add optional structured extraction during reflection:

- facts, instructions, preferences, timelines, kg triples.
- Regex-based always-on path for no-LLM mode.
- Optional LLM-based path when `model_chat` is available.

Store in the existing facts/journal/annotations tables; do not create a
separate "MEMORIA" layer.

**Why infrastructure:** richer extraction, but requires an LLM for the full
path. Regex fallback keeps it usable without one.

### MV2-I004: Polyphonic recall (optional plugin)
**Priority:** Low

Mnemosyne has a polyphonic recall mode that fuses vector, graph, fact, and
temporal voices via RRF. memlife's unified score is intentionally simple.
Offer polyphonic recall as an optional retrieval strategy:

- Configurable voices.
- RRF fusion.
- Default remains unified score.

**Why infrastructure:** significant complexity; should be opt-in only.

## V2 Non-goals

These Mnemosyne features are deliberately not on the backlog because they
either duplicate existing memlife layers or pull the design in a different
direction:

- **Working-memory auto-injection:** memlife keeps working memory in the
  agent's message list, not the memory store. Moving it in would blur the
  agent/memory boundary.
- **L3 Persona layer:** memlife's journal already shapes tone privately. A
  separate persona tier would create two sources of truth about agent belief.
- **Scratchpad:** temporary reasoning workspace is the agent's context, not a
  memory-layer concern.
- **Sync subsystem:** useful, but separate from the memory lifecycle. Should
  be a separate package or optional module.
- **Multi-agent identity / collaborative attestation:** memlife is
  single-agent by design. Adding author/validator chains is out of scope
  unless the product scope changes.
- **Built-in local LLM consolidation chain:** memlife's reflection uses an
  injected `model_chat`. Adding a built-in local GGUF chain would add heavy
  dependencies and break the zero-dependency promise.

## Notes

- Core items must keep the zero-dependency contract: DummyEmbedder +
  DummyChat must still run store/retrieve/decay/GC without any new packages.
- Infrastructure items must degrade gracefully when their dependency is
  absent.
- The V2 list is additive to the MF-001..MF-005 bug-fix backlog above; fix
  the bugs before starting V2 architecture work.