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

**Status:** Confirmed patched in Ingrid backend and upstreamed into the
standalone memlife package (`src/memlife/store.py`, `src/memlife/config.py`).

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
`last_contradiction_scan` and `reflection_cycle`. Full suite green.
Upstreamed into the standalone memlife package.

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
contradictions older than `contradiction_retirement_cycles` (default 14, configurable) are retired.
Upstreamed into the standalone memlife package (`src/memlife/_journal.py`).

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
`run_gc()` no longer runs `VACUUM`; `run_vacuum()` is separate.
Upstreamed into the standalone memlife package (`src/memlife/_gc.py`).

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

**Status:** Confirmed patched in Ingrid backend and upstreamed into the
standalone memlife package (`src/memlife/_facts.py`).

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

**Status:** Confirmed patched in Ingrid backend and upstreamed into the
standalone memlife package (`src/memlife/store.py`).

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

**Status:** Confirmed patched in Ingrid backend and upstreamed into the
standalone memlife package (`src/memlife/_gc.py`).

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

**Status:** Confirmed patched in Ingrid backend and upstreamed into the
standalone memlife package (`src/memlife/reflection.py`).

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

**Status:** Confirmed patched in Ingrid backend and upstreamed into the
standalone memlife package (`src/memlife/reflection.py`,
`src/memlife/adapters/ollama.py`).

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

**Status:** Confirmed patched in Ingrid backend and upstreamed into the
standalone memlife package (`src/memlife/_episodes.py`).

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

**Status:** Confirmed patched in Ingrid backend and upstreamed into the
standalone memlife package (`src/memlife/reflection.py`).

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

------

# memlife V2 Status

All V2 items were implemented in memlife 0.4.0b0–0.4.1b0 and are shipped in
`main`. The original design notes are preserved below as a status record, not as
pending work.

- **Core V2:** complete and tested.
- **Infrastructure V2:** complete and opt-in.
- **Bug fixes MF-001..MF-016:** complete and upstreamed.
- **Current version:** 0.4.1b0
- **Test status:** 118 passed

Remaining architectural work: `store.py` mixin refactor (see
`docs/refactor-store-split.md`).

## V2 Core (no new dependencies)

### MV2-001: Tiered episodic degradation — DONE
Implemented via per-episode-type halflives in `MemoryConfig`:
- `episode_tool_success_halflife_days=21.0`
- `episode_failure_halflife_days=3.0`
- `episode_observation_halflife_days=1.0`

Episode properties `is_success`, `is_failure`, and `has_tool_calls` select the
correct tier during retrieval. See `tests/test_tiered_decay.py`.

### MV2-002: Hybrid retrieval scoring — DONE
Unified score is `relevance × confidence × recency`, with relevance blended
from vector, text, source, and veracity signals. Weights are exposed in
`MemoryConfig`:
- `recall_vector_weight=0.5`
- `recall_text_weight=0.3`
- `recall_source_weight=0.2`
- `recall_veracity_weight=0.05`

Signals are normalised per query. See `src/memlife/retrieval.py` and
`tests/test_retrieval.py`.

### MV2-003: Temporal triple store — DONE
`temporal_triples` table with methods:
- `store.store_fact_triple(...)`
- `store.current_truth(subject, predicate)`
- `store.truth_as_of(subject, predicate, as_of)`
- `store.triples_for_fact(fact_id)`
- `store.expire_triples_for_fact(fact_id)`

See `tests/test_temporal_triples.py`.

### MV2-004: AnnotationStore for multi-valued metadata — DONE
`annotations` table and methods:
- `store.annotate_fact(fact_id, kind)`
- `store.annotate_journal(journal_id, kind)`

`Episode`, `Fact`, and `JournalEntry` expose `.annotations`. See
`tests/test_annotations.py`.

### MV2-005: Veracity-weighted recall — DONE
Veracity bonus applied during retrieval; current temporal triples boost a
fact's veracity, expired triples do not. Configurable via
`recall_veracity_weight`. See `tests/test_veracity.py`.

### MV2-006: Recall path diagnostics — DONE
`retrieve(query, debug=True)` returns a structured result with per-candidate
`why` explanations. `store.recall_stats()` exposes path counters. See
`tests/test_recall_diagnostics.py`.

### MV2-007: Layer-aware configurable decay — DONE
Per-layer halflives in `MemoryConfig`:
- `fact_decay_halflife_days=365.0`
- `episode_decay_halflife_days=7.0`
- `journal_decay_halflife_days=30.0`

Decay floors: `fact_decay_floor=0.1`, `journal_decay_floor=0.15`.

### MV2-008: Temporal gap markers — DONE
`gap_marker_threshold_hours=24.0` triggers synthetic gap-marker episode
insertion when a meaningful time gap is detected. See
`tests/test_gap_markers.py`.

### MV2-009: Journal as belief/opinion network — DONE
Journal entries support:
- `belief_type` (`user_preference`, `world_model`, `agent_self`,
  `relationship`)
- `source_episodes` / `source_facts` provenance
- `link_journal_entries(..., relation, strength)` for belief-network edges
- retirement / supersession semantics

See `tests/test_belief_network.py`.

## V2 Infrastructure (opt-in, may add dependencies)

### MV2-I001: sqlite-vec native vector backend — DONE
`vec_backend.py` auto-detects `sqlite-vec`; `MemoryConfig.use_sqlite_vec`
controls it. Falls back to JSON embeddings when unavailable. See
`tests/test_sqlite_vec.py`.

### MV2-I002: Binary vector compression — DONE
`binary_vectors.py` provides `binarize`, `debinarize`, `hamming_distance`,
`hamming_similarity`. Binary embeddings are stored as `binary:dim:base64` and
decoded in `models.py`. See `tests/test_binary_vectors.py`.

### MV2-I003: Structured MEMORIA-style extraction — DONE
`memorias.py` provides regex-based extraction of facts, preferences,
instructions, timelines, and KG triples, plus `persist_extraction()` to store
them. Controlled by `MemoryConfig.memorias_extraction`. See
`tests/test_memorias.py`.

### MV2-I004: Polyphonic recall — DONE
`polyphonic.py` implements reciprocal-rank-fusion across retrieval voices.
`MemoryConfig.use_polyphonic_recall` toggles it; default unified score is
unchanged. See `tests/test_polyphonic.py`.

## V2 Non-goals

Unchanged — deliberately out of scope:

- Working-memory auto-injection
- L3 Persona layer
- Scratchpad
- Sync subsystem
- Multi-agent identity / collaborative attestation
- Built-in local LLM consolidation chain

## Notes

- Zero-dependency contract preserved: DummyEmbedder + DummyChat run the full
  lifecycle.
- Infrastructure features degrade gracefully when dependencies are absent.
- The MF-001..MF-016 bug fixes are upstreamed and live.
- Remaining work: `store.py` mixin refactor.
