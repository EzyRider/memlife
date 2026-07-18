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

## Next cycle — open items

The MF-001..MF-016 audit items are closed. What remains for the next reroll
is a mix of a targeted hotfix (0.6.8), documentation gaps, structural
cleanups, design items staged for 0.7.0, and a handful of CoPilot audit
findings that still need code verification. None of the non-hotfix items
are load-bearing; all are deliberate.

The order of priority for the next reroll is: **0.6.8 hotfix → 0.7.0
design decisions → unverified audit items → documentation → structural
→ refinements.** The hotfix ships regardless; the rest are argued about.

### 0.6.8 hotfix — thread safety & PRAGMA hardening

**Status:** Scope agreed, ready to implement. Point release, no design
changes bundled in.

Three confirmed issues from real consumer usage (Ingrid's TUI, MCP server,
ZeroClaw all poke at `_LockedConn` from multiple threads). The public API
*implies* thread safety; the implementation has gaps. Each fix is
surgical and isolated.

#### HF-001: `_LockedConn.cursor()` returns a context-managed proxy
**Priority:** High (correctness, thread-safety contract)

`_LockedConn` currently exposes no `cursor()` method, so all 246
`self.conn.*` call sites in the store go through `execute` / `commit` /
`rollback` / `transaction`. The gap is the streaming pattern in
`_gc.py:393` and similar: those iterators hold a raw `Cursor` after the
lock has been released for the next caller, which is a use-after-free in
the thread-safety sense (the cursor is still bound to the connection, but
the lock is gone). CoPilot framed this as "cursors leak from
`_LockedConn`," which is not quite the mechanism — `_LockedConn` doesn't
return cursors, the store layer does. The underlying issue is real: any
code path that holds a `Cursor` reference across method calls on the
proxy is unsafe.

**Fix:** add a `cursor()` method on `_LockedConn` that returns a
`_LockedCursor` context manager. The proxy acquires the lock for the
duration of `__enter__` / `__exit__` and the cursor is closed on exit.
Audit all `self.conn.execute(...)` call sites for ones that iterate the
returned cursor and convert those to `with self.conn.cursor() as cur:`
form. Expect 1-3 call sites in `_gc.py`; the rest are fire-and-forget
executes that are already safe.

#### HF-002: `row_factory` getter/setter wrap in `self._lock`
**Priority:** High (correctness, thread-safety contract)

`sqlite3.Connection.row_factory` is a Python-level attribute. Reading or
writing it on a `_LockedConn` instance bypasses the reentrant lock
entirely because the proxy has no `__getattr__` / `__setattr__` guard
for it. Two concurrent threads swapping `row_factory` mid-query will
race; the loser sees a `ProgrammingError` or a corrupted row tuple.

**Fix:** add explicit `row_factory` property (getter and setter) on
`_LockedConn` that acquires `self._lock` around the underlying access.
Audit all other `Connection` attributes that the proxy implicitly
forwards (via `__getattr__`) and decide case by case whether they need
the same treatment. The auditing list: `isolation_level`,
`text_factory`, `total_changes`, `iterdump`, `backup`, anything else
the store touches. Most can stay unlocked (read-only, no race); the
ones that mutate connection state need the wrapper.

#### HF-003: PRAGMA hardening — whitelist validation
**Priority:** High (security, SQL injection)

`PRAGMA journal_mode` and similar settings are interpolated via f-string
in `_set_pragma`. The pragma name and value come from `MemoryConfig`,
which is the only thing standing between user input and SQL execution.
A user setting `journal_mode = "WAL; DROP TABLE facts; --"` would
currently execute that, because the f-string does no escaping. This is
a low-likelihood attack vector (the library is local, the user owns
their config), but the fix is small and the bug is unambiguous.

**Fix:** add a `_validate_pragma_name(name)` helper with an explicit
allowlist (`journal_mode`, `synchronous`, `foreign_keys`, `busy_timeout`,
`cache_size`, `temp_store`, `mmap_size`) and a `_validate_pragma_value(name, value)`
helper that type-checks the value (integer vs string vs boolean) and
for string pragmas enforces a value whitelist (e.g. `journal_mode` ∈
`{DELETE, TRUNCATE, PERSIST, MEMORY, WAL, OFF}`). Apply both inside
`_set_pragma` before the f-string interpolation. Add a
`MemoryConfig.validate()` method that runs the same checks at config
construction time so misconfiguration fails fast at startup, not at the
first PRAGMA execution. Update the MemoryConfig docstring to call out
that PRAGMA keys and values are validated.

**Not in scope for 0.6.8:** rewriting `_set_pragma` to use parameterised
SQL. SQLite's PRAGMA syntax does not support bound parameters for
pragma names, so the f-string is unavoidable; the fix has to be
validation, not parameterisation.

#### Release plan
- Bump to 0.6.8, update `CHANGELOG.md` with the three items under a new
  `### Fixed` section.
- Add tests: (1) `cursor()` context manager releases the lock on exit and
  on exception; (2) `row_factory` setter blocks under contention;
  (3) `MemoryConfig.validate()` rejects bad pragma names and values;
  (4) `_set_pragma` raises on a bad name even if validation is bypassed.
- No new dependencies. No public API breakage (the additions are
  additive; existing call sites are unchanged).
- Target: same release hygiene as 0.6.7 (PyPI + GitHub tag, README
  "What's New" sync).

### Documentation

### MD-001: Vector backends reference doc missing
**Priority:** Medium  
**Source:** 0.6.0 audit section 2.4

The 0.6.0 roadmap calls for `docs/vector-backends.md` covering the ABC
(`memlife.vector_backends.base.VectorBackend`), the three shipped
implementations (`json_backend`, `binary_backend`, `sqlite_vec_backend`),
selection guidance (speed vs memory vs native), and how to plug in a custom
backend. File does not exist. Consumers writing custom backends currently
have to read the source.

### MD-002: Namespaces guide missing
**Priority:** Medium  
**Source:** 0.6.0 audit section 2.4

`docs/namespaces.md` was planned to cover the `switch_namespace()` lifecycle,
the "stateless embedder" rule (per-namespace cache keys on namespace,
shared embedder only when model + dim match), and the `clone_to_namespace()`
post-0.5.0 follow-up. File does not exist. Currently documented only inline
in the source and in the ROADMAP.

### MD-003: Reflection audit doc missing
**Priority:** Low  
**Source:** 0.6.0 audit section 2.4

`docs/reflection-audit.md` was planned to capture the operational telemetry
from running the reflector on a real store (cycle counts, contradiction
yield, retirement rates, false-positive rate). Superseded by the
`refactor-store-split.md` document for the refactor side, but the audit
findings themselves are not written down anywhere outside the 0.6.0 roadmap
file.

### MD-004: `ROADMAP.md` is stale
**Priority:** Low  
**Source:** Audit pass, July 2026

`docs/ROADMAP.md` documents the 0.4.1 → 0.5.0 → 0.6.0 plan. Everything it
covers for 0.5.0 (namespaces, vector backend ABC, reflection audit) and
0.6.0 (all MF-001..016 items) has shipped. Two options:

- Archive: rename to `ROADMAP-0.4-to-0.6-historical.md` and replace the file
  with a pointer to the current 0.6.0 roadmap and this backlog.
- Refresh: rewrite to cover 0.7.0 (see NX-001 below).

No action required if you prefer the 0.6.0 roadmap to remain the single
source of truth.

### Structural

### ST-001: `vec_backend.py` legacy module
**Priority:** Medium  
**Source:** Audit pass, July 2026

`src/memlife/vec_backend.py` predates the `vector_backends/` ABC. It is
still actively used:

- Re-exported from `memlife/__init__.py`.
- Imported by three test files (`test_sqlite_vec.py`,
  `test_sqlite_vec_backfill.py`, `test_sqlite_vec_recall.py`).
- The `refactor-store-split.md` plan still lists it as a dependency of the
  planned `_embeddings.py` mixin.

The 0.5.0 plan said "migrate current helpers, then remove once migrated."
That step was never done. Two clean paths:

- **Wrap:** keep the module, but make it a thin re-export over
  `vector_backends.sqlite_vec_backend.SqliteVecBackend` for the tests and
  any third-party consumers, and document the module as a zero-deps
  convenience entry point.
- **Remove:** migrate the three test files to the ABC and delete
  `vec_backend.py`. Breaks any external consumer that imported it.

Recommendation: wrap, not remove, until the next major version.

### ST-002: Sync store fate
**Priority:** Low  
**Source:** 0.6.0 roadmap §"Sync subsystem" / V2 non-goals

`SyncMemoryStore` is a functional thin wrapper around the async store and
is still tested and shipped. The 0.6.0 roadmap lists it as a V2 non-goal
("a separate package") but no extraction has happened. Options:

- Keep as-is — the wrapper is small, well-tested, and the back-compat
  cost is low.
- Document the deprecation intent in the module docstring and a
  `DeprecationWarning` on import, with a target version for extraction.
- Extract to `memlife-sync` companion package.

Recommendation: keep as-is, add a docstring note that it is maintained but
not the primary API.

### ST-003: Hardcoded model defaults in adapters
**Priority:** Low  
**Source:** MF-016 follow-up

Three adapter default arg names bake in model names:

- `OllamaInterface(..., model="mxbai-embed-large:latest", ...)`
- `OpenAIEmbedder(..., model="text-embedding-3-small", ...)`
- `OpenAIChat(..., model="gpt-4o-mini", ...)`

These are sensible defaults but make the assumption that "if you didn't
specify, you wanted the cheap one." For a library this is normal; for a
project where everyone uses Ollama locally, the default is wrong about
half the time. Two options:

- Add a `MemoryConfig.embedder_default_model` / `chat_default_model` knob
  that the adapters consult when called with no model arg.
- Leave the defaults but document the assumption clearly in the adapter
  docstrings.

Recommendation: option 2 unless the adapters are being called from code
that doesn't already specify the model.

### Refinements (low priority, not urgent)

### RF-001: MF-005 contradiction embedding backfill
**Status:** Documentation only, not a code change

`backfill_embeddings()` in `_gc.py` intentionally skips rows with
`type = 'contradiction'`. MF-005 originally called for re-embedding
contradictions on model swap. The current skip is a deliberate design
choice: contradictions are short, retrieval-routed, and re-embedding them
on every model bump would amplify drift between fact and contradiction
vectors for no real benefit (contradiction lookups use exact pair matching,
not vector search). Add a comment in `_gc.py` next to the `!=` clause
explaining this, and a note in MD-001.

### RF-002: `source_episodes_json` field naming
**Priority:** Trivial

The same JSON field is called `source_episodes_json` across facts, journal
entries, and contradictions, but for contradictions it is populated with
`source_facts` (the two fact IDs in the conflicting pair). The MF-016
audit flagged this. Either:

- Rename to `source_provenance_json` (breaking schema change, needs a
  migration).
- Leave as-is and document the semantics per row type in MD-001.

Recommendation: leave as-is. The cost of the rename outweighs the value.

### RF-003: Legacy lexical scoring paths
**Status:** Not a bug, leave as-is

`recall_journal_vector` uses cosine * alpha + recency * beta, a unified
score. The legacy text-only paths still set `_match_score` and
`_text_score` directly (lexical match counts divided by token count). The
MF-016 audit flagged this as inconsistency, but the two paths are different
retrieval modes with different scoring semantics. The vector path is the
primary one; the lexical fallback is a degraded mode and its score
formula is intentionally simple. No change needed.

### Future direction

### NX-001: 0.7.0 themes
**Priority:** Subject to discussion  
**Source:** 0.6.0 audit, production usage across Ingrid/ZeroClaw/OpenClaw,
CoPilot audit pass (July 2026)

The next major-version reroll is not yet committed. The themes below are
candidates, not a roadmap. Each one has a real consumer pull; none are
feature creep. Theme ordering is by how much they unblock other work.

**A. Multi-agent identity and author attribution (highest priority).**

The store currently has no concept of *who* wrote a fact. Facts have
`source` (string) and episodes have `agent_id` (string), but the two
fields are independent, inconsistently populated, and never queried
together. Now that memlife is shared across Ingrid (Python-native),
ZeroClaw (Rust via MCP), and OpenClaw (Node.js via MCP), the lack of
attribution is biting: when two agents disagree about the same fact,
there is no way to ask "who said what, and when."

Concrete items:

- **Author field on every write path.** `store_fact()`, `add_journal_entry()`,
  `add_contradiction()`, and the episode recorder each take an
  `author: str` parameter. Default to `"system"` for backward compat;
  log a deprecation warning when the default fires.
- **Attribution-aware recall.** New `recall_by_author(author, query, ...)`
  helper that filters the candidate set by author before scoring. This
  is the foundation for "what does Ingrid think vs what does OpenClaw
  think" queries.
- **Conflict surfaces between authors.** When `add_contradiction()` is
  called, record both authors. The contradiction row gains
  `author_a`, `author_b`, and the existing `source_facts` slot carries
  the pair. This lets downstream agents see the disagreement
  structure, not just the fact pair.
- **Author identity not enforced.** Two agents with the same
  `author` string are treated as the same writer. No auth, no
  namespace ownership. This is a library, not a service; the caller
  owns the identity layer.

**B. Shared memory scopes for cross-agent recall.**

Today every agent gets its own namespace. That's correct for private
memory but wrong for shared knowledge ("James's home address is
3 Earle Close" is true for all three agents and shouldn't be
re-stored three times). A `scope` dimension on the namespace —
`"private" | "shared"` — addresses this without breaking the
per-agent namespace model.

Concrete items:

- `MemoryConfig(scope="private" | "shared")`. Default `"private"` for
  backward compat.
- `MemoryStore.list_namespaces()` helper that enumerates the namespaces
  the current process can see (its own private + all shared). Returns
  a list of `Namespace` objects with name, scope, and db path.
- `MemoryConfig.db_path_for(namespace, scope=None)` helper that
  computes the canonical DB path for a namespace, including scope
  suffixing. This replaces the ad-hoc path construction currently
  scattered across `_store.py` and the namespace module.
- Shared namespaces are read-write for the owning process and
  read-only-by-default for other processes. Explicit
  `MemoryConfig(shared_write=True)` opt-in for the case where two
  agents genuinely need to mutate the same shared store (rare;
  mostly the right answer is "one writer, many readers").
- Recall falls back to shared namespaces automatically when a private
  recall returns no results. Tunable via `MemoryConfig.recall_scope_order`
  (default `["private", "shared"]`).

**C. Inspectable and correctable reflection loop.**

The reflector ships and is tested, but it is opaque: you can run it
and see the resulting journal entries, but you cannot inspect its
in-progress reasoning, and you cannot correct a single bad step
without re-running the whole cycle. For a library that is meant to
back personal agents running unattended, this is the gap that
matters most in production.

Concrete items:

- **Cycle trace.** A `Reflector.run_cycle(trace=True)` mode that
  returns a `CycleTrace` object: the candidates it considered, the
  contradictions it found, the retirement decisions, and the journal
  entries it wrote. Caller can `cycle_trace.to_dict()` to persist or
  surface in a TUI.
- **Step-level veto.** `Reflector.run_cycle(plan_only=True)` returns
  the planned changes without committing. Caller reviews and either
  calls `apply_plan(plan)` or `discard_plan(plan)`. This is the
  "correctable" half of the pair.
- **Human-in-the-loop toggle.** `MemoryConfig.reflection_mode` ∈
  `{auto, plan_only, disabled}`. Default `auto`. `plan_only` is
  for the TUI case where James wants to see what the reflector is
  about to do before it does it.

**D. Embedding model migration tooling.**

The content-addressable embedding cache shipped in 0.6.0 and works
well, but the *migration* path when the embedder model changes is
still "call `backfill_embeddings()` and hope you remember to." For
local Ollama users who swap models frequently, this is a footgun.

Concrete items:

- **Startup mismatch warning.** `MemoryStore.__init__` checks the
  configured embedder's `(model, dim)` against the most-recent cache
  entry. If they don't match, log a clear warning with the
  one-line migration command.
- **One-shot `migrate_embeddings()` command.** Already exists as
  `backfill_embeddings()`, but the new entry point is a single-call
  migration that re-embeds only the rows whose cache entry's
  `(model, text_hash)` doesn't match the current embedder, and is
  idempotent. The current `backfill_embeddings()` is broader
  (re-embeds everything) and should be kept as the "I changed my
  mind and want to start over" tool.
- **Per-model cache partitioning.** The cache is already
  content-addressable by `(model, text_hash)`, so this is mostly
  already done. The work is to expose the partitioning in the GC
  output and the new `list_namespaces()` helper so users can see
  how much of their cache is "stale" (model no longer in use).

**E. (lower priority) Async-vs-sync parity tests.**

`SyncMemoryStore` is a thin wrapper around the async store. The
behavioural test suite runs against the async store. A parity
matrix — same tests, both stores, same expected output — would
guarantee no drift. No such matrix exists today. The work is
mostly test infrastructure: a pytest marker that runs the suite
twice with `@pytest.mark.parametrize("store", [async_store,
sync_store])`.

**F. (lower priority) Belief network tooling.**

The belief network (journal `belief_type`, `link_journal_entries`)
is shipped and tested but has no first-class query path. A
`query_belief_graph(anchor, depth)` helper would let downstream
agents ask "what do I believe about X, and why" without writing
recursive SQL. Mostly a query-API addition, not a storage
change.

**G. (lower priority) Plugin hooks for custom retrieval voices.**

The polyphonic recall system has fixed voices today. A
`register_voice()` API would let downstream agents inject
domain-specific retrieval without forking. This is a pure
extensibility play and depends on no other 0.7.0 work.

**Not in scope for 0.7.0:** the `vec_backend.py` legacy module
(ST-001), sync store extraction (ST-002), and the
`source_episodes_json` rename (RF-002). These are 0.6.x material
at best and shouldn't hold up 0.7.0.

### Unverified — CoPilot audit findings

**Status:** Items below were flagged by a CoPilot audit pass (July
2026). They are recorded here so they don't get lost, but the
mechanism, severity, or even the existence of the bug has not been
confirmed against the current code. **Do not start work on these
without a code-verification pass first.** The audit had a tendency
to surface real concerns in the wrong package (see HF-001 and HF-002
above, which are real bugs but not for the reasons CoPilot cited).

#### UV-001: CoPilot's "246 raw cursor leaks"
**CoPilot claim:** `_LockedConn` returns raw cursors that leak the
lock; 246 call sites affected.
**Verification status:** Mechanism corrected in HF-001 above. The
count of 246 is also wrong — those are `self.conn.execute(...)` /
`commit()` / `rollback()` / `transaction()` call sites, of which only
the ones that iterate the returned cursor are affected. Expect 1-3
real conversion sites, mostly in `_gc.py`. **Action:** fold into
HF-001; the audit's number is not the scope.

#### UV-002: CoPilot's "row_factory" diagnosis
**CoPilot claim:** `row_factory` is mutated without lock acquisition.
**Verification status:** Mechanism confirmed in HF-002. The audit
underestimated the scope: every connection-state-mutating attribute
(`isolation_level`, `text_factory`, anything that writes back to the
C-level connection) has the same gap. **Action:** fold into HF-002;
HF-002 should include a one-pass audit of the full attribute list,
not just `row_factory`.

#### UV-003: Async store lock-scope review
**CoPilot claim:** The async store's lock acquisition may not cover
the full mutation path on `store_fact()`.
**Verification status:** Unverified. The async store was not read
in the audit pass. The claim is plausible — async lock discipline is
harder to get right than sync — but I have not looked. **Action:**
before 0.6.8 ships, open `src/memlife/_async_store.py` (or wherever
it lives), trace one full `store_fact()` call from start to commit,
and check that every mutation is inside the lock. If gaps exist,
file a new HF item; if not, expire this finding.

#### UV-004: Reflector output stability across LLM providers
**CoPilot claim:** Reflector prompt is tuned for one provider; output
structure drifts on others.
**Verification status:** Unverified, low confidence. The reflector
prompt is stringly structured (numbered list, "Contradictions:"
header) but I have not tested it against the OpenAI adapter. The
extraction layer (the part that parses the reflector's output into
journal rows) is the failure surface, not the prompt itself.
**Action:** add a parameterised test that runs the reflector with
the same input through the Ollama, OpenAI, and DummyChat adapters and
checks the parsed output is structurally equivalent (same number of
journal rows, same `belief_type` distribution, no parse errors).
If the test passes on all three, expire this finding.

#### UV-005: Embedding cache key collision under text-hash truncation
**CoPilot claim:** If `text_hash` is truncated to N bytes, two
distinct texts can collide.
**Verification status:** Unverified, very low confidence. CoPilot
did not say what N is, and the cache uses a full SHA-256 (per
`_cache_lookup` and `_cache_store` in `_gc.py`). The audit may have
been looking at an older code path. **Action:** before expiring,
read `_cache_lookup` and `_cache_store` to confirm the hash function
and length. If it's SHA-256, expire this finding. If it's anything
weaker, file a new HF item.

**No unverified items become backlog items without a code-verification
pass.** The 0.6.8 hotfix is the priority; everything in this section
is parallel work that should not block the release.
