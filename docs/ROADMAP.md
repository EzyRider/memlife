# memlife Roadmap

## Current state: 0.4.1 (stable)

memlife is now a production-stable, single-agent lifecycle memory system:

- Four memory tiers: episodes, facts, journal, decay/prune
- Unified scoring: `relevance × confidence × recency`
- Reflection loop with critic gating and contradiction lifecycle
- Temporal triples, annotations, veracity weighting, gap markers
- Optional sqlite-vec, binary vectors, polyphonic recall, MEMORIA extraction
- `store.py` refactored into focused mixins
- 118 tests passing, ruff clean

This roadmap defines the path to **0.5.0** and beyond. The guiding rule is:
**extend the lifecycle thesis, don't dilute it.** Every feature must either
improve graceful degradation or extend memlife to a new well-defined context.

---

## 0.5.0: Namespaces, Vectors, and Reflection Transparency

Target: three core deliverables that unlock hosted usage and improve trust.

### 1. Namespace isolation (multi-user, single-agent-per-namespace)

**Goal:** Let one memlife installation serve multiple users/agents without
memory leakage between them. Each namespace is still a single-agent memory,
preserving the lifecycle semantics.

**Why this first:** Multi-user isolation is a prerequisite for any hosted,
SaaS, or team deployment. It does not require changing the agent model.

**Design:**

```python
from memlife import MemoryStore, MemoryConfig

cfg = MemoryConfig(
    db_path="./memlife.db",
    namespace="julie",          # new
)
store = MemoryStore(config=cfg)
```

Two implementation strategies to evaluate:

| Strategy | Pros | Cons |
|----------|------|------|
| **A. Separate DB files** `memlife_<namespace>.db` | Zero schema changes, trivial backup/restore, natural sharding | More files, harder to query cross-namespace |
| **B. Shared schema with `namespace` column** | One file, global analytics possible, easier hosted ops | Schema migration required, every query needs namespace filter, risk of leakage if filter forgotten |

**Recommendation:** Start with **Strategy A** for 0.5.0. It keeps the change
surgical and preserves the single-file-per-agent mental model. Strategy B can
be explored later if hosted demand justifies it.

**API surface:**

- `MemoryConfig.namespace: str = "_default"`
- `MemoryConfig.db_path_for(namespace: str) -> str` helper
- `MemoryStore.list_namespaces()` — scan configured directory for `memlife_*.db`
- `MemoryStore.switch_namespace(name: str) -> MemoryStore` — returns a new store
  pointing at the namespaced DB, sharing the embedder where safe

**Namespace validation and safety:**

- Sanitize namespace values before using them in paths. Reject characters that
  could escape the configured directory: `/`, `\\`, `..`, control characters,
  and leading/trailing whitespace. A valid namespace matches `^[a-zA-Z0-9_-]+$`.
- `NamespaceError` (new exception) is raised for invalid names.
- Default namespace `_default` is reserved and maps to the original DB path
  behavior, preserving backward compatibility.
- **Case normalization:** namespaces are lowercased in `db_path_for()` to avoid
  silent collisions on case-insensitive filesystems (e.g. `Julie` and `julie`
  map to the same DB file). The original value is preserved in
  `MemoryConfig.namespace` for display/logging.
- Embedder sharing is safe only when the model and dimensions match.
  `switch_namespace()` docs must state this explicitly. Embedder instances
  should be stateless with respect to store data; any per-store cache keys on
  namespace. If an embedder implementation cannot guarantee that, create a
  fresh embedder on switch rather than sharing.

**Migration path (post-0.5.0 follow-up):**

- `MemoryStore.clone_to_namespace(name: str)` — copy the current DB's content
  into a new namespaced DB. Useful for users who start single-user and later
  want per-user split.

**Files to touch:**

- `src/memlife/config.py` — add namespace field, path helper, `validate()`
- `src/memlife/store.py` — resolve DB path in `__init__`
- `src/memlife/sync_store.py` — pass namespace through
- `src/memlife/mcp_server.py` — add `--namespace` / `MEMLIFE_NAMESPACE` CLI arg
- `tests/test_namespaces.py` — new test module

**Verification:**

- Two namespaces sharing a directory do not see each other's episodes/facts/journal
- `tmp_path` fixture: create two stores in the same temp dir with different
  namespaces and assert writes in one are invisible to the other.
- Switching namespaces creates/uses the correct DB file
- Default namespace preserves existing DB path (migration-safe)
- Invalid namespace values raise a clear error before touching the filesystem
- Embedder sharing does not leak vectors between namespaces with different models

---

### 2. sqlite-vec as a first-class vector backend

**Goal:** Move sqlite-vec from a silent fallback to a real indexed vector backend
that users can choose and trust.

**Current state:** `MemoryConfig.use_sqlite_vec` exists, but the path is
fallback-only and not well tested in CI (sqlite-vec is an optional dependency).

**Design:**

```python
cfg = MemoryConfig(
    db_path="./memlife.db",
    vector_backend="sqlite-vec",   # "json" | "sqlite-vec" | "binary"
)
```

Deprecate `use_sqlite_vec: bool` and `use_binary_vectors: bool` in favor of a
single `vector_backend: str` enum. Keep old flags as aliases for one release.

**Work items:**

1. **Backend enum:** use `VectorBackend` enum (or `Literal["json", "sqlite-vec", "binary"]`),
  not a plain `str`, for validation and IDE support.
2. **Auto-detection:** try to load `sqlite_vec` at runtime. If the user did
  **not** explicitly request `sqlite-vec`, fall back to JSON with a logged
  warning. If the user explicitly set `vector_backend="sqlite-vec"` and the
  package is not installed, raise a clear `RuntimeError` instead of silently
  falling back.
3. **Schema management:** create virtual tables per embedding kind
  (`vec_episodes`, `vec_facts`, `vec_journal`) on first connection.
4. **Dimension guards:** store expected dimension per table; reject mismatched
  vectors with a clear error instead of silent corruption. Provide a way to
  drop/recreate virtual tables and re-backfill when the embedding model changes.
5. **Explicit migration:** switching an existing DB to sqlite-vec does **not**
  auto-backfill. `backfill_embeddings()` is the user-facing opt-in; it logs
  progress and can be resumed.
6. **Backfill integration:** `backfill_embeddings()` populates sqlite-vec
  tables when active.
7. **CI:** add a parameterized test fixture that runs the vector recall suite
  against all three backends, plus a job that installs `memlife[sqlite-vec]`.
8. **Benchmarks:** add a small script comparing recall latency and accuracy
  across JSON, binary, and sqlite-vec backends on synthetic data.

**Files to touch:**

- `src/memlife/vec_backend.py` — expand to a proper backend manager
- `src/memlife/_embeddings.py` — route serialization through chosen backend
- `src/memlife/_schema.py` — create/drop virtual tables
- `src/memlife/config.py` — add `vector_backend`, deprecate booleans
- `pyproject.toml` — add `sqlite-vec` extra and update `all` extra
- `tests/test_sqlite_vec.py` — expand coverage; parameterize vector recall tests
- `docs/vector-backends.md` — new comparison doc

**Verification:**

- All vector recall tests pass with each backend
- Parameterize existing vector recall tests with a `vector_backend` fixture
- Backfill produces correct vectors in the active backend
- Dimension mismatch raises a clear error
- Explicit `vector_backend="sqlite-vec"` without the package installed raises
- CI runs sqlite-vec job

---

### 3. Reflection audit and correction propagation

**Goal:** Make the reflection loop inspectable and correctable. If the user
rejects a reflected belief, memlife should remember the correction and the
reflector should avoid regenerating it.

**Why this matters:** Reflection is the engine of learning, but right now it's
a black box. Users need to see what was proposed, what was kept, and what was
dropped — and they need a way to steer it.

**Design:**

```python
# New public API on MemoryStore
store.record_reflection_pass(
    proposed=[...],
    kept=[...],
    dropped=[...],
    model_used="qwen3.5:cloud",
)

audit = store.reflection_audit(limit=10)
# returns structured dict with per-pass episodes, proposals, critic decisions

# User correction becomes a first-class memory
store.record_user_correction(
    target_entry_id="jrn_abc123",
    correction="Actually I prefer tea, not coffee.",
)
```

**Work items:**

1. **Reflection pass persistence:** store the raw reflection output, critic
  scores, and final kept/dropped lists. Because raw output can be large,
  either (a) compress it as zlib/gzip into a BLOB column, or (b) keep full
  detail for the most recent `reflection_pass_retention` passes and summarize
  older ones (drop raw proposals, keep counts and aggregate scores).
2. **Retention cap:** add `reflection_pass_retention: int = 100` config. Prune
  oldest passes on insert so the table does not grow unbounded.
3. **`reflection_audit()` API:** paginated retrieval of past passes with enough
  detail to debug quality regressions. Include per-proposal `critic_score` and
  aggregate score distribution for the pass.
4. **User correction entries:** a new journal type `user_correction` that
  supersedes the incorrect belief. Corrections are retrieved into future
  reflection prompts with high base confidence and explicit weighting.
5. **Critic calibration:** expose `critic_score` per proposed entry; let users
  tune thresholds via `MemoryConfig`.
6. **Reflection prompt injection:** include recent `user_correction` entries in
  the reflector prompt so the same mistake is not repeated.

**Files to touch:**

- `src/memlife/_schema.py` — new `reflection_passes` table
- `src/memlife/_journal.py` — correction APIs
- `src/memlife/reflection.py` — persist pass details, read corrections
- `src/memlife/config.py` — critic threshold tuning
- `tests/test_reflection.py` — audit and correction tests

**Verification:**

- A user correction supersedes the target journal entry
- A subsequent reflection pass includes the correction in context
- `reflection_audit(limit=10)` returns passes in reverse chronological order
- `reflection_audit()` returns the last N passes with kept/dropped breakdowns
- Retention pruning removes the oldest pass when the cap is exceeded
- Reflection pass history stays capped at `reflection_pass_retention`

---

## Cross-cutting concerns for 0.5.0

These apply to all three work tracks:

### Configuration validation

Add a public `MemoryConfig.validate()` method that users can call directly
and that `MemoryStore.__init__()` invokes automatically. Check at minimum:

- `namespace` matches the allowed character set.
- `vector_backend` is a known value.
- `db_path` is a writable path (not a directory).
- Embedding model/dimension compatibility when an embedder is provided.

Example standalone use:

```python
cfg = MemoryConfig(db_path="...", namespace="...", vector_backend="sqlite-vec")
cfg.validate()
store = MemoryStore(cfg)
```

This prevents half-initialized stores and surfaces misconfiguration before
any data is written.

### Observability hooks

Build on the existing recall path counters (`_recall_counters`) with a small
optional event stream:

```python
store.metrics()  # returns structured counters: recall, embedding health, GC
```

For 0.5.0 this is read-only and lightweight. Later releases can add callbacks
or async event streams for hosted users who want to monitor namespace growth,
reflection pass frequency, and backend latency.

### Documentation split

- `docs/namespaces.md` — namespace design, migration, backup/restore story
- `docs/vector-backends.md` — comparison of JSON, binary, and sqlite-vec
- `docs/reflection-audit.md` — reflection transparency and correction usage

### Deprecation timeline

- 0.5.0: `use_sqlite_vec` and `use_binary_vectors` still accepted but emit
  `DeprecationWarning`, mapping to the new `vector_backend` enum.
- 0.6.0: remove the boolean flags entirely.

---

## 0.5.0 Non-goals

These are deliberately out of scope to keep the release shippable:

- **Multi-agent identity / collaborative attestation:** still single-agent per
  namespace. Multi-agent is a research-level change, not a 0.5.0 feature.
- **Working-memory auto-injection:** stays outside the store.
- **New persona layer:** journal already shapes tone privately.
- **Graph memory (Graphiti-style):** temporal triples + belief links are the
  current graph-adjacent layer. A full graph backend is a separate project.

---

## Beyond 0.5.0

### 0.6.0: Sync and server modes

- **HTTP server wrapper** for the MCP server so multiple clients can share one
  memlife instance safely.
- **Sync subsystem** as a separate package: replicate memory state between
  devices/agents with conflict resolution.
- **Backup/restore API** with namespace-aware import/export.

### 0.7.0: Multi-agent identity

- Author attribution on every memory.
- Shared memory scopes vs private agent scopes.
- Cross-agent contradiction detection and consensus scoring.
- Identity attestation for externally contributed facts.

### 0.8.0: Advanced retrieval

- Graph traversal over temporal triples and belief links.
- Query-time re-ranking with a small cross-encoder.
- Long-context summarisation of retrieved memory bundles.

---

## Release checklist for 0.5.0

- [ ] Namespace isolation implemented and tested
- [ ] sqlite-vec promoted to first-class backend
- [ ] Reflection audit and correction APIs implemented
- [ ] All existing tests pass (118+)
- [ ] New tests cover namespaces, sqlite-vec, reflection audit
- [ ] Ruff clean on `src/` and `tests/`
- [ ] README updated with namespace and backend examples
- [ ] `docs/namespaces.md`, `docs/vector-backends.md`, and `docs/reflection-audit.md` created
- [ ] CHANGELOG.md created and populated for 0.5.0
- [ ] Version bumped to `0.5.0b0` for beta, then `0.5.0`
- [ ] PyPI upload
- [ ] GitHub release notes
- [ ] MCP server CLI updated with `--namespace` / `MEMLIFE_NAMESPACE`
- [ ] sqlite-vec optional dependency added to `pyproject.toml`

---

## Decision register

| ID | Decision | Default | Notes |
|----|----------|---------|-------|
| R1 | Namespace strategy | Separate DB files | Keeps schema and migration simple |
| R2 | Vector backend config | `VectorBackend` enum | Replaces boolean flags; string values accepted as aliases for one release |
| R3 | Multi-agent scope | Out of 0.5.0 | Single-agent-per-namespace only |
| R4 | Corrections as journal entries | Yes | Reuse existing supersession/retirement machinery |
| R5 | Sync subsystem | Separate package | Avoid bloating core memlife |
| R6 | Vector backend migration | Explicit opt-in | Auto-backfill on backend change is a footgun |
| R7 | Reflection pass retention | Cap at 100 | Prevents unbounded growth of pass history |

---

*Drafted: 2026-07-10*
*Reviewed: 2026-07-10*
*Status: ready for implementation after final steering*
