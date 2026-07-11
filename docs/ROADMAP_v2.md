# memlife Roadmap (Revised)

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

**Namespace path mapping rule (R1a):**

Namespace determines the SQLite file path deterministically. The default
namespace `_default` is reserved and preserves pre-0.5.0 behaviour exactly.

```python
def db_path_for(db_path: str | Path, namespace: str) -> Path:
    p = Path(db_path)
    if namespace == "_default":
        return p
    if p.is_dir() or str(db_path).endswith(("/", "\\")):
        return p / f"memlife_{namespace}.db"
    return p.with_name(f"{p.stem}_{namespace}{p.suffix}")
```

Examples:

| `db_path` | `namespace` | Resolved file |
|---|---|---|
| `./memlife.db` | `_default` | `./memlife.db` |
| `./memlife.db` | `julie` | `./memlife_julie.db` |
| `./data/` | `julie` | `./data/memlife_julie.db` |
| `/var/memlife/prod.db` | `team-a` | `/var/memlife/prod_team-a.db` |

This keeps the change surgical and preserves the single-file-per-agent mental
model. A shared-schema strategy (column-based namespaces) can be explored later
if hosted demand justifies it.

**API surface:**

- `MemoryConfig.namespace: str = "_default"`
- `MemoryConfig.db_path_for(namespace: str) -> Path` helper
- `MemoryStore.list_namespaces(root: Path | str | None = None)` — scan
  configured directory or provided root for `memlife_*.db`
- `MemoryStore.clone_to_namespace(name: str, *, copy_embeddings: bool = True)` —
  copy current DB content into a new namespaced DB (part of 0.5.0, not a
  follow-up)
- `MemoryStore.switch_namespace(name: str, *, embedder: Embedder | None = None)` —
  returns a new store pointing at the namespaced DB

**Namespace validation and safety:**

- Valid namespace regex: `^[a-zA-Z0-9_-]{1,64}$`. Raise `NamespaceError` for
  invalid names before touching the filesystem.
- Reject reserved names except `_default`.
- `_default` always maps to the original DB path (migration-safe).
- Embedder sharing is **opt-in**, not default. `switch_namespace()` creates a
  fresh embedder from `config` unless the caller explicitly passes one. This
  prevents cross-namespace model/dimension mismatches.
- If an embedder caches per-store state, invalidate or isolate the cache when
  switching.
- Namespaces are isolation boundaries, not security boundaries. Separate DB
  files prevent accidental leakage; they do not encrypt or authenticate.

**Files to touch:**

- `src/memlife/config.py` — add namespace field and path helper
- `src/memlife/store.py` — resolve DB path in `__init__`
- `src/memlife/sync_store.py` — pass namespace through
- `src/memlife/mcp_server.py` — accept `--namespace` / `MEMLIFE_NAMESPACE`
- `tests/test_namespaces.py` — new test module

**Verification:**

- Two namespaces sharing a directory do not see each other's episodes/facts/journal
- Switching namespaces creates/uses the correct DB file
- Default namespace preserves existing DB path (migration-safe)
- Invalid namespace values raise a clear error before touching the filesystem
- Embedder sharing does not leak vectors between namespaces with different models
- `clone_to_namespace()` copies all tables and produces an independent DB

---

### 2. sqlite-vec as a first-class vector backend

**Goal:** Move sqlite-vec from a silent fallback to a real indexed vector backend
that users can choose and trust.

**Current state:** `MemoryConfig.use_sqlite_vec` exists, but the path is
fallback-only and not well tested in CI (sqlite-vec is an optional dependency).

**Design:**

```python
from memlife.config import VectorBackend

cfg = MemoryConfig(
    db_path="./memlife.db",
    vector_backend=VectorBackend.SQLITE_VEC,   # JSON | SQLITE_VEC | BINARY
)
```

Deprecate `use_sqlite_vec: bool` and `use_binary_vectors: bool` in favor of a
single `vector_backend` enum. Keep old flags as aliases for one release.

**Work items:**

1. **Backend enum:** use `VectorBackend` enum (`json`, `sqlite_vec`, `binary`),
  not a plain `str`, for validation and IDE support.
2. **Auto-detection:** try to load `sqlite_vec` at runtime; fall back to JSON
  with a logged warning if unavailable and backend was not explicitly set.
3. **Schema management:** create virtual tables per embedding kind
  (`vec_episodes`, `vec_facts`, `vec_journal`) on first connection.
4. **Dimension guards:** store expected dimension per table; reject mismatched
  vectors with a clear error instead of silent corruption. Provide a way to
  drop/recreate virtual tables and re-backfill when the embedding model changes.
5. **Explicit migration:** switching an existing DB to sqlite-vec does **not**
  auto-backfill. `backfill_embeddings()` is the user-facing opt-in; it logs
  progress and can be resumed.
6. **Migration status API:** add `store.migration_status()` that reports:
  - active backend
  - rows with JSON embeddings
  - rows in sqlite-vec virtual tables
  - rows with mismatched embedding_model
  - recommended next action
  If `vector_backend=SQLITE_VEC` but sqlite-vec tables are empty while
  JSON embeddings exist, emit a clear warning at init:
  ```
  WARNING: vector_backend is sqlite-vec but no sqlite-vec vectors found.
  Call store.backfill_embeddings() to migrate, or set vector_backend="json"
  to use existing vectors.
  ```
7. **Backfill integration:** `backfill_embeddings()` populates sqlite-vec
  tables when active.
8. **CI:** add a parameterized test fixture that runs the vector recall suite
  against all three backends, plus a job that installs `memlife[sqlite-vec]`.
9. **Benchmarks:** add a small script comparing recall latency and accuracy
  across JSON, binary, and sqlite-vec backends on synthetic data. Target:
  p95 recall latency under 50 ms for 100k rows with sqlite-vec.

**Files to touch:**

- `src/memlife/vec_backend.py` — expand to a proper backend manager
- `src/memlife/_embeddings.py` — route serialization through chosen backend
- `src/memlife/_schema.py` — create/drop virtual tables
- `src/memlife/config.py` — add `vector_backend`, deprecate booleans
- `tests/test_sqlite_vec.py` — expand coverage
- `docs/vector-backends.md` — new comparison doc

**Verification:**

- All vector recall tests pass with each backend
- Backfill produces correct vectors in the active backend
- Dimension mismatch raises a clear error
- CI runs sqlite-vec job
- `migration_status()` reports accurate actionable state

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
# Public API
audit = store.reflection_audit(limit=10)
# returns structured dict with per-pass episodes, proposals, critic decisions

store.record_user_correction(
    target_entry_id="jrn_abc123",
    correction="Actually I prefer tea, not coffee.",
)

# Optional debugging API
last_pass = store.last_reflection_pass()
```

Internal reflection pass persistence uses a dataclass:

```python
@dataclass
class ReflectionPass:
    id: str
    created_at: float
    episode_ids: list[str]
    proposed: list[dict]
    kept: list[dict]
    dropped: list[dict]
    model_used: str
    critic_model_used: str | None
    total_timeout: float
    elapsed_seconds: float
```

`Reflector` calls `store._record_reflection_pass(pass: ReflectionPass)` after
each run. Third-party reflection systems can use `store.import_reflection_pass()`
if they want to plug in.

**Work items:**

1. **Reflection pass persistence:** store the raw reflection output, critic
  scores, and final kept/dropped lists. Currently only aggregate metrics are
  stored.
2. **Retention cap:** add `reflection_pass_retention_count: int = 100` and
  `reflection_pass_retention_days: int = 90`. Prune oldest passes when either
  limit is exceeded.
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
- `reflection_audit()` returns the last N passes with kept/dropped breakdowns
- Reflection pass history stays capped by count and days
- `last_reflection_pass()` returns the most recent pass or `None`

---

## Cross-cutting concerns for 0.5.0

These apply to all three work tracks:

### Configuration validation

Add a `MemoryConfig.validate()` method that runs at `MemoryStore` init and
fails fast with clear messages. Check at minimum:

- `namespace` matches the allowed character set and length.
- `vector_backend` is a known `VectorBackend` value.
- `db_path` is a writable path (not a directory when namespace is `_default`).
- Embedding model/dimension compatibility when an embedder is provided.

This prevents half-initialized stores and surfaces misconfiguration before
any data is written.

### Observability hooks

Build on the existing recall path counters (`_recall_counters`) with a small
read-only metrics API:

```python
@dataclass
class Metrics:
    recalls_total: int
    recalls_vector: int
    recalls_keyword: int
    recalls_hybrid: int
    embeddings_total: int
    embeddings_stale: int
    embeddings_backfilled: int
    gc_last_run: float | None
    gc_pruned: dict[str, int]
    reflection_passes_total: int
    reflection_corrections_total: int
    namespace: str
    vector_backend: str

store.metrics()  # returns Metrics
```

For 0.5.0 this is read-only and lightweight. Later releases can add callbacks
or async event streams for hosted users who want to monitor namespace growth,
reflection pass frequency, and backend latency. Version the schema within the
`Metrics` dataclass and commit to backward compatibility within major versions.

### Documentation split

- `docs/namespaces.md` — namespace design, migration, backup/restore story
- `docs/vector-backends.md` — comparison of JSON, binary, and sqlite-vec
- `docs/reflection-audit.md` — reflection transparency and correction usage

### Deprecation timeline

- 0.5.0: `use_sqlite_vec` and `use_binary_vectors` still accepted but emit
  `DeprecationWarning`, mapping to the new `VectorBackend` enum.
- 0.6.0: remove the boolean flags entirely.

### Security and validation

- Namespace regex: `^[a-zA-Z0-9_-]{1,64}$`. Reject `..`, `/`, `\`, control
  characters, whitespace, and leading/trailing punctuation.
- Normalize whitespace before validation.
- Ensure the directory for `db_path` is writable before creating the DB.
- Document that namespaces are isolation boundaries, not encryption or
  authentication boundaries.
- If the MCP server exposes namespace selection, validate the value server-side.

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

### 0.6.0: Multi-agent identity

- Author attribution on every memory.
- Shared memory scopes vs private agent scopes.
- Cross-agent contradiction detection and consensus scoring.
- Identity attestation for externally contributed facts.

*Rationale:* Multi-agent identity should come before cross-device sync. If
multiple agents will contribute to the same replicated memory, you need
author attribution and conflict resolution rules before the sync subsystem
can merge their changes meaningfully.

### 0.7.0: Sync and server modes

- **HTTP server wrapper** for the MCP server so multiple clients can share one
  memlife instance safely.
- **Sync subsystem** as a separate package: replicate memory state between
  devices/agents with conflict resolution.
- **Backup/restore API** with namespace-aware import/export.

### 0.8.0: Advanced retrieval

- Graph traversal over temporal triples and belief links.
- Query-time re-ranking with a small cross-encoder.
- Long-context summarisation of retrieved memory bundles.

---

## Release checklist for 0.5.0

- [ ] Namespace isolation implemented and tested
- [ ] `clone_to_namespace()` implemented and tested
- [ ] MCP server accepts `--namespace` / `MEMLIFE_NAMESPACE`
- [ ] sqlite-vec promoted to first-class backend
- [ ] `migration_status()` implemented for vector backend switches
- [ ] Reflection audit and correction APIs implemented
- [ ] `Metrics` dataclass and `store.metrics()` implemented
- [ ] All existing tests pass (118+)
- [ ] New tests cover namespaces, sqlite-vec, reflection audit
- [ ] Ruff clean on `src/` and `tests/`
- [ ] README updated with namespace and backend examples
- [ ] `docs/namespaces.md` and `docs/vector-backends.md` created
- [ ] CHANGELOG.md created and populated for 0.5.0
- [ ] Version bumped to `0.5.0b0` for beta, then `0.5.0`
- [ ] PyPI upload
- [ ] GitHub release notes
- [ ] MCP server examples/docs updated if CLI args change

---

## Decision register

| ID | Decision | Default | Notes |
|----|----------|---------|-------|
| R1 | Namespace strategy | Separate DB files | Keeps schema and migration simple |
| R1a | Namespace path mapping | Stem suffix for files, `memlife_<ns>.db` for directories | `_default` preserves exact `db_path` |
| R2 | Vector backend config | `vector_backend: VectorBackend` enum | Replaces `use_sqlite_vec`/`use_binary_vectors` booleans |
| R2a | Vector backend migration | Explicit opt-in | Auto-backfill on backend change is a footgun |
| R3 | Multi-agent scope | Out of 0.5.0 | Single-agent-per-namespace only |
| R4 | Corrections as journal entries | Yes | Reuse existing supersession/retirement machinery |
| R5 | Sync subsystem | Separate package, moved to 0.7.0 | Avoid bloating core memlife; needs identity first |
| R6 | Reflection pass retention | Count + days | `100` passes or `90` days, whichever is stricter |
| R7 | Embedder sharing across namespaces | Opt-in only | Default is fresh embedder per namespace |
| R8 | Reflection pass API | Internal `_record_reflection_pass`, public `reflection_audit` | Loose-dict public API avoided |

---

*Drafted: 2026-07-10*
*Revised: 2026-07-10*
*Status: ready for steering review*
