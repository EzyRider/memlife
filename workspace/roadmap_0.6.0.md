# memlife 0.6.0 Roadmap

> Consolidated from `BACKLOG.md`, `docs/ROADMAP_v2.md`, and the shipped 0.5.x
> releases. Only items that are genuinely pending, deferred, or not yet covered
> are listed here. Done items are referenced but not repeated.
>
> Last updated: 2026-07-11

## 1. Bug fixes / correctness (from MF backlog)

| ID | Item | Priority | Status |
|----|------|----------|--------|
| MF-005 | Embedding-version backfill for contradiction embeddings on model swap | Low | Not done |
| MF-012 | SQL injection in `import_jsonl` via unvalidated column names | High | Not done |
| MF-014 | `DummyEmbedder` hash vectors produce misleading cosine similarity | Medium | Not done |

Notes:
- MF-001..MF-004, MF-006..MF-011, MF-013, MF-015, MF-016 are shipped in 0.5.x.
- MF-012 is a real security hole for anyone restoring from JSONL backups.
- MF-014 makes the zero-LLM / quickstart path unreliable; worth fixing before
  promoting "no-LLM mode" more loudly.
- MF-005 is low because contradiction embeddings are currently only created
  when an explicit contradiction record exists; backfill is a model-change
  edge case.

## 2. Cleanup / hygiene (from MF-016 latent list)

| Item | Notes |
|------|-------|
| `SyncMemoryStore._run()` swallows "already running" `RuntimeError` then calls `asyncio.run()` from inside a running loop | Concurrency bug; can crash the sync wrapper |
| `OpenAIChat.chat()` does not handle empty `response.choices` | Robustness |
| `sentence_transformers` adapter uses deprecated `asyncio.get_event_loop()` | Deprecation cleanup |
| `Reflector` duplicates decay/config parameters instead of accepting `MemoryConfig` | API hygiene |
| `source_episodes_json` column name overload for contradictions | Rename or document |

## 3. Features / architecture (next thesis extensions)

### 3.1 Entity graphing (planned next major feature)
- Extract entities from facts/episodes/journal.
- Store entity-entity relations (e.g. person-to-person, person-to-project).
- Track entity-fact provenance: which fact came from which episode/source.
- Target interface: plug-and-play Python API + MCP exposure for non-Python consumers.
- Exact interface TBD.

### 3.2 Embedding cache / model versioning per vector
- Content-addressable cache keyed on `(model_name, sha256(text)) -> vector`.
- Makes model swaps cheap and repeated text instant.
- Was deferred in 0.5.0 as a nice-to-have; now promoted because entity
  graphing and reflection both benefit from fast, deterministic embeddings.

### 3.3 Incremental conflict detection
- Stop O(n²) pairwise contradiction scans.
- Use vector-neighborhood or indexing to reduce scan cost as fact count grows.
- **Gate:** profile current reflection cost on a representative fact set
  before committing to the implementation. Entity graphing may provide
  clustering shortcuts.

### 3.4 Widen reflection window
- Allow reflection to look further back than the current narrow window.
- Trade-off: cost vs. consolidation quality.

### 3.5 Index tool calls / outcomes
- Better querying and recall of tool-call episodes by tool name, success/failure,
  and outcome metadata.

### 3.6 Polyphonic recall & MEMORIA extraction
- Already partially present as opt-in flags in 0.5.x.
- Decide whether to promote to default-on, harden, or leave opt-in.

## 4. Operational / scale (deferred until scale demands it)

| Item | Notes |
|------|-------|
| Brute-force vector search replacement | Only when row counts or latency demand it |
| Journal consolidation false-merge mitigation | Currently latent; revisit with more data |
| SQLite backup rotation via cron | Hermes priority list item #4 |
| Retention / GC tuning beyond current `run_gc()` + `run_vacuum()` | Hermes priority list item #5 |
| Long-context summarisation of retrieved memory bundles | 0.8.0 roadmap candidate |

## 5. Docs / packaging

| Item | Notes |
|------|-------|
| `docs/vector-backends.md` | Comparison of JSON / binary / sqlite-vec |
| `docs/namespaces.md` | Namespace design, migration, backup/restore |
| `docs/reflection-audit.md` | Reflection transparency and correction usage |
| README no-LLM example fix | See 0.5.5 release notes |
| Add `CHANGELOG.md` entries for 0.5.x if not already current | Verify against git tags |

## 6. Release planning

### 0.5.5 (patch, soon)
Security / correctness / concurrency fixes that should not wait for 0.6.0.

1. **MF-012** — `import_jsonl` SQL injection fix (whitelist columns per table).
2. **MF-014** — `DummyEmbedder` bag-of-words correctness fix.
3. **`SyncMemoryStore._run()` concurrency fix** — stop swallowing the
   "already running" `RuntimeError` and calling `asyncio.run()` inside a loop.
4. **README no-LLM example fix** — make the "No-LLM Mode" snippet valid Python.

### 0.6.0 (minor)
Focused on entity graphing and the storage layer it depends on.

1. **Entity graphing MVP**
2. **Embedding cache / model-versioned vectors** — do this first in the 0.6.0
   cycle so graphing and reflection build on it.
3. **Incremental conflict detection** — only if profiling during 0.6.0 shows
   O(n²) scans are hurting.
4. **`Reflector` config deduplication cleanup**
5. **Docs split** — `docs/vector-backends.md`, `docs/namespaces.md`,
   `docs/reflection-audit.md`.

### Beyond 0.6.0
- Multi-agent identity / author attribution (was 0.6.0 in ROADMAP_v2; now
  considered a larger research-level change).
- Sync subsystem as a separate package (was 0.7.0).
- HTTP server wrapper for MCP.
- Advanced retrieval: graph traversal, cross-encoder reranking, long-context
  summarisation.

## 7. Decision register (carried forward)

| ID | Decision | Default | Notes |
|----|----------|---------|-------|
| R1 | Namespace strategy | Separate DB files | Shipped in 0.5.x |
| R2 | Vector backend config | `vector_backend` enum/string | Shipped in 0.5.x |
| R2a | Vector backend migration | Explicit opt-in | Shipped in 0.5.x |
| R3 | Multi-agent scope | Out of 0.6.0 | Moved to post-0.6.0 |
| R4 | Corrections as journal entries | Yes | Shipped in 0.5.x |
| R5 | Sync subsystem | Separate package, later | Unchanged |
| R6 | Reflection pass retention | Count + days | Shipped in 0.5.x |
| R7 | Embedder sharing across namespaces | Opt-in only | Shipped in 0.5.x |
| R8 | Reflection pass API | Internal record, public audit | Shipped in 0.5.x |
| R9 | Next major feature | Entity graphing | Decided 2026-07-11 |
| R10 | Incremental conflict detection | Defer unless profiling justifies | Added 2026-07-11 |
| R11 | Embedding cache sequencing | First 0.6.0 item | Added 2026-07-11 |

---

*Source files: `BACKLOG.md`, `docs/ROADMAP.md`, `docs/ROADMAP_v2.md`,
`CHANGELOG.md`, `workspace/vector_backend_roadmap_0.5.0.md`.*
