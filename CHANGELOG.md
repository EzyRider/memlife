# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.6.11] - 2026-07-24

### Fixed

- GitHub Actions test matrix now uses Python 3.11 and 3.12 only, removing the
  obsolete Python 3.10 job that failed after the minimum version was raised.

## [0.6.10] - 2026-07-24

### Changed

- **Minimum Python version raised to 3.11.** memlife no longer supports Python
  3.10. The `typing_extensions` fallback for `typing.Self` and the conditional
  dependency added in 0.6.8 have been removed.

## [0.6.9] - 2026-07-24

### Changed

- (Superseded by 0.6.10; the Python 3.11 minimum is now documented in 0.6.10.)

## [0.6.8] - 2026-07-24

### Fixed

- `_LockedConn.cursor()` now returns a context-managed `_LockedCursor` proxy
  that acquires the store lock on entry, closes the real cursor on exit, and
  releases the lock even if the iteration body raises. This prevents cursors
  from outliving the lock that serialises access to the underlying SQLite
  connection.
- `_LockedConn` now wraps `row_factory`, `isolation_level`, and `text_factory`
  getters/setters under the same reentrant lock used for `execute`/`commit`,
  removing a latent race when connection-level attributes are mutated from
  multiple threads.
- `MemoryConfig.validate()` and `MemoryStore._set_pragma()` now validate
  PRAGMA names and values against explicit allowlists before any f-string
  interpolation, closing an SQL-injection vector via `sqlite_journal_mode` or
  similar config fields.

### Changed

- Audited cursor-iteration sites in `_gc.py`, `_triples.py`, and `_schema.py`
  to use explicit `with self.conn.cursor()` cursors or `.fetchall()`, ensuring
  no cursor is iterated while the lock is released.
- Added `tests/test_locked_conn.py` covering `_LockedCursor` context-manager
  behaviour and locked `row_factory` access under contention.
- Added `tests/test_config.py` coverage for PRAGMA name/value validation.

## [0.6.7] - 2026-07-18

### Fixed

- Graph-integrated retrieval now matches lowercase queries against stored
  entities and aliases. Previously `retrieve("james")` would not follow graph
  links for an entity stored as "James" because the query parser relied solely
  on the capitalised-proper-noun extractor. Retrieval now scans the query
  against known entity names and aliases case-insensitively in addition to
  running the generic extractor.

## [0.6.6] - 2026-07-18

### Fixed

- Triple query paths (`triples_about`, `triples_from`, `triples_to`,
  `current_truth`, `truth_as_of`, `entity_neighbors`) now resolve entities
  case-insensitively, matching the insertion path. Querying "james" now
  returns triples stored under "James" or via an alias.

## [0.6.5] - 2026-07-18

### Fixed

- Infinite-loop bug in `_prune_unreferenced_embedding_cache`: when the first
  batch of cache rows were all still referenced, the old `LIMIT`-only scan
  would fetch the same rows forever. The scan now uses keyset pagination by
  `cache_key` so deletion never causes rows to be skipped or re-scanned.
- `_cache_lookup` and `_cache_store` now use batched SQL statements, reducing
  round-trips from O(N) per text to 1–2 per batch.
- `extract_and_link_entities` now persists mention triples and entity aliases
  in a single transaction instead of committing once per extracted entity.
- `memory_gc` MCP tool output now includes `mention_triples_for_deleted_sources`,
  `episode_tools`, `embedding_cache_unreferenced`, and
  `embedding_cache_evicted_lru` counts.

### Changed

- `_prune_unreferenced_embedding_cache` streams referenced rows from the cursor
  rather than `fetchall()` so memory usage stays flat for large databases.
- `_cache_store` now skips non-numeric embedding vectors instead of writing
  them to the cache.
- MCP tool-call dedup eviction is now guarded by a lock to remove a latent
  race when multiple tool threads log calls concurrently.
- `memlife.__version__` and `pyproject.toml` version bumped to 0.6.5.

## [0.6.4] - 2026-07-18

### Changed

- Synchronised `main` branch with the 0.6.2 and 0.6.3 release tags so GitHub
  README and source tree match PyPI.
- `memlife.__version__` and `pyproject.toml` version bumped to 0.6.4.

## [0.6.3] - 2026-07-18

### Changed

- README now reports the current version and includes a "What's New" section.
- `memlife.__version__` and `pyproject.toml` version bumped to 0.6.3.

## [0.6.2] - 2026-07-18

### Fixed

- Graph relationship traversal now follows **incoming** edges as well as
  outgoing edges, so querying an entity that appears as the object of a
  relationship (e.g. "Bob" in "Alice knows Bob") discovers related sources.
- Entity canonicalisation is now case-insensitive when creating or ensuring
  entities. Manual `store_triple("James", ...)` reuses an auto-extracted
  canonical entity "james" instead of creating a duplicate "James" node.
- `SyncMemoryStore.retrieve()` and `MemoryStore.retrieve()` now accept a
  `debug=True` flag and return the structured debug dict.
- `SyncMemoryStore.store_mention_triple()` added for parity with the async
  `MemoryStore` API.

### Added

- Regression test suite for graph-integrated retrieval
  (`tests/test_graph_retrieval.py`) covering mention-triple boosts,
  outgoing/incoming relationship hops, superseded-fact filtering,
  closed-relationship filtering, case canonicalisation, and debug output.

## [0.6.1] - 2026-07-18

### Fixed

- `_schema._migrate()` now re-reads `journal` columns before adding
  `annotations_json` / `links_json`, making migration idempotent on
  partially-migrated databases.
- `FactStore.check_conflicts()` keyword fallback now respects
  `fact_conflict_threshold` instead of hardcoding 0.5.
- Graph-integrated retrieval now scales `graph_signal` by the confidence of
  the strongest currently-valid linking triple, rather than giving every
  linked candidate a flat 1.0 boost.
- Graph retrieval no longer surfaces superseded facts, retired/superseded
  journal entries, or contradiction rows.
- `retrieve()` now skips vector-recalled episodes that were already added
  via graph expansion, avoiding duplicate episode candidates.
- Graph expansion failures are caught and logged instead of crashing the
  whole retrieval call.
- Polyphonic recall now runs RRF only on candidates that passed the score
  cutoff, so cutoff configuration remains meaningful.
- Debug output now includes the actual triples that produced a graph link
  in `graph_triples`.

## [0.6.0] - 2026-07-18

### Added

- **Embedding cache** — content-addressable cache keyed on
  `(model_name, sha256(text))` storing canonical JSON float vectors.
  Cache read/write is transparently wrapped into `embed_texts()` so only
  misses hit the embedder. Controlled by `MemoryConfig.embedding_cache_enabled`
  and `embedding_cache_max_mb` with env overrides
  `MEMLIFE_EMBEDDING_CACHE_ENABLED` / `MEMLIFE_EMBEDDING_CACHE_MAX_MB`.
  GC enforces an LRU size cap and removes unreferenced rows.
- **Automatic entity extraction** — deterministic, zero-LLM extraction of
  proper-noun-like phrases, allowlist terms, and short acronyms from facts,
  episodes, and journal entries. Opt-in via
  `MemoryConfig.auto_entity_extraction`; mention triples are created when
  `auto_entity_mentions` is true. Tuned with `auto_entity_confidence`,
  `entity_extraction_allowlist`, and `entity_extraction_blocklist`.
- **Graph-integrated retrieval** — `retrieve()` can now boost candidates that
  are linked to entities mentioned in the query. Enabled with
  `MemoryConfig.use_graph_retrieval`; weight controlled by
  `graph_retrieval_weight`. The graph signal follows `mentions` triples and
  one-hop relationship triples, scaled by triple confidence and source
  recency. Debug output exposes `graph_signal` and the expanding triples.

### Changed

- `retrieve()` candidates now include `graph_signal`, `triples`, and a
  `why` explanation when debug mode is enabled.
- `_facts.py`, `_journal.py`, and `_triples.py` updated to record entity
  provenance and expose relationship-based source lookup.

## [0.5.5] - 2026-07-17

### Fixed

- `memlife.__version__` now matches `pyproject.toml` (0.5.5); previously it was
  still reporting 0.5.3 after the 0.5.4 release.
- README MCP server tool list now lists only implemented tools and clarifies
  that `memlife://contradictions` is a resource, not a tool. The `MemoryConfig`
  example snippet now uses real fields (`reflection_timeout`,
  `journal_decay_halflife_days`, `journal_decay_floor`).
- `shutdown_mcp_server()` is now idempotent: a sentinel prevents SIGTERM and
  `atexit` from both attempting cleanup and closing resources twice.
- `list_namespaces()` normalizes directory names to lowercase and warns/ignores
  mixed-case duplicates, preventing case-insensitive filesystems (Windows,
  macOS) from presenting two directories as separate namespaces when they share
  one database file.

### Added

- Advisory warning when `data_dir` resolves under a known cloud-sync folder
  (OneDrive, Dropbox, Google Drive, iCloud, Box, Nextcloud, ownCloud,
  Syncthing). SQLite WAL sidecar files are constantly rewritten and can be
  locked or corrupted by sync clients and real-time antivirus scanners.
- The in-memory tool-call dedup cache (`--log-tool-calls`) is capped at 1000
  entries so long-running MCP servers cannot grow it without bound.

### Changed

- `import time` in `memlife.mcp_server` moved to module level (code hygiene).

## [0.5.4] - 2026-07-11

### Added

- `memlife-mcp-server` now exposes `--reflection-timeout` and
  `--reflection-total-timeout` CLI flags, forwarded to `MemoryConfig`.
  This prevents `memory_reflect` from being killed by short MCP client
  timeouts during larger reflection batches.
- `memlife-mcp-server` now exposes `--memorias-extraction` and
  `--polyphonic-recall` CLI flags and forwards them to `MemoryConfig`.
- `memory_vacuum` and `memory_reflect` are now listed in the README tools
  table.

### Changed

- `memory_search_episodes` tool description now clarifies that
  `tool_name`-based search only finds episodes that were created with an
  explicit `tool_name` (e.g. via `remember(..., tool_name=...)`). This is
  an integration concern, not a search bug.
- `memlife-mcp-server --vector-backend` now defaults to `json` explicitly
  and no longer shows an empty choice in `--help` output. Behaviour is
  unchanged because `MemoryConfig` already resolved to `json` by default.

## [0.5.3] - 2026-07-11

### Fixed

- `MemoryConfig.validate()` no longer returns early when `vector_backend` is
  `None`. The default auto-resolution (None) was introduced in 0.5.2, but the
  existing `if backend is None: return` guard caused most validation checks to
  be skipped for any config that did not explicitly set a backend.
- `BinaryVectorBackend.deserialize()` and `_unpack()` now fall back to
  JSON-decoding stored vectors. Switching an existing store from the default
  JSON backend to the binary backend no longer makes previously stored vectors
  invisible to search.

### Added

- Regression tests for `validate()` running all checks with `vector_backend=None`.
- Regression test for binary backend reading vectors stored as JSON by the
  default backend.

## [0.5.2] - 2026-07-11

### Fixed

- Vector backend `search()` is now used by fact, episode, and journal recall.
  Previously only `sqlite_vec` was special-cased; `json` and `binary` backends
  did their own inline cosine/Hamming computation, so selecting
  `vector_backend="binary"` did not actually use Hamming distance search.
- `JsonVectorBackend.search()` now applies the contradiction filter correctly.
- `_recall_facts_sqlite_vec()` no longer reaches through `conn._raw` directly;
  it uses the public `self.vector_backend.search()` API.
- Legacy `use_binary_vectors=True` is promoted to `vector_backend="binary"`
  when no explicit backend is set.
- `MemoryConfig.resolved_vector_backend()` centralises backend precedence:
  explicit `vector_backend` > `use_binary_vectors` > `use_sqlite_vec` > `json`
  default. When both legacy flags are True a warning is logged and binary wins.
- MCP server shutdown now closes the store, embedder, chat adapter, and
  reflector, and SIGTERM exits the process after cleanup instead of hanging.
- Ollama embedder/chat adapters now create their `aiohttp.ClientSession` lazily
  on the first async call, avoiding `RuntimeError` when instantiated outside a
  running event loop.

### Added

- `tests/test_vector_backends.py` now covers `BinaryVectorBackend` store, delete,
  serialize, deserialize, search, and end-to-end binary recall.
- `tests/test_config.py` covers vector backend precedence, normalisation, and
  ambiguity warnings.

## [0.5.1] - 2026-07-11

### Fixed

- `memlife-mcp-server` now accepts `--vector-backend {json,sqlite_vec,binary}`
  and falls back to `MEMLIFE_VECTOR_BACKEND`. Previously backend selection
  required the environment variable; the CLI wrapper had no equivalent flag.

## [0.5.0] - 2026-07-11

### Added

- `MemoryStore.metrics()` returns a public `Metrics` snapshot with counts,
  embedding coverage, reflection aggregates, recall counters, DB metadata,
  and schema migration health. Exposed on `SyncMemoryStore` and rendered by
  the `memlife://stats` MCP resource.
- `MemoryStore.migration_status()` reports schema health: expected vs present
  tables and columns, SQLite version, page stats, and a `healthy` boolean.
  Exposed on `SyncMemoryStore` and included in `Metrics`.
- `BinaryVectorBackend` — a dedicated pluggable vector backend that stores
  embeddings as bit-packed binary vectors and searches with Hamming distance.
  Select with `MemoryConfig(vector_backend="binary")` or
  `MEMLIFE_VECTOR_BACKEND=binary`.
- `Metrics` dataclass exported from `memlife`.

### Changed

- `MemoryStore` now prefers `pysqlite3` over the stdlib `sqlite3` module when
  `pysqlite3` is installed and supports SQLite extension loading. This makes
  the `sqlite_vec` vector backend usable on interpreters whose stdlib SQLite is
  compiled without `ENABLE_LOAD_EXTENSION` (e.g. manylinux wheels).
- `sqlite-vec` optional dependency now includes `pysqlite3-binary` on Linux so
  the fallback driver is installed automatically.
- `vec_backend` module transparently falls back to a `pysqlite3` connection
  when the caller passes a stdlib connection that cannot load extensions.
- `memlife://stats` resource now uses `store.metrics()` and returns structured
  counts, embeddings, reflection, recall, and migration sections.
- Namespace default changed to `_default`; namespace validation and vector
  backend validation now run before any SQLite file is opened.
- Reflection passes are persisted with proposed/kept/dropped items, model
  metadata, and timing for audit/debugging.

### Fixed

- `recall_facts()` now consults `vector_backend.name` instead of the legacy
  `config.use_sqlite_vec` flag, so explicit backend selection works correctly.
- `gap_marker_threshold_hours` query in `_episodes.py` now uses an explicit
  `ORDER BY created_at DESC LIMIT 1` instead of relying on SQLite's bare-column
  aggregate behavior, which is unsupported by some SQLite builds (including
  pysqlite3).

## [0.4.6] - 2026-07-11

### Added

- Pluggable vector backend abstraction (`memlife.vector_backends`) with
  `VectorBackend` ABC and JSON/sqlite-vec implementations.
- `MemoryConfig.vector_backend` configuration (env `MEMLIFE_VECTOR_BACKEND`);
  deprecated `use_sqlite_vec` in favour of explicit backend selection.
- `MemoryStore` recall and embedding paths now route through the configured
  vector backend.
- `tests/test_vector_backends.py` covering backend selection and contract.

### Security

- `validate_namespace()` now normalizes namespaces to lowercase. This prevents
  `Julie` and `julie` from mapping to different directories on case-sensitive
  filesystems, matching the behaviour on macOS/Windows and the original roadmap
  design.

### Fixed

- Renamed `VectorBackend.store` property to `memory_store` to avoid collision
  with the abstract `store()` method.
- Removed unused TYPE_CHECKING imports in `_embeddings.py`.

### Added

- Regression tests for namespace case normalization and
  `switch_namespace()` embedding model preservation.

## [0.4.5] - 2026-07-11

### Added

- Pluggable vector backend abstraction (`memlife.vector_backends`).
- `VectorBackend` base class with JSON and `sqlite-vec` implementations.
- `MemoryConfig.vector_backend` option (env var `MEMLIFE_VECTOR_BACKEND`) to
  select `json` (default, no extra dependencies) or `sqlite_vec`.
- `MemoryConfig.use_sqlite_vec` is now deprecated and maps to
  `vector_backend == "sqlite_vec"`.

### Changed

- `FactStore`, `EpisodeStore`, and `JournalStore` now delegate vector storage
  to the configured backend instead of calling `vec_backend` module functions
  directly.
- `EmbeddingMixin` uses the backend for vector serialization, search, and
  distance scoring, making binary-vector and sqlite-vec paths consistent.

### Fixed

- `_supersede_fact` savepoint/transaction interaction cleaned up; no longer
  releases the savepoint before the update completes.

## [0.4.4] - 2026-07-11

### Security

- Validate `MemoryConfig.namespace` against `^[a-zA-Z0-9_-]+$` and reject path
  separators, `..`, control characters, and empty names. Raises `NamespaceError`.
  Previously a crafted namespace could escape `data_dir` and access arbitrary
  files.

### Added

- `memlife.NamespaceError`, `memlife.validate_namespace()`, and
  `memlife.list_namespaces()`.
- `MemoryStore.list_namespaces()` class method and `MemoryStore.switch_namespace()`
  instance method for enumerating and switching between namespace directories.

## [0.4.3] - 2026-07-11

### Added

- Entity graph layer (MV2 entity-graph): `entities`, `entity_aliases`, and
  `triple_provenance` tables with schema migrations for existing databases.
- `TripleMixin` expanded with normalized entity storage, alias resolution,
  provenance tracking, and BFS neighbor traversal.
- New store methods: `store_triple`, `store_fact_triple`, `resolve_entity`,
  `add_entity_alias`, `triples_about`, `triples_from`, `triples_to`,
  `triples_for_fact`, `current_truth`, `truth_as_of`, and `entity_neighbors`.
- MCP tools: `memory_store_triple`, `memory_search_triples`,
  `memory_entity_neighbors`.
- MEMORIA structured extraction wired into `Reflector.reflect()` when
  `config.memorias_extraction` is enabled; extracted KG triples carry
  provenance back to the grounding episodes.
- `SyncMemoryStore` passthroughs for all new triple/entity methods.

### Changed

- `effective_triple_confidence()` applies the same age-based exponential decay
  to triple confidence as `Fact.effective_confidence()`.
- `_veracity_for_fact()` now blends the fact's confidence with age-decayed
  triple confidence instead of raw static triple confidence.
- `run_gc()` now prunes closed `temporal_triples` older than
  `gc_closed_triples_days` (default 90), then cleans up orphaned
  `triple_provenance`, `entity_aliases`, and `entities` rows. This closes
  the accumulation gap in the graph layer.
- `MemoryConfig` gained `gc_closed_triples_days` with env var
  `MEMLIFE_GC_CLOSED_TRIPLES_DAYS`.

### Fixed

- Graph traversal results now include `created_at` so consumers can compute
  decay anchors consistently.

## [0.4.2] - 2026-07-11

### Fixed

- `Fact.retired` property added so `resolve_fact()` no longer raises
  `AttributeError` when walking superseded fact chains. This was a
  ship-blocking crash for reflection loops that had detected and then revised
  contradictions.
- `run_gc()` now reports the correct `episode_tools` prune count instead of
  reusing the stale `episodes` cursor.
- `backfill_embeddings()` now serializes vectors through `_serialize_vec()`
  so binary-vector mode is honored for backfilled rows as well as freshly
  stored rows.
- JSONL export/import round-trip now preserves `facts.annotations_json` and
  journal `last_detected`, `annotations_json`, and `links_json` — previously
  veracity annotations, belief-network links, and contradiction cycle state
  were silently dropped on backup/restore.
- `DummyChat` grounds extraction now skips the system prompt and extracts
  real episode IDs from the reflection prompt, so the zero-dependency quickstart
  actually exercises reflection grounding.
- `reinforce_unresolved_contradictions()` accepts detected fact pairs and
  only reinforces contradictions re-detected in the current pass, making
  `contradiction_retirement_cycles` retire entries not re-detected in N passes
  rather than reinforcing every active contradiction unconditionally.
- Reflection grounding validation now includes historical episode IDs so
  long-term pattern hypotheses keep their citations.
- `SyncMemoryStore` no longer uses the deprecated `asyncio.get_event_loop()`
  probe; it now uses `asyncio.get_running_loop()` for forward compatibility.
- README status aligned with the package version and PyPI classifier.

### Added

- `Fact.retired` public API property: returns `True` when a fact was expired
  via the `__retired__:` sentinel, `False` when merely superseded.
