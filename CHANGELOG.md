# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `MemoryStore.metrics()` returns a public `Metrics` snapshot with counts,
  embedding coverage, reflection aggregates, recall counters, and DB metadata.
  Exposed on `SyncMemoryStore` and rendered by the `memlife://stats` MCP
  resource.
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
  counts, embeddings, reflection, and recall sections.

### Fixed

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
