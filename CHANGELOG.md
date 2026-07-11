# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.5] - 2026-07-13

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

