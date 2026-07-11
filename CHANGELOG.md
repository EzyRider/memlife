# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

