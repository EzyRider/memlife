# memlife store.py Refactor Plan

## Goal
Split store.py (~2,500 lines) into focused mixin modules without changing
the public API. Every consumer (Ingrid, Nano, OpenClaw, tests, PyPI)
keeps working without modification.

## Approach: Mixin pattern
MemoryStore inherits from domain mixins. Each mixin has access to `self`
(the store instance) so no API change is needed. Files use leading
underscore to signal internal modules.

## File structure

```
src/memlife/
  store.py            # MemoryStore class definition, __init__, conn/lock
                      # property, transaction(), close, __enter__/__exit__
                      # Inherits all mixins. ~150 lines.
  _locked_conn.py     # _LockedConn proxy class (extracted from store.py)
                      # ~50 lines.
  _schema.py          # _init_schema, _migrate. SchemaMixin.
                      # ~180 lines.
  _episodes.py        # EpisodeStore mixin: remember, recall, recent,
                      # episodes_since, episodes_by_ids, search_episodes_by_tool,
                      # search_episodes_by_keyword, embed_episode,
                      # gap marker insertion, episodes_sample_historical.
                      # ~250 lines.
  _facts.py           # FactStore mixin: store_fact, _refresh_fact,
                      # adjust_confidence, _supersede_fact, check_conflicts,
                      # expire_fact, revise_fact, fact_by_id, resolve_fact,
                      # _active_facts, _facts_with_embeddings,
                      # _facts_with_embeddings_since, _normalize.
                      # ~350 lines.
  _journal.py         # JournalStore mixin: add_journal_entry, embed_journal_entry,
                      # journal_recent, journal_by_type, journal_contradictions,
                      # search_journal, journal_relevant, supersede_journal,
                      # retire_journal, consolidate_journal, list_contradictions,
                      # has_active_contradiction, touch_active_contradiction,
                      # reinforce_unresolved_contradictions,
                      # retire_stale_contradictions, link_journal_entries,
                      # _load_links, _journal_from_row, _active_journal_sql,
                      # annotate_fact, annotate_journal, _annotations_for.
                      # ~500 lines.
  _runs.py            # RunStore mixin: start_run, complete_run,
                      # save_checkpoint, get_last_checkpoint, trace_event,
                      # get_incomplete_run, list_sessions, create_session.
                      # ~200 lines.
  _gc.py              # GCMixin: run_gc, run_vacuum, recall_stats.
                      # ~120 lines.
  _embeddings.py      # EmbedMixin: embed_texts, embedding_health,
                      # backfill_embeddings, _maybe_store_vec,
                      # _serialize_vec, _deserialize_vec.
                      # ~200 lines.
  _triples.py         # TripleMixin: store_fact_triple, expire_triples_for_fact,
                      # current_truth, truth_as_of, triples_for_fact.
                      # ~120 lines.
```

## Shared state (stays on MemoryStore in store.py)
- self.config (MemoryConfig)
- self.embedder (Embedder | None)
- self.embedding_model_name (str)
- self.db_path (str)
- self.fact_merge_threshold (float)
- self.fact_conflict_threshold (float)
- self._embed_failures (int)
- self._conn (_LockedConn | None)
- self._lock (threading.RLock)
- self._recall_counters (dict[str, int])

## Dependencies between mixins
Each mixin uses `self.conn`, `self.config`, `self.embedder`, etc.
No mixin imports another — they all operate through `self` which is
the fully-constructed MemoryStore. This means no circular imports.

The only import chain is:
  _locked_conn.py  (no deps)
  _schema.py       (imports constants from store.py or passes them)
  _episodes.py     (imports from memlife.models)
  _facts.py        (imports from memlife.models, memlife.vectors)
  _journal.py      (imports from memlife.models, memlife.vectors)
  _runs.py         (no deps beyond stdlib)
  _gc.py           (no deps beyond stdlib)
  _embeddings.py   (imports from memlife.vec_backend, memlife.binary_vectors)
  _triples.py      (no deps beyond stdlib)
  store.py         (imports all mixins, defines MemoryStore)

## Execution plan

### Step 1: Extract _LockedConn (5 min)
- Move _LockedConn class from store.py to _locked_conn.py
- Import in store.py
- Run tests — should pass with no changes

### Step 2: Extract _schema.py SchemaMixin (10 min)
- Move _init_schema and _migrate into SchemaMixin
- Add to MemoryStore inheritance
- Run tests

### Step 3: Extract _runs.py RunMixin (10 min)
- Move start_run, complete_run, save_checkpoint, get_last_checkpoint,
  trace_event, get_incomplete_run, list_sessions, create_session
- Run tests

### Step 4: Extract _gc.py GCMixin (5 min)
- Move run_gc, run_vacuum, recall_stats
- Run tests

### Step 5: Extract _triples.py TripleMixin (5 min)
- Move store_fact_triple, expire_triples_for_fact, current_truth,
  truth_as_of, triples_for_fact
- Run tests

### Step 6: Extract _embeddings.py EmbedMixin (10 min)
- Move embed_texts, embedding_health, backfill_embeddings,
  _maybe_store_vec, _serialize_vec, _deserialize_vec
- Run tests

### Step 7: Extract _episodes.py EpisodeStore (15 min)
- Move remember, recall, recent, episodes_since, episodes_by_ids,
  search_episodes_by_tool, search_episodes_by_keyword, embed_episode,
  gap marker logic, episodes_sample_historical
- Run tests

### Step 8: Extract _facts.py FactStore (15 min)
- Move store_fact, _refresh_fact, adjust_confidence, _supersede_fact,
  check_conflicts, expire_fact, revise_fact, fact_by_id, resolve_fact,
  _active_facts, _facts_with_embeddings, _facts_with_embeddings_since,
  _normalize
- Run tests

### Step 9: Extract _journal.py JournalStore (20 min)
- Move all journal methods (largest mixin — ~500 lines)
- Run tests

### Step 10: Final cleanup (10 min)
- Remove dead imports from store.py
- Verify ruff clean
- Verify all 118 tests pass
- Verify `from memlife import MemoryStore` works
- Verify `pip install -e .` still works
- Update __init__.py if needed for new module exports

## Verification at each step
After every extraction:
1. `python -m pytest tests/ --tb=short -q` — must be 118 passed
2. `python -m ruff check src/memlife tests` — must be clean
3. `python -c "from memlife import MemoryStore"` — must import

## What does NOT change
- Public API: MemoryStore, SyncMemoryStore, all method signatures
- Tests: zero changes needed
- PyPI package structure: same files, same imports
- Ingrid / Nano / OpenClaw: zero changes needed
- pyproject.toml: no changes
- README: no changes

## Risk
Low. The mixin pattern is mechanical — move methods, add import, add
inheritance, run tests. No logic changes. The only risk is missing
an import that a method needs (e.g., `re`, `json`, `time`) — caught
immediately by the test suite.

## Estimated time
90-120 minutes with testing at each step. Suitable for a single session.

## Post-refactor benefits
- store.py: ~150 lines (class definition + conn/lock/init only)
- Largest mixin file: ~500 lines (_journal.py)
- Each domain is independently navigable
- New features go in the relevant mixin, not the megafile
- Easier for external agents to review specific domains