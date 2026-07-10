# memlife — guidance for AI assistants

## What this project is

memlife is a **lifecycle memory system for AI agents**, not a generic vector database.
It stores memories in four tiers (episodes, facts, journal, and a decay/prune layer)
and retrieves them with a unified score:

    score = relevance × confidence × recency

The core idea is **graceful degradation**: memories decay, get revised, are retired,
and the system garbage-collects stale data. Perfect recall is not the goal —
appropriate forgetting is.

## How the code is organised

- `src/memlife/store.py` — thin `MemoryStore` shell: lifecycle plumbing only.
- `src/memlife/_*.py` — domain mixins (schema, runs, GC, triples, embeddings,
  episodes, facts, journal, locked connection, shared utils).
- `src/memlife/config.py` — `MemoryConfig`, the single source of truth for settings.
- `src/memlife/reflection.py` — `Reflector`, the nightly/critic-gated reflection loop.
- `src/memlife/retrieval.py` — cross-layer `retrieve()` and polyphonic recall.
- `src/memlife/models.py` — data classes: `Episode`, `Fact`, `JournalEntry`.
- `src/memlife/protocols.py` — `Embedder` and `ChatCallable` protocols.
- `src/memlife/embedders.py` — `DummyEmbedder` (bag-of-words, zero deps).
- `src/memlife/llm.py` — `DummyChat` and helpers.
- `src/memlife/vectors.py` — `cosine`, `recency_weight`.
- `src/memlife/adapters/` — optional Ollama, OpenAI, sentence-transformers adapters.
- `src/memlife/mcp_server.py` — stdio MCP server exposing memory tools/resources.
- `tests/` — 118+ pytest suite.

## Design principles

1. **Zero-dependency quickstart works.** `MemoryStore()` with no embedder or chat
   can store, retrieve, decay, and GC. Only reflection needs an LLM.
2. **Thread safety is mandatory.** `_LockedConn` wraps the SQLite connection and
   serialises access via `RLock`. `transaction()` gives multi-statement atomicity.
3. **WAL + busy_timeout are on by default.** See MF-002. Don't change this without
   a strong reason.
4. **Config is the source of truth.** Most tunables live in `MemoryConfig`. When
   adding a new setting, put it there and thread it through, don't add one-off
   constructor arguments.
5. **Don't break the public API.** `MemoryStore`, `MemoryConfig`, `DummyEmbedder`,
   `SyncMemoryStore`, and the package-level `__all__` are the contract.
6. **Tests run after every change.** `python -m pytest --tb=short -q`
7. **Ruff must stay clean.** `python -m ruff check src/memlife tests`

## Common tasks

### Add a new memory field

1. Add the column to `_schema.py` initial `CREATE TABLE` (and `_migrate()` for
   backward compatibility with existing DBs).
2. Update the relevant model in `models.py` if it surfaces in public objects.
3. Update `_journal_from_row`, `Episode.from_row`, or equivalent row constructors.
4. Add/update tests.

### Add a new config knob

1. Add a typed default to `MemoryConfig` with a doc comment.
2. Thread it to the consumer. Don't duplicate parameters in `Reflector` constructors
   when `MemoryConfig` already has the value.
3. Add a test that exercises the default and an override.

### Change the vector backend

- The current backends are JSON-in-SQLite, binary vectors, and optional sqlite-vec.
- `MemoryConfig.use_sqlite_vec` / `use_binary_vectors` are the legacy booleans.
- For 0.5.0 we plan to move to a `VectorBackend` enum; see `docs/ROADMAP.md`.

### Run reflection

```python
from memlife import MemoryStore, MemoryConfig
from memlife.adapters.ollama import OllamaChat

store = MemoryStore(MemoryConfig(), embedder=...)  # optional
reflector = Reflector(memory=store, model_chat=OllamaChat(model="..."))
result = await reflector.reflect()
```

## Pitfalls

- **Never construct raw SQL with column names from untrusted input.**
  `import_jsonl()` is the one place that did this (MF-012) and needs a whitelist.
- **Don't add a `namespace` column filter and forget it somewhere.**
  For 0.5.0 we chose separate DB files per namespace to avoid leakage.
- **Don't run `VACUUM` inside `run_gc()`.** It's split out as `run_vacuum()` for
  a reason (MF-006).
- **Don't recreate the `Reflector` every pass unless you restore
  `last_contradiction_scan`.** That disables incremental contradiction scanning
  (MF-003).
- **Don't call `OllamaEmbedder.session` synchronously outside an async context.**
  It's a latent issue; proper fix is tracked but low risk.

## Release process

1. Update `BACKLOG.md` / `docs/ROADMAP.md` if relevant.
2. Update version in `pyproject.toml` and `src/memlife/__init__.py`.
3. Update `README.md` if install commands or examples changed.
4. Run full tests and ruff.
5. `python -m build`
6. `python -m twine upload dist/memlife-VERSION*`
7. `git tag vVERSION && git push origin main --tags`
8. Write GitHub release notes.

## Testing

```bash
cd /path/to/memlife
source .venv/bin/activate
python -m pytest --tb=short -q
python -m ruff check src/memlife tests
```

## Notes for agent consumers

- memlife is **single-agent per store** today. Multi-user isolation is coming in
  0.5.0 via namespaces; multi-agent identity is a later research track.
- The journal is **private** — retrieve it into context, never quote it verbatim.
- All memory confidence is capped at `0.99` (`MAX_FACT_CONFIDENCE`) so every fact
  remains revisable.

---

*This file is meant for AI assistants working on or with memlife. For user-facing
 documentation, see README.md and docs/*.md.*
