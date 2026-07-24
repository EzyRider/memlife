# memlife 0.6.12 Roadmap

**Branch:** `fix/0.6.12-thread-safety-consistency`  
**Target release:** 2026-07-31  
**Status:** planning — awaiting steering go/no-go on scope  

This release adjudicates the Ingrid backend audit dated 2026-07-24 (filed as
`/home/ezyrider/PROJECTS/ACTIVE/Ingrid/workspace/issues_and_fixes.md`) against
the current 0.6.11 source. The audited modules are unchanged between 0.6.10
and 0.6.11.

---

## 1. Adjudication summary

| # | Issue | Verdict | Severity | In 0.6.12? |
|---|-------|---------|----------|------------|
| 1 | Graph retrieval can surface superseded facts | **Mitigated in practice** — the direct fact loop in `_graph_expand()` already applies `f.superseded_by` to all loaded facts, including relationship-hop sources. Layering is fragile because filtering happens far from loading. | Low | No — refactor only |
| 2 | `retrieve()` silently swallows recall-path failures | **Live bug** — every recall path is `try/except Exception` with only `logger.debug`. No failure counters exist. | High | **Yes** |
| 3 | `MemoryConfig.from_env()` does not validate | **Live** — `from_env()` returns without calling `validate()`. Failure surfaces later at store creation. | Medium | **Yes** |
| 4 | Vector backend `delete()` interpolates table names | **Live** — `json_backend.py` and `binary_backend.py` use f-string table names without an allowlist. | High | **Yes** |
| 5 | MCP server hardcodes Ollama chat adapter | **Live** — `_get_reflector()` always instantiates `OllamaChat`. No `--chat-adapter` flag exists. | High | **Yes** |
| 6 | Entity extractor can extract sentence-start words as entities | **Partially mitigated** — blocklist + first-token check help, but newline collapse still removes sentence boundaries. | Low | No |
| 7 | Polyphonic recall voice-hit counters are semantically off | **Live** — counters measure fused-pool overlap with each voice's top-N, not source attribution. | Medium | **Yes** |
| 8 | `_supersede_fact()` savepoint handling is hard to reason about | **Live** — `transaction()` only yields the connection; the savepoint is effectively the whole transaction. | Medium | No — deferred to 0.6.13 |
| 9 | Embedding-cache GC does redundant hashing and holds the lock for a long time | **Live** — Python-side SHA-256 per row and streaming queries under the store lock. | Low | No |
| 10 | `metrics()` issues many small, non-atomic count queries | **Live** — counts are fetched separately and can be inconsistent under writes. | Low | No |
| 11 | README and CHANGELOG "Unreleased" sections are empty | **Live** — no public mention of recent graph-retrieval work beyond the CI matrix note. | Low | **Yes** (docs pass) |
| 12 | `BACKLOG.md` / `docs/ROADMAP.md` may contain stale items | **Confirmed stale** — `docs/ROADMAP.md` is a 0.5.0 draft from 2026-07-10 and does not reflect the 0.6.x reality. `BACKLOG.md` is mostly a status record. | Low | **Yes** (this file replaces it) |
| 13 | `SyncMemoryStore._run()` uses fragile RuntimeError string matching | **Live** — matches its own error message, but still fragile. | Low | No |
| 14 | Missing test coverage for several edge cases | **Live** — no regression tests for the issues above. | Medium | **Yes**, for items taken into 0.6.12 |

---

## 2. Scope decision

**In scope for 0.6.12 (four fixes + docs/tests):**

1. **retrieve() failure visibility** — log at `WARNING`, increment new `_recall_counters` failure keys, expose in `metrics()`.
2. **MemoryConfig.from_env() validation** — call `cfg.validate()` before returning; add `tests/test_config.py` regression.
3. **Vector backend delete() allowlist** — add `_TABLES = {"facts", "episodes", "journal"}` and raise `ValueError` for invalid `kind`.
4. **MCP server chat-adapter selection** — add `--chat-adapter {ollama,openai}`, `--chat-base-url`, `--chat-model`; instantiate `OpenAIChat` when requested.
5. **Polyphonic counter semantics** — count source attribution per candidate voice in the fused pool.
6. **CHANGELOG / README backfill** — document graph-retrieval hardening in 0.6.12 unreleased section.
7. **Regression tests** for the five code fixes above.

**Deferred to 0.6.13 / future:**

- #1 graph-retrieval layering refactor (not a live bug)
- #6 entity-extractor sentence-boundary rework
- #8 `_supersede_fact()` transaction/savepoint simplification
- #9 embedding-cache GC efficiency
- #10 `metrics()` atomic counts
- #13 `SyncMemoryStore._run()` robustness

Rationale: 0.6.12 is a tight patch release. The four high-priority items are
small, well-defined, and fix real operational gaps. The deferred items are
either not live bugs or require larger refactors that should not block the
patch.

---

## 3. Work items

### 3.1 `retrieve()` failure visibility

**Files:**
- `src/memlife/retrieval.py` — wrap each recall path's `except` block with `logger.warning(...)` and increment `self._recall_counters["{path}_failures"]`.
- `src/memlife/store.py` (or `_gc.py`) — include failure counters in `metrics()` output.

**Tests:**
- `tests/test_retrieval.py` — monkey-patch `recall_facts()` to raise and assert the failure counter increments and retrieval still returns a result.

### 3.2 `MemoryConfig.from_env()` validation

**Files:**
- `src/memlife/config.py` — add `cfg.validate()` at the end of `from_env()`.

**Tests:**
- `tests/test_config.py` — set `MEMLIFE_NAMESPACE` to an invalid value and assert `ValueError` is raised by `from_env()`.

### 3.3 Vector backend `delete()` allowlist

**Files:**
- `src/memlife/vector_backends/json_backend.py`
- `src/memlife/vector_backends/binary_backend.py`

Add:
```python
_TABLES = {"facts", "episodes", "journal"}
```
and validate `kind` before interpolating the table name.

**Tests:**
- `tests/test_vector_backends.py` — assert `ValueError` on invalid `kind` for both backends.

### 3.4 MCP server chat-adapter selection

**Files:**
- `src/memlife/mcp_server.py` — add CLI flags and branch in `_get_reflector()`.
- `src/memlife/adapters/openai.py` — ensure `OpenAIChat` constructor signature matches what `_get_reflector()` needs.

**Tests:**
- `tests/test_mcp.py` — create a server config with `chat_adapter="openai"` and verify the reflector's chat adapter is an `OpenAIChat` instance.

### 3.5 Polyphonic counter semantics

**Files:**
- `src/memlife/retrieval.py` — replace current `voice_hits_vector` / `voice_hits_text` / `voice_hits_source` / `voice_hits_veracity` / `voice_hits_recency` overlap counts with source-attribution counts keyed by actual voice name.

**Tests:**
- `tests/test_polyphonic.py` — update assertions to match source-attribution semantics.

### 3.6 Docs and changelog

**Files:**
- `CHANGELOG.md` — populate `[Unreleased]` with 0.6.12 items.
- `README.md` — update "What's new in 0.6.12" section.
- `docs/roadmap_0.6.12.md` — this file.

---

## 4. Release checklist

- [ ] Branch `fix/0.6.12-thread-safety-consistency` cut from `main`
- [ ] All five code fixes implemented
- [ ] Regression tests added and passing
- [ ] Full test suite green (`pytest`)
- [ ] `ruff check src/ tests/` clean
- [ ] `CHANGELOG.md` [Unreleased] populated
- [ ] `README.md` "What's new in 0.6.12" updated
- [ ] `pyproject.toml` and `memlife.__version__` bumped to `0.6.12`
- [ ] Merge to `main` and tag `v0.6.12`
- [ ] PyPI upload
- [ ] GitHub release notes

---

## 5. Decision register

| ID | Decision | Default | Notes |
|----|----------|---------|-------|
| R1 | 0.6.12 scope | Four high-priority fixes + docs/tests | Keeps patch small and shippable |
| R2 | Deferred items | 0.6.13 or later | Medium/low-risk refactors and polish |
| R3 | Branch name | `fix/0.6.12-thread-safety-consistency` | Continues the 0.6.8 thread-safety theme |
| R4 | Polyphonic counters | Source attribution | More meaningful telemetry than overlap counts |
| R5 | MCP chat adapter | `--chat-adapter` flag | Preserves backward compatibility (default `ollama`) |

---

*Drafted:* 2026-07-24  
*By:* Ingrid (adjudication of 0.6.10/0.6.11 audit)  
*Status:* pending James/Hermes steering confirmation
