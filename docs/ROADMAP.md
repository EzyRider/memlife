# memlife Roadmap

## Current state: 0.6.12

memlife is a production-stable, single-agent lifecycle memory system. The 0.6.x
series focuses on hardening, observability, and adapter coverage while keeping
the core four-tier model unchanged.

This roadmap supersedes the earlier 0.5.0 planning draft.

---

## 0.6.12: Thread-safety & operational hardening

**Target release:** 2026-07-31  
**Branch:** `fix/0.6.12-thread-safety-consistency`

This release addresses the backend audit dated 2026-07-24. The scope is
deliberately tight: four live bugs, one telemetry fix, docs, and regression
tests. Larger refactors are deferred to 0.7.0.

### In scope

| Issue | Fix | Severity |
|-------|-----|----------|
| `retrieve()` silently swallows recall-path failures | Log at `WARNING`, increment `*_failures` counters, expose in `metrics()` | High |
| `MemoryConfig.from_env()` does not validate | Call `cfg.validate()` before returning | Medium |
| Vector backend `delete()` interpolates table names | Allowlist `facts`/`episodes`/`journal`; raise `ValueError` | High |
| MCP server hardcodes Ollama chat adapter | Add `--chat-adapter {ollama,openai}`, `--chat-base-url`, `--chat-api-key` | High |
| Polyphonic voice-hit counters are semantically off | Count source attribution per fused candidate | Medium |
| CHANGELOG / README stale | Backfill 0.6.12 notes | Low |
| `docs/ROADMAP.md` stale | Replace with this document | Low |

### Deferred to 0.7.0

- Graph-retrieval layering refactor (code smell, not a live bug)
- Entity-extractor sentence-boundary precision
- `_supersede_fact()` savepoint simplification
- Embedding-cache GC efficiency
- `metrics()` atomic counts
- `SyncMemoryStore._run()` RuntimeError string matching

---

## 0.7.0 and beyond (planned)

- Lifecycle improvements: retention/retirement refinements, GC efficiency,
  atomic metrics.
- Graph layer polish: entity extraction precision, retrieval layering cleanup.
- Multi-agent identity and collaborative attestation (research track — not
  committed to a version yet).
- Backup/restore API and namespace-aware import/export.

---

## Historical note

The 0.5.0 planning content that previously lived in this file (namespaces as
new work, vector backend ABC design, reflection audit APIs) has been mostly
shipped or superseded by the 0.6.x implementation. The 0.5.0 draft is retained
in git history for reference.
