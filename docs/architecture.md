# Architecture

## Overview

memlife is a four-tier lifecycle memory system backed by SQLite. It provides
episodic, semantic, and reflective memory with automatic decay, revision, and
garbage collection. The system is designed for single-agent use with zero
external dependencies in its no-LLM mode.

## The Four Tiers

### 1. Episodes (Episodic Memory)

Raw events — one agent run, one interaction, one tool call sequence. Episodes
are the shortest-lived layer. They include:

- `task` — what was asked
- `outcome` — success/failed/cancelled
- `summary` — what happened
- `tool_calls` — indexed for behavioural search

Episodes support keyword recall and optional vector recall (when an embedder
is configured). They're the "what happened" layer.

### 2. Facts (Semantic Memory)

Durable truths about the user, the environment, and the work. Facts have:

- `confidence` — capped at 0.99 (never immutable)
- `embedding` — for cosine similarity recall
- `embedding_model` — for versioning (detect stale vectors)
- `superseded_by` — links to the replacement when revised

Facts go through a layered dedup at store time:
1. Exact-normalised content match → return existing ID
2. Containment check → keep the longer, more specific wording
3. Semantic merge (cosine ≥ 0.90) → supersede the lower-confidence one

### 3. Journal (Reflective Memory)

Private reflections — observations, hypotheses, revisions, and contradictions.
Written by the reflection loop, not by the agent directly. Journal entries have:

- `confidence` that decays over time (30-day halflife, floored at 0.15)
- `superseded_by` for revisions (closes the loop on prior beliefs)
- `type` — observation, hypothesis, revision, or contradiction

Journal entries are never quoted verbatim to the user. They shape tone and
assumptions silently.

### 4. Decay/Prune (Garbage Collection)

Everything degrades:
- Facts: confidence ceiling at 0.99, recall bumps +0.05, revision supersedes
- Journal: 30-day confidence halflife, retires below 0.15 floor
- Superseded facts: pruned after 90 days
- Superseded journal: pruned after 90 days
- Completed runs + checkpoints: pruned after 60 days
- Reflection metrics: pruned after 30 days
- VACUUM reclaims disk space after pruning

## Unified Retrieval

Before every response, memories are pulled from all three layers and scored
by a unified metric:

```
score = relevance × confidence × recency
```

- `relevance` — cosine similarity (vector) or keyword overlap
- `confidence` — fact confidence or journal effective confidence (after decay)
- `recency` — exponential decay weight (14-day halflife)

Candidates are pooled across all layers, ranked, cut off, and deduplicated
(Jaccard or embedding cosine). The top N are formatted as labelled sections:

```
── What I know (facts) ──
── What happened (episodes) ──
── What I believe (PRIVATE — never quote verbatim) ──
```

## Reflection Loop

The reflection loop runs nightly (or on demand). It:

1. Gathers pending (unreflected) episodes
2. Samples 15 historical episodes across 5 time bins
3. Pulls 10 recent + 15 historical journal entries
4. Asks the LLM to synthesise observations, hypotheses, and revisions
5. Optionally runs a critic pass (second LLM call) to gate entries on:
   - Grounding (is it supported by evidence?)
   - Non-obviousness (is it insightful?)
   - Redundancy (does it repeat a prior entry?)
6. Stores kept entries, marks episodes as reflected
7. Runs consolidation (retire low-confidence, merge near-duplicates)
8. Detects contradictions (incremental: only new facts vs full set)

The whole pass is bounded by `reflection_total_timeout` (default 300s).
Per-call timeout is `reflection_timeout` (default 120s).

## Embedding Versioning

Every vector stores the model name that created it. When the configured
embedding model changes:

1. `embedding_health()` reports stale count (vectors with mismatched model)
2. `/stats` or `memlife://health` surfaces the staleness
3. `backfill_embeddings()` re-embeds all stale vectors in batches

This prevents a mixed embedding space from silently degrading recall quality.

## Adapters

The `Embedder` and `ChatCallable` protocols are the injection points:

```python
class Embedder(Protocol):
    async def embed(self, texts: Sequence[str]) -> list[list[float]] | None: ...

class ChatCallable(Protocol):
    async def chat(self, messages: list[dict], model: str) -> str: ...
```

Built-in adapters:
- `DummyEmbedder` / `DummyChat` — zero dependencies, hash-based
- `OllamaEmbedder` / `OllamaChat` — Ollama API
- `OpenAIEmbedder` / `OpenAIChat` — OpenAI API
- `STEmbedder` — Sentence Transformers (local)

## SQLite Schema

Seven tables:
- `episodes` — episodic memory with optional embeddings
- `facts` — semantic memory with confidence and embeddings
- `journal` — reflective memory with decay and supersession
- `episode_tools` — index of tool calls per episode
- `agent_runs` + `checkpoints` — run tracking and resumption
- `reflection_queue` — pending episodes for reflection
- `reflection_metrics` — per-reflection quality metrics
- `sessions` — conversation persistence

WAL mode, partial indexes on embedding columns, additive migrations.

## Limitations

- **Single-process:** SQLite is single-writer. Not suitable for multi-agent
  concurrent access without a server wrapper.
- **Single-agent:** No multi-user isolation. Designed for one agent's memory.
- **Reflection quality depends on the LLM.** The critic gate helps, but a
  weak model produces weak reflections.