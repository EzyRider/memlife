"""Unified retrieval — cross-layer memory scoring and formatting.

Pulls and ranks memories across all layers (episodes, facts, journal),
scores them by a unified metric, and formats the top N as structured context.

The unified score is:

    score = relevance × confidence × recency

where relevance is a weighted blend of:

    relevance = w_v * vector_sim + w_t * text_score + w_s * source_weight + w_r * veracity

Signals are normalised per query so vector, text, source and veracity
components share a common [0, 1] scale.  This makes the ranking transparent
and tunable via ``MemoryConfig``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from memlife.config import MemoryConfig
from memlife.models import Episode, Fact, JournalEntry
from memlife.store import MemoryStore
from memlife.vectors import cosine, recency_weight

logger = logging.getLogger(__name__)


# Default source-weight multipliers.  These are small nudges that reflect the
# layer's inherent reliability without overriding confidence/recency.
_SOURCE_WEIGHTS: dict[str, float] = {
    "user": 1.0,
    "tool": 0.95,
    "agent": 0.9,
    "journal": 0.85,
    "imported": 0.8,
    "episode": 0.6,
}


@dataclass
class _RecallSignals:
    """Raw and blended signals for one recall candidate."""

    item: Episode | Fact | JournalEntry
    kind: str
    labelled_text: str
    dedup_text: str
    fact_id: str
    vector_sim: float
    text_score: float
    source_weight: float
    veracity: float
    confidence: float
    recency: float
    relevance: float
    score: float
    why: str = ""


def _snippet_tokens(text: str) -> frozenset[str]:
    """Content tokens for Jaccard dedup.

    Lowercase, alphanumeric/underscore, length > 4 — the >4 filter drops
    common stopwords and short glue words that would inflate similarity
    between unrelated snippets.
    """
    return frozenset(
        t for t in re.findall(r"[A-Za-z0-9_]+", text.lower()) if len(t) > 4
    )


def _normalize_signal(values: list[float], eps: float = 1e-9) -> list[float]:
    """Min-max normalise a list of values to [0, 1].

    If all values are identical, return a list of 1.0 so strong absolute
    signals are not zeroed out — unless the identical value is effectively
    zero, in which case the signal is absent and should contribute nothing.
    """
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi < eps:
        return [0.0 for _ in values]
    if hi - lo < eps:
        return [1.0 for _ in values]
    return [(v - lo) / (hi - lo + eps) for v in values]


def _text_score_for(tokens: set[str], text: str) -> float:
    """Fraction of query tokens found in ``text`` (case-insensitive)."""
    if not tokens:
        return 0.0
    hay = text.lower()
    hits = sum(1 for t in tokens if t in hay)
    return hits / len(tokens)


def _source_weight_for(item: Episode | Fact | JournalEntry) -> float:
    """Layer/source reliability multiplier.

    Facts get the source from their ``source`` column; episodes and journal
    entries use the layer name as a fallback.
    """
    if isinstance(item, Fact):
        return _SOURCE_WEIGHTS.get(item.source, _SOURCE_WEIGHTS["agent"])
    if isinstance(item, Episode):
        return _SOURCE_WEIGHTS["episode"]
    if isinstance(item, JournalEntry):
        return _SOURCE_WEIGHTS["journal"]
    return 0.5


def _veracity_for_fact(store: MemoryStore, fact: Fact) -> float:
    """Compute veracity signal for a fact.

    Combines the fact's own confidence with temporal-triple support.  If the
    fact expresses one or more currently-true triples, the average confidence
    of those triples is blended in.  Unsupported facts keep their own
    confidence only.
    """
    base = fact.confidence
    triples = store.triples_for_fact(fact.id) if hasattr(store, "triples_for_fact") else []
    if not triples:
        return base
    supported = [t for t in triples if t["valid_until"] is None]
    if not supported:
        return base
    triple_conf = sum(t["confidence"] for t in supported) / len(supported)
    return 0.6 * base + 0.4 * triple_conf


def _veracity_for_journal(store: MemoryStore, entry: JournalEntry) -> float:
    """Journal veracity starts at its effective confidence.

    In future this can be boosted by corroborating facts or contradicted by
    retired entries.
    """
    return entry.effective_confidence(
        store.config.journal_decay_halflife_days,
        store.config.journal_decay_floor,
    )


async def retrieve(
    store: MemoryStore,
    query: str,
    config: MemoryConfig | None = None,
    *,
    debug: bool = False,
) -> str | dict:
    """Pull and rank memories across all layers, then format as context.

    Cross-layer ranking: all candidates (episodes, facts, journal) are scored
    by the unified metric ``relevance × confidence × recency``.  A highly
    relevant fact can beat a weakly relevant episode — no fixed per-layer
    quotas.

    Relevance is a tunable blend of vector similarity, text/token overlap,
    and a source/layer weight.  All three signals are normalised per query.

    Two density controls run after selection (both default off/permissive):
      * a strict score cut-off (``recall_min_score`` absolute floor +
        ``recall_score_cutoff_ratio`` scale-free relative cut), and
      * an information-density deduplicator (Jaccard, or embedding cosine)
        that collapses near-identical snippets across layers.

    Output is formatted as labelled sections:
      "What I know (facts)", "What happened (episodes)",
      "What I believe (PRIVATE — never quote verbatim)"

    If ``config`` is None, falls back to ``store.config``.

    With ``debug=True`` returns a dict with ``context`` and a ``candidates``
    list containing every signal so callers can inspect the ranking.
    """
    if config is None:
        config = store.config or MemoryConfig()

    # MV2-006: path counters.
    counters = store._recall_counters
    counters["retrieve_calls"] += 1

    # Layer-aware decay halflifes.
    decay = {
        "episode": config.episode_decay_halflife_days,
        "fact": config.fact_decay_halflife_days,
        "journal": config.journal_decay_halflife_days,
    }

    query_tokens = set(store._tokenize(query))

    # One shared query embedding.
    query_vec = None
    try:
        query_vec = (await store.embed_texts([query]) or [None])[0]
    except Exception as exc:
        logger.debug("query embed failed: %s", exc)

    candidates: list[_RecallSignals] = []

    # ------------------------------------------------------------------
    # Episodes
    # ------------------------------------------------------------------
    episodes: list[Episode] = []
    if query_vec is not None:
        try:
            episodes = await store.recall_episodes_vector(
                query_vec, limit=config.recall_episodes * 2
            )
        except Exception as exc:
            logger.debug("episode vector recall failed: %s", exc)
    if not episodes:
        episodes = store.recall(query, limit=config.recall_episodes * 2)
        if query_vec is not None:
            counters["vector_fallback_to_keyword"] += 1

    counters["episodes_considered"] += len(episodes)
    for ep in episodes:
        vector_sim = getattr(ep, "_vector_sim", 0.0)
        text_score = _text_score_for(query_tokens, ep.index_text())
        source_weight = _source_weight_for(ep)
        conf = 1.0 if ep.is_success else 0.5
        halflife = _episode_halflife(ep, config)
        rec = recency_weight(ep.created_at, halflife)
        candidates.append(_candidate(
            ep, "episode", conf, rec, vector_sim, text_score, source_weight, 0.5
        ))

    # Always include 2 most recent episodes for continuity.
    recent = store.recent(limit=2)
    seen_ids = {e.id for e in episodes}
    for ep in recent:
        if ep.id not in seen_ids:
            text_score = _text_score_for(query_tokens, ep.index_text())
            halflife = _episode_halflife(ep, config)
            rec = recency_weight(ep.created_at, halflife)
            candidates.append(_candidate(
                ep, "episode", 0.5, rec, 0.0, text_score, _SOURCE_WEIGHTS["episode"], 0.5
            ))

    # ------------------------------------------------------------------
    # Facts
    # ------------------------------------------------------------------
    try:
        facts = await store.recall_facts(
            query,
            limit=config.recall_facts * 2,
            query_vector=query_vec,
        )
    except Exception as exc:
        logger.debug("fact recall failed: %s", exc)
        facts = []

    counters["facts_considered"] += len(facts)
    for f in facts:
        vector_sim = getattr(f, "_vector_sim", 0.0)
        text_score = _text_score_for(query_tokens, f.content)
        source_weight = _source_weight_for(f)
        rec = recency_weight(f.updated_at, decay["fact"])
        veracity = _veracity_for_fact(store, f)
        eff_conf = f.effective_confidence(
            config.fact_decay_halflife_days, config.fact_decay_floor
        )
        candidates.append(_candidate(
            f, "fact", eff_conf, rec, vector_sim, text_score, source_weight, veracity,
            fact_id=f.id,
        ))

    # ------------------------------------------------------------------
    # Journal
    # ------------------------------------------------------------------
    notes: list[JournalEntry] = []
    if query_vec is not None:
        try:
            notes = await store.recall_journal_vector(
                query_vec,
                limit=config.recall_journal * 2,
            )
        except Exception as exc:
            logger.debug("journal vector recall failed: %s", exc)
    if not notes:
        notes = store.journal_relevant(query, limit=config.recall_journal * 2)
        if query_vec is not None:
            counters["vector_fallback_to_keyword"] += 1

    counters["journal_considered"] += len(notes)
    for j in notes:
        vector_sim = getattr(j, "_vector_sim", 0.0)
        text_score = _text_score_for(query_tokens, j.content)
        source_weight = _source_weight_for(j)
        eff_conf = j.effective_confidence(
            config.journal_decay_halflife_days,
            config.journal_decay_floor,
        )
        rec = recency_weight(j.created_at, decay["journal"])
        veracity = _veracity_for_journal(store, j)
        candidates.append(_candidate(
            j, "journal", eff_conf, rec, vector_sim, text_score, source_weight, veracity
        ))

    # ------------------------------------------------------------------
    # Blend and rank
    # ------------------------------------------------------------------
    _blend_candidates(candidates, config)
    candidates.sort(key=lambda c: c.score, reverse=True)

    top_n = config.recall_episodes + config.recall_facts + config.recall_journal
    pool = _apply_recall_cutoff(candidates[:top_n], config)

    # MV2-I004: optional polyphonic recall fuses per-voice rankings via RRF.
    if config.use_polyphonic_recall:
        from memlife import polyphonic

        counters["polyphonic_fusion_calls"] += 1
        voice_groups: dict[str, list[_RecallSignals]] = {
            "vector": sorted(candidates, key=lambda c: c.vector_sim, reverse=True),
            "text": sorted(candidates, key=lambda c: c.text_score, reverse=True),
            "source": sorted(candidates, key=lambda c: c.source_weight, reverse=True),
            "veracity": sorted(candidates, key=lambda c: c.veracity, reverse=True),
            "recency": sorted(candidates, key=lambda c: c.recency, reverse=True),
        }
        pool = polyphonic.fuse_candidates(voice_groups, config)[:top_n]
        # Count how many candidates each voice contributed to the fused pool.
        voice_ids = {c.item.id for c in pool if getattr(c.item, "id", None)}
        counters["voice_hits_vector"] += len(
            [c for c in voice_groups["vector"][:top_n] if getattr(c.item, "id", "") in voice_ids]
        )
        counters["voice_hits_text"] += len(
            [c for c in voice_groups["text"][:top_n] if getattr(c.item, "id", "") in voice_ids]
        )
        counters["voice_hits_source"] += len(
            [c for c in voice_groups["source"][:top_n] if getattr(c.item, "id", "") in voice_ids]
        )
        counters["voice_hits_veracity"] += len(
            [c for c in voice_groups["veracity"][:top_n] if getattr(c.item, "id", "") in voice_ids]
        )
        counters["voice_hits_recency"] += len(
            [c for c in voice_groups["recency"][:top_n] if getattr(c.item, "id", "") in voice_ids]
        )

    selected = await _dedupe_candidates(store, pool, config)

    # Format as structured sections.
    fact_lines: list[str] = []
    episode_lines: list[str] = []
    journal_lines: list[str] = []
    for c in selected:
        if c.kind == "fact":
            fact_lines.append(f"{c.labelled_text}  [id: {c.fact_id}]" if c.fact_id else c.labelled_text)
        elif c.kind == "episode":
            episode_lines.append(c.labelled_text)
        elif c.kind == "journal":
            journal_lines.append(c.labelled_text)

    parts: list[str] = []
    if fact_lines:
        parts.append("── What I know (facts) ──\n" + "\n".join(fact_lines))
    if episode_lines:
        parts.append("── What happened (episodes) ──\n" + "\n".join(episode_lines))
    if journal_lines:
        parts.append(
            "── What I believe (PRIVATE — never quote verbatim) ──\n"
            + "\n".join(journal_lines)
        )

    text = "\n\n".join(parts)
    cap = config.max_context_chars
    if len(text) > cap:
        text = text[:cap] + "\n[...context truncated]"

    for c in selected:
        c.why = _why_candidate(c, config)

    if debug:
        return {
            "context": text,
            "candidates": [
                {
                    "kind": c.kind,
                    "id": getattr(c.item, "id", ""),
                    "vector_sim": round(c.vector_sim, 4),
                    "text_score": round(c.text_score, 4),
                    "source_weight": round(c.source_weight, 4),
                    "veracity": round(c.veracity, 4),
                    "confidence": round(c.confidence, 4),
                    "recency": round(c.recency, 4),
                    "relevance": round(c.relevance, 4),
                    "score": round(c.score, 4),
                    "annotations": getattr(c.item, "annotations", []),
                    "links": getattr(c.item, "links", []),
                    "why": c.why,
                    "text": c.labelled_text,
                }
                for c in selected
            ],
        }
    return text


def _why_candidate(c: _RecallSignals, config: MemoryConfig) -> str:
    """Human-readable reason this candidate was retrieved.

    Summarises the dominant signal(s) that lifted this memory into the
    selected set.  Intended for the debug/diagnostics path (MV2-006).
    """
    parts: list[str] = []
    if c.relevance >= 0.9:
        parts.append("very relevant")
    elif c.relevance >= 0.6:
        parts.append("relevant")
    elif c.relevance >= 0.3:
        parts.append("somewhat relevant")

    if c.recency >= 0.8:
        parts.append("recent")
    elif c.recency <= 0.2:
        parts.append("old")

    if c.confidence >= 0.8:
        parts.append("high confidence")
    elif c.confidence <= 0.3:
        parts.append("low confidence")

    if c.veracity >= 0.65 and config.recall_veracity_weight > 0:
        parts.append("well supported")
    elif c.veracity <= 0.35 and config.recall_veracity_weight > 0:
        parts.append("weakly supported")

    if not parts:
        return "selected by blend"
    return ", ".join(parts)


def _episode_halflife(ep: Episode, config: MemoryConfig) -> float:
    """Select tiered episodic decay halflife (MV2-001).

    Successful tool episodes linger longest, failures and plain observations
    fade fast.  Gap markers keep the default episode halflife.
    """
    if ep.is_gap_marker:
        return config.episode_decay_halflife_days
    if ep.is_failure:
        return config.episode_failure_halflife_days
    if ep.is_success and ep.has_tool_calls:
        return config.episode_tool_success_halflife_days
    return config.episode_observation_halflife_days


def _candidate(
    item: Episode | Fact | JournalEntry,
    kind: str,
    confidence: float,
    recency: float,
    vector_sim: float,
    text_score: float,
    source_weight: float,
    veracity: float,
    fact_id: str = "",
) -> _RecallSignals:
    """Build a recall-signals object with a human-readable label."""
    if isinstance(item, Episode):
        text = f"[{item.outcome}] {item.task}"
        if item.summary:
            text += f" → {item.summary[:120]}"
        dedup = f"{item.task} {item.summary}"
    elif isinstance(item, Fact):
        text = f"[conf={confidence:.2f}] {item.content}"
        dedup = item.content
    elif isinstance(item, JournalEntry):
        text = f"[{item.type} conf={confidence:.2f}] {item.content}"
        dedup = item.content
    else:
        text = str(item)
        dedup = text

    return _RecallSignals(
        item=item,
        kind=kind,
        labelled_text=text,
        dedup_text=dedup,
        fact_id=fact_id,
        vector_sim=vector_sim,
        text_score=text_score,
        source_weight=source_weight,
        veracity=veracity,
        confidence=confidence,
        recency=recency,
        relevance=0.0,
        score=0.0,
    )


def _blend_candidates(candidates: list[_RecallSignals], config: MemoryConfig) -> None:
    """Normalise per-query signals and blend them into relevance/score."""
    if not candidates:
        return

    vector_sims = [c.vector_sim for c in candidates]
    text_scores = [c.text_score for c in candidates]
    source_weights = [c.source_weight for c in candidates]
    veracities = [c.veracity for c in candidates]

    norm_vector = _normalize_signal(vector_sims)
    norm_text = _normalize_signal(text_scores)
    norm_source = _normalize_signal(source_weights)
    norm_veracity = _normalize_signal(veracities)

    w_v = config.recall_vector_weight
    w_t = config.recall_text_weight
    w_s = config.recall_source_weight
    w_r = config.recall_veracity_weight
    total = w_v + w_t + w_s + w_r
    if total < 1e-9:
        total = 1.0

    for c, nv, nt, ns, nr in zip(candidates, norm_vector, norm_text, norm_source, norm_veracity):
        c.relevance = (w_v * nv + w_t * nt + w_s * ns + w_r * nr) / total
        c.score = c.relevance * c.confidence * c.recency


def _apply_recall_cutoff(
    pool: list[_RecallSignals], config: MemoryConfig
) -> list[_RecallSignals]:
    """Apply the strict score cut-off to a pre-sorted (desc) candidate pool."""
    if not pool:
        return pool
    min_score = config.recall_min_score
    if min_score > 0:
        pool = [c for c in pool if c.score >= min_score]
    ratio = config.recall_score_cutoff_ratio
    if ratio > 0 and len(pool) > 1:
        top = pool[0].score
        if top > 0:
            kept = [c for c in pool if c.score >= ratio * top]
            min_keep = max(1, config.recall_journal)
            if len(kept) < min_keep:
                kept = pool[:min_keep]
            pool = kept
    return pool


async def _dedupe_candidates(
    store: MemoryStore,
    selected: list[_RecallSignals],
    config: MemoryConfig,
) -> list[_RecallSignals]:
    """Collapse near-identical snippets so the same fact isn't injected 3×."""
    threshold = config.recall_dedup_threshold
    if threshold >= 1.0 or not selected:
        return selected
    if config.recall_dedup_method == "embedding":
        return await _dedupe_embedding(store, selected, threshold)
    return _dedupe_jaccard(selected, threshold)


def _dedupe_jaccard(
    selected: list[_RecallSignals], threshold: float
) -> list[_RecallSignals]:
    kept: list[_RecallSignals] = []
    for cand in selected:
        toks = _snippet_tokens(cand.dedup_text)
        if not toks:
            kept.append(cand)
            continue
        is_dup = False
        for k in kept:
            ktoks = _snippet_tokens(k.dedup_text)
            if not ktoks:
                continue
            union = toks | ktoks
            inter = toks & ktoks
            if inter and len(inter) / len(union) >= threshold:
                is_dup = True
                break
            if len(toks) >= 2 and toks <= ktoks:
                is_dup = True
                break
        if not is_dup:
            kept.append(cand)
    return kept


async def _dedupe_embedding(
    store: MemoryStore,
    selected: list[_RecallSignals],
    threshold: float,
) -> list[_RecallSignals]:
    """Embedding-cosine dedup. One batched embed call for the snippets."""
    texts = [c.dedup_text for c in selected]
    try:
        vecs = await store.embed_texts(texts) or []
    except Exception as exc:
        logger.debug("dedup embed failed (%s); skipping dedup", exc)
        return selected
    if len(vecs) != len(selected):
        return selected
    kept_idx: list[int] = []
    for i, vec in enumerate(vecs):
        if vec is None:
            kept_idx.append(i)
            continue
        dup = False
        for j in kept_idx:
            jv = vecs[j]
            if jv is not None and cosine(vec, jv) >= threshold:
                dup = True
                break
        if not dup:
            kept_idx.append(i)
    return [selected[i] for i in kept_idx]
