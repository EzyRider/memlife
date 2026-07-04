"""Unified retrieval — cross-layer memory scoring and formatting.

Pulls and ranks memories across all layers (episodes, facts, journal),
scores them by a unified metric — relevance × confidence × recency —
and formats the top N as structured context.
"""

from __future__ import annotations

import logging
import re

from memlife.config import MemoryConfig
from memlife.store import MemoryStore
from memlife.vectors import cosine, recency_weight

logger = logging.getLogger(__name__)


def _snippet_tokens(text: str) -> frozenset[str]:
    """Content tokens for Jaccard dedup.

    Lowercase, alphanumeric/underscore, length > 4 — the >4 filter drops
    common stopwords and short glue words that would inflate similarity
    between unrelated snippets.
    """
    return frozenset(
        t for t in re.findall(r"[A-Za-z0-9_]+", text.lower()) if len(t) > 4
    )


# Candidate tuple type:
# (score, kind, labelled_text, exempt_from_floor, dedup_text, fact_id)
_Candidate = tuple[float, str, str, bool, str, str]


async def retrieve(store: MemoryStore, query: str, config: MemoryConfig | None = None) -> str:
    """Pull and rank memories across all layers, then format as context.

    Cross-layer ranking: all candidates (episodes, facts, journal) are scored
    by a unified metric — ``relevance × confidence × recency`` — and the top N
    overall are selected. A highly relevant fact can beat a weakly relevant
    episode — no fixed per-layer quotas.

    Two density controls run after selection (both default off/permissive):
      * a strict score cut-off (``recall_min_score`` absolute floor +
        ``recall_score_cutoff_ratio`` scale-free relative cut), and
      * an information-density deduplicator (Jaccard, or embedding cosine)
        that collapses near-identical snippets across layers.

    Output is formatted as labelled sections:
      "What I know (facts)", "What happened (episodes)",
      "What I believe (PRIVATE — never quote verbatim)"

    If ``config`` is None, falls back to ``store.config``.
    """
    if config is None:
        config = store.config or MemoryConfig()
    rhl = config.recency_halflife_days

    # One shared query embedding.
    query_vec = None
    try:
        query_vec = (await store.embed_texts([query]) or [None])[0]
    except Exception as exc:
        logger.debug("query embed failed: %s", exc)

    candidates: list[_Candidate] = []

    # Episodes — vector + keyword, plus always a few recent.
    episodes: list = []
    if query_vec is not None:
        try:
            episodes = await store.recall_episodes_vector(
                query_vec, limit=config.recall_episodes * 2
            )
        except Exception as exc:
            logger.debug("episode vector recall failed: %s", exc)
    if not episodes:
        episodes = store.recall(query, limit=config.recall_episodes * 2)
    for ep in episodes:
        rel = getattr(ep, "_relevance", 0.0)
        conf = 1.0 if ep.outcome == "success" else 0.5
        score = rel * conf * recency_weight(ep.created_at, rhl)
        text = f"[{ep.outcome}] {ep.task}"
        if ep.summary:
            text += f" → {ep.summary[:120]}"
        candidates.append(
            (score, "episode", text, False, f"{ep.task} {ep.summary}", "")
        )

    # Always include 2 most recent episodes for continuity.
    recent = store.recent(limit=2)
    seen_ids = {e.id for e in episodes}
    for ep in recent:
        if ep.id not in seen_ids:
            candidates.append(
                (
                    recency_weight(ep.created_at, rhl) * 0.15,
                    "episode",
                    f"[recent {ep.outcome}] {ep.task}",
                    True,
                    ep.task,
                    "",
                )
            )

    # Facts.
    try:
        facts = await store.recall_facts(
            query,
            limit=config.recall_facts * 2,
            query_vector=query_vec,
        )
    except Exception as exc:
        logger.debug("fact recall failed: %s", exc)
        facts = []
    for f in facts:
        rel = getattr(f, "_relevance", 0.0)
        score = rel * f.confidence * recency_weight(f.updated_at, rhl)
        candidates.append(
            (score, "fact", f"[conf={f.confidence:.2f}] {f.content}", False, f.content, f.id)
        )

    # Journal — vector + keyword.
    notes: list = []
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
    for j in notes:
        rel = getattr(j, "_relevance", 0.0)
        eff = j.effective_confidence(
            config.journal_decay_halflife_days,
            config.journal_decay_floor,
        )
        score = rel * eff * recency_weight(j.created_at, rhl)
        candidates.append(
            (score, "journal", f"[{j.type} conf={eff:.2f}] {j.content}", False, j.content, "")
        )

    # Rank, cut off, dedupe.
    candidates.sort(key=lambda c: c[0], reverse=True)
    top_n = config.recall_episodes + config.recall_facts + config.recall_journal
    pool = _apply_recall_cutoff(candidates[:top_n], config)
    selected = await _dedupe_candidates(store, pool, config)

    # Format as structured sections.
    fact_lines: list[str] = []
    episode_lines: list[str] = []
    journal_lines: list[str] = []
    for _, kind, text, _, _, fid in selected:
        if kind == "fact":
            fact_lines.append(f"{text}  [id: {fid}]" if fid else text)
        elif kind == "episode":
            episode_lines.append(text)
        elif kind == "journal":
            journal_lines.append(text)

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
    return text


def _apply_recall_cutoff(pool: list[_Candidate], config: MemoryConfig) -> list[_Candidate]:
    """Apply the strict score cut-off to a pre-sorted (desc) candidate pool."""
    if not pool:
        return pool
    min_score = config.recall_min_score
    if min_score > 0:
        pool = [c for c in pool if c[3] or c[0] >= min_score]
    ratio = config.recall_score_cutoff_ratio
    if ratio > 0 and len(pool) > 1:
        top = pool[0][0]
        if top > 0:
            kept = [c for c in pool if c[3] or c[0] >= ratio * top]
            min_keep = max(1, config.recall_journal)
            if len(kept) < min_keep:
                kept = pool[:min_keep]
            pool = kept
    return pool


async def _dedupe_candidates(
    store: MemoryStore,
    selected: list[_Candidate],
    config: MemoryConfig,
) -> list[_Candidate]:
    """Collapse near-identical snippets so the same fact isn't injected 3×."""
    threshold = config.recall_dedup_threshold
    if threshold >= 1.0 or not selected:
        return selected
    if config.recall_dedup_method == "embedding":
        return await _dedupe_embedding(store, selected, threshold)
    return _dedupe_jaccard(selected, threshold)


def _dedupe_jaccard(selected: list[_Candidate], threshold: float) -> list[_Candidate]:
    # MF-016: kept list stores (score, kind, text, exempt, dedup_text_str, toks, fid)
    # so we can return dedup_text as the string the _Candidate type expects,
    # while using toks (frozenset) for Jaccard comparison.
    kept: list[tuple[float, str, str, bool, str, frozenset[str], str]] = []
    for score, kind, text, exempt, dedup_text, fid in selected:
        toks = _snippet_tokens(dedup_text)
        is_dup = False
        for _, _, _, _, _, ktoks, _ in kept:
            if not toks or not ktoks:
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
            kept.append((score, kind, text, exempt, dedup_text, toks, fid))
    return [(s, k, t, e, _dt, fid) for s, k, t, e, _dt, _, fid in kept]


async def _dedupe_embedding(
    store: MemoryStore,
    selected: list[_Candidate],
    threshold: float,
) -> list[_Candidate]:
    """Embedding-cosine dedup. One batched embed call for the snippets."""
    texts = [c[4] for c in selected]
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