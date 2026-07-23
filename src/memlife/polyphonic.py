"""Polyphonic recall — optional RRF fusion across retrieval voices (MV2-I004).

When ``MemoryConfig.use_polyphonic_recall`` is True, ``polyphonic_retrieve``
collects candidates from separate voices (vector, text, temporal, source,
veracity) and merges them with Reciprocal Rank Fusion.  The default
``retrieve`` path remains unchanged so the baseline stays simple and
predictable.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memlife.config import MemoryConfig
    from memlife.retrieval import _RecallSignals

logger = logging.getLogger(__name__)

# Default RRF constant.  Higher values flatten the fusion; 60 is a common
# default from the original RRF paper.
_RRF_K = 60


def reciprocal_rank_fusion(
    rankings: list[list[tuple[str, float]]],
    *,
    k: int = _RRF_K,
) -> dict[str, float]:
    """Merge multiple ranked lists into a single score per item id.

    ``rankings`` is a list of lists, each ordered best-to-worst, where each
    element is ``(item_id, voice_score)``.  The voice_score is ignored by RRF;
    only rank matters.  Returns ``{item_id: rrf_score}``.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, (item_id, _score) in enumerate(ranking, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    return scores


def fuse_candidates(
    voices: dict[str, list[_RecallSignals]],
    config: MemoryConfig,
    *,
    k: int = _RRF_K,
) -> list[_RecallSignals]:
    """Blend per-voice candidate orderings into one ranking.

    ``voices`` maps voice name to a list of candidates already sorted by that
    voice's own metric.  The function returns candidates re-ordered by RRF
    score, with each candidate's ``score`` field updated to the fused value.

    Supported voices are configured by the caller; the function is agnostic to
    how each voice is produced.
    """
    rankings: list[list[tuple[str, float]]] = []
    candidate_map: dict[str, _RecallSignals] = {}

    for voice_candidates in voices.values():
        ranking: list[tuple[str, float]] = []
        for c in voice_candidates:
            item_id = getattr(c.item, "id", "")
            if not item_id:
                continue
            candidate_map[item_id] = c
            ranking.append((item_id, c.score))
        if ranking:
            rankings.append(ranking)

    if not rankings:
        return []

    rrf_scores = reciprocal_rank_fusion(rankings, k=k)
    # Tie-break by the highest per-voice score among the candidates.
    scored = []
    for item_id, rrf in rrf_scores.items():
        c = candidate_map[item_id]
        c.score = rrf
        scored.append(c)

    scored.sort(key=lambda c: c.score, reverse=True)
    return scored
