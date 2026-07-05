"""Tests for MV2-I004 polyphonic recall / RRF fusion."""

import pytest

from memlife import MemoryConfig, DummyEmbedder, polyphonic
from memlife.models import Fact
from memlife.retrieval import _RecallSignals


def _fact(id: str, content: str, score: float) -> _RecallSignals:
    f = Fact(id=id, content=content)
    return _RecallSignals(
        item=f, kind="fact", labelled_text=content, dedup_text=content,
        fact_id=id, vector_sim=0.0, text_score=0.0, source_weight=0.9,
        veracity=0.5, confidence=0.8, recency=1.0, relevance=0.5,
        score=score,
    )


def test_reciprocal_rank_fusion_basic():
    rankings = [
        [("a", 1.0), ("b", 0.8), ("c", 0.6)],
        [("b", 1.0), ("c", 0.7), ("a", 0.5)],
    ]
    scores = polyphonic.reciprocal_rank_fusion(rankings, k=60)
    # b is ranked 2nd and 1st; a is 1st and 3rd; c is 3rd and 2nd.
    # b should win.
    assert scores["b"] > scores["a"]
    assert scores["b"] > scores["c"]


def test_fuse_candidates_reorders_by_rrf():
    cfg = MemoryConfig()
    voices = {
        "vector": [_fact("a", "alpha", 0.9), _fact("b", "beta", 0.8)],
        "text": [_fact("b", "beta", 0.95), _fact("a", "alpha", 0.85)],
    }
    fused = polyphonic.fuse_candidates(voices, cfg)
    ids = [getattr(c.item, "id") for c in fused]
    assert len(ids) == 2
    assert fused[0].score >= fused[1].score


def test_fuse_candidates_empty_voices():
    cfg = MemoryConfig()
    assert polyphonic.fuse_candidates({}, cfg) == []


def test_fuse_candidates_single_voice_is_identity():
    cfg = MemoryConfig()
    voices = {"text": [_fact("x", "xray", 0.5), _fact("y", "yankee", 0.4)]}
    fused = polyphonic.fuse_candidates(voices, cfg)
    assert [getattr(c.item, "id") for c in fused] == ["x", "y"]


@pytest.mark.asyncio
async def test_polyphonic_config_flag_defaults_off(tmp_path):
    db = tmp_path / "poly.db"
    from memlife import MemoryStore

    cfg = MemoryConfig(db_path=str(db))
    store = MemoryStore(config=cfg, embedder=DummyEmbedder())
    assert store.config.use_polyphonic_recall is False
    store.conn.close()
