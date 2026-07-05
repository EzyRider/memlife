"""Tests for MV2-001 tiered episodic degradation."""

import time

import pytest

from memlife import MemoryConfig, MemoryStore, retrieve
from memlife.vectors import recency_weight


@pytest.fixture
def tiered_store(tmp_path):
    db = tmp_path / "tiered.db"
    cfg = MemoryConfig(
        db_path=str(db),
        recall_vector_weight=0.0,
        recall_text_weight=1.0,
        recall_source_weight=0.0,
        recall_veracity_weight=0.0,
        episode_tool_success_halflife_days=21.0,
        episode_failure_halflife_days=3.0,
        episode_observation_halflife_days=1.0,
    )
    store = MemoryStore(cfg)
    yield store
    store.close()


@pytest.mark.asyncio
async def test_successful_tool_episode_decays_slowest(tiered_store):
    """A successful tool episode keeps more recency than a plain observation."""
    store = tiered_store
    now = time.time()
    age_days = 2.0
    then = now - age_days * 86400

    store.conn.execute(
        "INSERT INTO episodes (id, task, outcome, summary, tool_calls_json, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("ep_tool", "ran query with search tool", "success", "", '[{"tool": "search"}]', then),
    )
    store.conn.execute(
        "INSERT INTO episodes (id, task, outcome, summary, tool_calls_json, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("ep_obs", "ran query manually", "success", "", "[]", then),
    )
    store.conn.commit()

    result = await retrieve(store, "ran query", debug=True)
    assert isinstance(result, dict)
    by_id = {c["id"]: c for c in result["candidates"]}
    assert "ep_tool" in by_id and "ep_obs" in by_id
    assert by_id["ep_tool"]["recency"] > by_id["ep_obs"]["recency"]


@pytest.mark.asyncio
async def test_failure_episode_decays_faster_than_successful_tool(tiered_store):
    """A failed episode fades faster than a successful tool episode."""
    store = tiered_store
    now = time.time()
    age_days = 10.0
    then = now - age_days * 86400

    store.conn.execute(
        "INSERT INTO episodes (id, task, outcome, summary, tool_calls_json, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("ep_fail", "ran query and it failed", "failed", "", "[]", then),
    )
    store.conn.execute(
        "INSERT INTO episodes (id, task, outcome, summary, tool_calls_json, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("ep_tool", "ran query with search tool", "success", "", '[{"tool": "search"}]', then),
    )
    store.conn.commit()

    result = await retrieve(store, "ran query", debug=True)
    assert isinstance(result, dict)
    by_id = {c["id"]: c for c in result["candidates"]}
    assert "ep_fail" in by_id and "ep_tool" in by_id
    assert by_id["ep_fail"]["recency"] < by_id["ep_tool"]["recency"]


@pytest.mark.asyncio
async def test_gap_marker_uses_default_decay(tiered_store):
    """Gap markers keep the default episode decay, not the observation tier."""
    store = tiered_store
    now = time.time()
    age_days = 2.0
    then = now - age_days * 86400

    store.conn.execute(
        "INSERT INTO episodes (id, task, outcome, summary, tool_calls_json, created_at, is_gap_marker) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("gap_1", "[gap: 3 days passed]", "success", "", "[]", then, 1),
    )
    store.conn.execute(
        "INSERT INTO episodes (id, task, outcome, summary, tool_calls_json, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("ep_obs", "ran query", "success", "", "[]", then),
    )
    store.conn.commit()

    result = await retrieve(store, "gap passed", debug=True)
    assert isinstance(result, dict)
    by_id = {c["id"]: c for c in result["candidates"]}
    assert "gap_1" in by_id and "ep_obs" in by_id
    assert by_id["gap_1"]["recency"] > by_id["ep_obs"]["recency"]


def test_episode_helpers():
    """Episode properties classify outcomes and tool use correctly."""
    from memlife.models import Episode

    ok_tool = Episode(
        id="a", task="t", outcome="success",
        tool_calls_json='[{"tool": "x"}]', created_at=1.0,
    )
    assert ok_tool.is_success
    assert ok_tool.has_tool_calls
    assert not ok_tool.is_failure

    fail = Episode(
        id="b", task="t", outcome="failed",
        tool_calls_json="[]", created_at=1.0,
    )
    assert fail.is_failure
    assert not fail.is_success
    assert not fail.has_tool_calls

    obs = Episode(
        id="c", task="t", outcome="OK",
        tool_calls_json="[]", created_at=1.0,
    )
    assert obs.is_success
    assert not obs.has_tool_calls


def test_recency_weight_matches_tiers():
    """Direct recency weights follow the expected tier ordering."""
    age_days = 2.0
    tool = recency_weight(time.time() - age_days * 86400, halflife_days=21.0)
    obs = recency_weight(time.time() - age_days * 86400, halflife_days=1.0)
    fail = recency_weight(time.time() - age_days * 86400, halflife_days=3.0)
    assert tool > fail > obs
