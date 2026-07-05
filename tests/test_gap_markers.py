"""Tests for MV2-008 temporal gap markers."""

import time

import pytest

from memlife import MemoryConfig, MemoryStore


@pytest.fixture
def gap_store(tmp_path):
    db = tmp_path / "gaps.db"
    cfg = MemoryConfig(db_path=str(db), gap_marker_threshold_hours=0.0001)
    store = MemoryStore(cfg)
    yield store
    store.close()


@pytest.fixture
def no_gap_store(tmp_path):
    db = tmp_path / "no_gaps.db"
    cfg = MemoryConfig(db_path=str(db), gap_marker_threshold_hours=0.0)
    store = MemoryStore(cfg)
    yield store
    store.close()


def test_gap_marker_inserted_after_long_silence(gap_store):
    """A synthetic episode appears when the gap exceeds the threshold."""
    ep1 = gap_store.remember("task 1", "ok")
    time.sleep(0.5)  # 0.5h > 0.1h threshold
    ep2 = gap_store.remember("task 2", "ok")

    recent = gap_store.recent(limit=10)
    ids = [e.id for e in recent]
    assert ep1 in ids
    assert ep2 in ids
    assert any(e.is_gap_marker for e in recent), "expected a gap marker episode"


def test_no_gap_marker_when_threshold_zero(no_gap_store):
    """Disabling the threshold prevents synthetic episodes."""
    ep1 = no_gap_store.remember("task 1", "ok")
    time.sleep(0.5)
    ep2 = no_gap_store.remember("task 2", "ok")

    recent = no_gap_store.recent(limit=10)
    assert not any(e.is_gap_marker for e in recent)
    assert len(recent) == 2


def test_gap_marker_not_inserted_within_threshold(gap_store):
    """No marker is created if the silence is shorter than the threshold."""
    ep1 = gap_store.remember("task 1", "ok")
    time.sleep(0.05)  # 0.05h < 0.1h threshold
    ep2 = gap_store.remember("task 2", "ok")

    recent = gap_store.recent(limit=10)
    assert not any(e.is_gap_marker for e in recent)
    assert len(recent) == 2


def test_gap_marker_label_format(gap_store):
    """The gap marker task describes the elapsed time."""
    gap_store.remember("task 1", "ok")
    time.sleep(0.5)
    gap_store.remember("task 2", "ok")

    recent = gap_store.recent(limit=10)
    marker = next((e for e in recent if e.is_gap_marker), None)
    assert marker is not None
    assert marker.task.startswith("[gap:")
    assert "passed" in marker.task


def test_gap_marker_does_not_trigger_its_own_gap(gap_store):
    """The synthetic marker itself should not cause another marker on next call."""
    gap_store.remember("task 1", "ok")
    time.sleep(0.5)
    gap_store.remember("task 2", "ok")
    recent_before = gap_store.recent(limit=10)
    marker_count_before = sum(1 for e in recent_before if e.is_gap_marker)

    # The marker was inserted slightly after ep1; the next immediate call is within threshold.
    time.sleep(0.05)
    gap_store.remember("task 3", "ok")
    recent_after = gap_store.recent(limit=10)
    marker_count_after = sum(1 for e in recent_after if e.is_gap_marker)
    assert marker_count_after == marker_count_before


def test_gap_marker_uses_correct_timestamp_order(gap_store):
    """The marker sits between the two real episodes in time order."""
    ep1 = gap_store.remember("task 1", "ok")
    time.sleep(0.5)
    ep2 = gap_store.remember("task 2", "ok")

    recent = gap_store.recent(limit=10)
    marker = next(e for e in recent if e.is_gap_marker)
    ep1_obj = next(e for e in recent if e.id == ep1)
    ep2_obj = next(e for e in recent if e.id == ep2)
    assert ep1_obj.created_at < marker.created_at
    assert marker.created_at < ep2_obj.created_at
