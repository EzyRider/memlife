"""Tests for reflection audit and user correction APIs."""

from __future__ import annotations

import pytest

from memlife import MemoryConfig, MemoryStore
from memlife.reflection import ReflectionPass


pytestmark = pytest.mark.anyio


@pytest.fixture
def store(tmp_path):
    return MemoryStore(
        config=MemoryConfig(
            db_path=str(tmp_path / "mem.db"),
            reflection_pass_retention_count=5,
            reflection_pass_retention_days=0,
        )
    )


async def test_record_and_audit_reflection_pass(store):
    pass_obj = ReflectionPass(
        id="rp_001",
        created_at=1.0,
        episode_ids=["ep_1"],
        proposed=[{"content": "test", "kind": "observation"}],
        kept=[{"content": "test", "kind": "observation"}],
        dropped=[],
        model_used="model-a",
        critic_model_used="model-b",
        total_timeout=300.0,
        elapsed_seconds=1.2,
    )
    pid = store.record_reflection_pass(pass_obj)
    assert pid == "rp_001"

    audit = store.reflection_audit(limit=10)
    assert len(audit) == 1
    assert audit[0]["id"] == "rp_001"
    assert audit[0]["episode_ids"] == ["ep_1"]
    assert audit[0]["kept"] == [{"content": "test", "kind": "observation"}]
    assert audit[0]["model_used"] == "model-a"
    assert audit[0]["critic_model_used"] == "model-b"


async def test_last_reflection_pass_returns_most_recent(store):
    for i in range(3):
        store.record_reflection_pass(
            ReflectionPass(
                id=f"rp_{i:03d}",
                created_at=float(i + 1),
                episode_ids=[],
                proposed=[],
                kept=[],
                dropped=[],
                model_used="m",
                critic_model_used=None,
                total_timeout=0.0,
                elapsed_seconds=0.0,
            )
        )
    last = store.last_reflection_pass()
    assert last is not None
    assert last["id"] == "rp_002"


async def test_reflection_pass_retention_count(store):
    for i in range(10):
        store.record_reflection_pass(
            ReflectionPass(
                id=f"rp_{i:03d}",
                created_at=float(i + 1),
                episode_ids=[],
                proposed=[],
                kept=[],
                dropped=[],
                model_used="m",
                critic_model_used=None,
                total_timeout=0.0,
                elapsed_seconds=0.0,
            )
        )
    audit = store.reflection_audit(limit=100)
    assert len(audit) == 5
    assert audit[0]["id"] == "rp_009"


async def test_user_correction_supersedes_journal(store):
    jid = store.add_journal_entry("observation", "the sky is green", 0.7)
    correction_id = store.add_user_correction(
        jid, "the sky is blue", confidence=0.95
    )

    assert correction_id.startswith("jrn_")
    rows = store.conn.execute(
        "SELECT id, type, content, confidence, source_episodes_json, "
        "private, created_at, superseded_by, embedding_json, last_detected, "
        "annotations_json, links_json FROM journal WHERE id = ?",
        (correction_id,),
    ).fetchall()
    assert len(rows) == 1
    entry = store._journal_from_row(rows[0])
    assert entry.type == "user_correction"
    assert entry.content == "the sky is blue"

    target_rows = store.conn.execute(
        "SELECT superseded_by FROM journal WHERE id = ?", (jid,)
    ).fetchall()
    assert target_rows[0][0] == correction_id

    corrections = store.recent_user_corrections(limit=10)
    assert len(corrections) == 1
    assert corrections[0].id == correction_id


async def test_user_correction_requires_non_empty_content(store):
    jid = store.add_journal_entry("observation", "x", 0.5)
    with pytest.raises(ValueError):
        store.add_user_correction(jid, "   ")


async def test_search_journal_includes_user_corrections(store):
    jid = store.add_journal_entry("observation", "openclaw runs on rust", 0.7)
    store.add_user_correction(
        jid, "openclaw runs on python", confidence=0.95
    )
    results = store.search_journal("python", limit=10)
    assert any(j.type == "user_correction" for j in results)
