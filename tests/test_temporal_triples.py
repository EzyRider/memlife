"""Tests for MV2-003 temporal triple store."""

import time

import pytest

from memlife import MemoryConfig, MemoryStore


@pytest.fixture
def triple_store(tmp_path):
    db = tmp_path / "triples.db"
    store = MemoryStore(MemoryConfig(db_path=str(db)))
    yield store
    store.close()


def test_store_fact_triple(triple_store):
    """A triple can be attached to a fact and read back."""
    fact_id = "fact_test_001"
    triple_store.conn.execute(
        "INSERT INTO facts (id, content, source, confidence, embedding_json, "
        "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (fact_id, "User lives in Melbourne", "user", 0.9, "", time.time(), time.time()),
    )
    triple_store.conn.commit()

    tid = triple_store.store_fact_triple(
        fact_id, "user", "lives_in", "Melbourne", confidence=0.9,
    )
    assert tid.startswith("triple_")
    obj, conf, _ = triple_store.current_truth("user", "lives_in")
    assert obj == "Melbourne"
    assert conf == pytest.approx(0.9)


def test_current_truth_expires_on_revision(triple_store):
    """When a fact is superseded, its open triples are expired."""
    import asyncio

    fact_id = "fact_old"
    triple_store.conn.execute(
        "INSERT INTO facts (id, content, source, confidence, embedding_json, "
        "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (fact_id, "User lives in Melbourne", "user", 0.9, "", time.time(), time.time()),
    )
    triple_store.conn.commit()

    triple_store.store_fact_triple(fact_id, "user", "lives_in", "Melbourne", confidence=0.9)
    assert triple_store.current_truth("user", "lives_in")[0] == "Melbourne"

    new_id = asyncio.run(triple_store.revise_fact(fact_id, "User lives in Sydney", confidence=0.9))
    assert new_id
    assert new_id != fact_id

    # The old triple is now expired.
    obj, _, _ = triple_store.current_truth("user", "lives_in")
    assert obj is None
    # The new fact has no triples until explicitly attached.
    assert triple_store.triples_for_fact(new_id) == []


def test_truth_as_of(triple_store):
    """truth_as_of returns the object valid at a given timestamp."""
    fact_id = "fact_history"
    triple_store.conn.execute(
        "INSERT INTO facts (id, content, source, confidence, embedding_json, "
        "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (fact_id, "History", "user", 0.9, "", time.time(), time.time()),
    )
    triple_store.conn.commit()

    t1 = time.time() - 1000
    t2 = time.time() - 500
    triple_store.store_fact_triple(
        fact_id, "user", "job", "truck driver", confidence=0.8, valid_from=t1,
    )
    triple_store.store_fact_triple(
        fact_id, "user", "job", "mechanic", confidence=0.8, valid_from=t2,
    )

    obj, _, _ = triple_store.truth_as_of("user", "job", t1 + 10)
    assert obj == "truck driver"
    obj, _, _ = triple_store.truth_as_of("user", "job", t2 + 10)
    assert obj == "mechanic"
    obj, _, _ = triple_store.current_truth("user", "job")
    assert obj == "mechanic"


def test_triples_for_fact(triple_store):
    """triples_for_fact returns all triples for a fact, newest first."""
    fact_id = "fact_multi"
    triple_store.conn.execute(
        "INSERT INTO facts (id, content, source, confidence, embedding_json, "
        "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (fact_id, "Multi", "user", 0.9, "", time.time(), time.time()),
    )
    triple_store.conn.commit()

    triple_store.store_fact_triple(fact_id, "user", "likes", "coffee")
    triple_store.store_fact_triple(fact_id, "user", "likes", "tea")
    triples = triple_store.triples_for_fact(fact_id)
    assert len(triples) == 2
    assert triples[0]["object"] == "tea"
    assert triples[1]["object"] == "coffee"


def test_unknown_truth_returns_none(triple_store):
    """Querying a non-existent triple returns (None, 0.0, None)."""
    obj, conf, tid = triple_store.current_truth("nobody", "nothing")
    assert obj is None
    assert conf == 0.0
    assert tid is None
