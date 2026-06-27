"""Tests for the memory lifecycle: Episode → Fact → Journal → Decay → Prune."""

import asyncio
import time
import pytest

from memlife import MemoryStore, DummyEmbedder, MemoryConfig


@pytest.mark.asyncio
async def test_episode_to_fact(store):
    """An episode can be stored, then a fact extracted from it."""
    ep_id = store.remember(
        task="User said they prefer pytest",
        outcome="success",
    )
    fact_id = await store.store_fact("User prefers pytest", confidence=0.8)
    assert ep_id.startswith("ep_")
    assert fact_id.startswith("fact_")
    # Both are retrievable
    eps = store.recent(limit=5)
    facts = await store.recall_facts("pytest", limit=5)
    assert any("pytest" in e.task for e in eps)
    assert any("pytest" in f.content for f in facts)


@pytest.mark.asyncio
async def test_fact_supersession(store):
    """A fact can be revised, superseding the old one."""
    old_id = await store.store_fact("User uses unittest", confidence=0.6)
    new_id = await store.revise_fact(old_id, "User uses pytest", confidence=0.8)
    assert old_id != new_id
    # Old fact is superseded
    old = store.fact_by_id(old_id)
    assert old.superseded_by == new_id
    # New fact is active
    new = store.fact_by_id(new_id)
    assert new.superseded_by == ""
    assert "pytest" in new.content


@pytest.mark.asyncio
async def test_fact_confidence_decay(store):
    """Facts don't decay — only journal entries do."""
    # Facts maintain their confidence; only journal decays.
    fact_id = await store.store_fact("Test fact", confidence=0.8)
    fact = store.fact_by_id(fact_id)
    assert abs(fact.confidence - 0.8) < 0.01


@pytest.mark.asyncio
async def test_journal_decay(store):
    """Journal entries have effective confidence that decays over time."""
    from memlife.models import JournalEntry
    import math
    
    jid = store.add_journal_entry(
        "observation", "Test observation", confidence=0.8,
    )
    entries = store.journal_recent(limit=5)
    assert len(entries) >= 1
    j = entries[0]
    # Fresh entry should have near-full effective confidence
    eff = j.effective_confidence(halflife_days=30.0, floor=0.15)
    assert eff > 0.7  # Close to 0.8 since it's fresh


@pytest.mark.asyncio
async def test_full_lifecycle(store, config):
    """Full lifecycle: episode → fact → reflection → GC."""
    from memlife import Reflector, DummyChat, run_gc
    
    # 1. Store an episode
    ep_id = store.remember(
        task="User mentioned they switched to vim",
        outcome="success",
    )
    
    # 2. Store a fact
    fact_id = await store.store_fact("User uses vim", confidence=0.7)
    
    # 3. Run reflection (with DummyChat)
    reflector = Reflector(
        memory=store,
        model_chat=DummyChat(),
        critic=False,  # DummyChat doesn't produce critic-ready output
    )
    result = await reflector.reflect()
    assert len(result.episode_ids) >= 1
    
    # 4. Check journal entries were created
    entries = store.journal_recent(limit=5)
    assert len(entries) >= 1
    
    # 5. Run GC (should not prune active data)
    gc_result = run_gc(store)
    assert gc_result["total_pruned"] == 0  # Nothing old enough
    
    # 6. Verify data is still intact
    facts = await store.recall_facts("vim", limit=5)
    assert any("vim" in f.content for f in facts)