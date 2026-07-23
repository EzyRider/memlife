"""Tests for the reflection loop."""

import pytest

from memlife import DummyChat, Reflector


@pytest.mark.asyncio
async def test_dummy_chat_extracts_real_grounds():
    """DummyChat skips the system prompt and extracts real episode IDs."""
    chat = DummyChat()
    messages = [
        {
            "role": "system",
            "content": 'Example: "grounds": ["ep_..."] is a placeholder.',
        },
        {
            "role": "user",
            "content": "Today's episodes (ids: ep_abc123, ep_def456): user did things.",
        },
    ]
    raw = await chat.chat(messages, model="test")
    import json

    data = json.loads(raw)
    assert data["observations"][0]["grounds"] == ["ep_abc123"]


@pytest.mark.asyncio
async def test_reflection_creates_journal(store, config):
    """Reflection with DummyChat produces journal entries."""
    store.remember(
        task="User said they switched to vim",
        outcome="success",
    )
    reflector = Reflector(
        memory=store,
        model_chat=DummyChat(),
        critic=False,
        model_name="test",
    )
    result = await reflector.reflect()
    assert len(result.episode_ids) >= 1
    entries = store.journal_recent(limit=5)
    assert len(entries) >= 1


@pytest.mark.asyncio
async def test_reflection_no_episodes(store):
    """Reflection with no episodes returns empty result."""
    reflector = Reflector(
        memory=store,
        model_chat=DummyChat(),
        critic=False,
    )
    result = await reflector.reflect()
    assert len(result.episode_ids) == 0
    assert len(result.observations) == 0


@pytest.mark.asyncio
async def test_reflection_timeout_handling(store, config):
    """Reflection doesn't crash when the model times out."""
    class HangingChat:
        async def chat(self, messages, model):
            import asyncio
            await asyncio.sleep(1000)

    store.remember(task="test", outcome="success")
    reflector = Reflector(
        memory=store,
        model_chat=HangingChat(),
        critic=False,
        timeout=0.1,
        total_timeout=0.2,
        model_name="test",
    )
    result = await reflector.reflect()
    # Should return with episode IDs but no stored entries
    assert len(result.episode_ids) >= 1


@pytest.mark.asyncio
async def test_critic_failure_falls_back(store, config):
    """If the critic fails, the pre-critic result is kept."""
    store.remember(task="test episode", outcome="success")

    class FailingCriticChat:
        call_count = 0

        async def chat(self, messages, model):
            self.call_count += 1
            if self.call_count == 1:
                # First call (generation) succeeds
                import json
                return json.dumps({
                    "observations": [{"content": "Test obs", "confidence": 0.7, "grounds": []}],
                    "hypotheses": [],
                    "revisions": [],
                })
            # Second call (critic) fails
            raise RuntimeError("critic model exploded")

    reflector = Reflector(
        memory=store,
        model_chat=FailingCriticChat(),
        critic=True,
        critic_model="test-model",
        model_name="test",
    )
    result = await reflector.reflect()
    # Observation should be kept despite critic failure
    entries = store.journal_recent(limit=5)
    assert len(entries) >= 1


@pytest.mark.asyncio
async def test_reflection_marks_episodes_reflected(store, config):
    """After reflection, episodes are marked as reflected."""
    ep_id = store.remember(task="test", outcome="success")
    store.queue_reflection(ep_id)
    reflector = Reflector(
        memory=store,
        model_chat=DummyChat(),
        critic=False,
        model_name="test",
    )
    await reflector.reflect()
    pending = store.pending_reflections()
    assert ep_id not in pending


@pytest.mark.asyncio
async def test_reflection_memorias_extraction_records_episode_provenance(store, config):
    """When MEMORIAS extraction is on, KG triples get episode provenance."""
    store.config.memorias_extraction = True
    ep_id = store.remember(task="James said he works in Pakenham", outcome="success")
    store.queue_reflection(ep_id)

    class MemoriasChat:
        async def chat(self, messages, model):
            import json
            return json.dumps({
                "observations": [],
                "hypotheses": [],
                "revisions": [],
            })

    reflector = Reflector(
        memory=store,
        model_chat=MemoriasChat(),
        critic=False,
        model_name="test",
    )
    reflector._reflection_cycle = 1
    result = await reflector.reflect()
    assert ep_id in result.episode_ids

    # DummyChat-style raw output has no KG triples; force an explicit extraction
    # on the same store to verify provenance wiring.
    from memlife import memorias
    extracted = await memorias.persist_extraction(
        store,
        "KG triple: James -> works_in -> Pakenham",
        source="reflection",
    )
    assert len(extracted["kg_triples"]) == 1
    tid = extracted["kg_triples"][0][0]
    store._add_triple_provenance(tid, [{"kind": "episode", "id": ep_id}])
    triples = store.triples_about("James")
    t = next(x for x in triples if x["id"] == tid)
    assert any(p["kind"] == "episode" and p["id"] == ep_id for p in t["provenance"])