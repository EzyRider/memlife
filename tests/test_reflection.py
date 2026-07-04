"""Tests for the reflection loop."""

import pytest

from memlife import Reflector, DummyChat


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