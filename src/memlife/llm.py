"""DummyChat — canned LLM output for testing and quickstart.

Returns a fixed reflection result. No external dependencies.
Lets you exercise the full reflection loop without a model.
"""

from __future__ import annotations

import json
import re


class DummyChat:
    """Canned chat responses for the reflection loop.

    Implements the ChatCallable protocol: ``await chat.chat(messages, model)``.
    """

    async def chat(self, messages: list[dict], model: str) -> str:
        # Extract episode IDs from the prompt to use as grounds.
        grounds = []
        for msg in messages:
            content = msg.get("content", "") if isinstance(msg, dict) else ""
            if "ep_" in content:
                grounds = re.findall(r"ep_\w+", content)
                break

        return json.dumps({
            "observations": [
                {
                    "content": "The user mentioned a preference worth remembering",
                    "confidence": 0.7,
                    "grounds": grounds[:1] if grounds else [],
                },
            ],
            "hypotheses": [],
            "revisions": [],
        })