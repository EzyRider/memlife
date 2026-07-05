"""Regex-based structured MEMORIA extraction (MV2-I003).

Parses free-form reflection text into categories that map onto memlife's
existing tables:

- facts       → store as facts
- preferences → store as facts with source='user' and annotation 'preference'
- instructions → store as facts with annotation 'instruction'
- timelines   → store as facts with annotation 'timeline'
- kg triples  → store as temporal triples anchored to a synthetic fact

No LLM is required.  When ``memorias_extraction`` is enabled in config, the
reflection pipeline can call :func:`persist_extraction` and persist the
results.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memlife.store import MemoryStore


# Conservative line-based patterns.  Each item ends when another labelled item
# or end-of-string is reached.
_ITEM_END = r"(?=\s*(?:[-*]\s*)?(?:fact|known fact|remember that|preference|prefers?|user prefers?|instruction|timeline|event|kg triple|triple|relationship)\s*[:\-]|\Z)"

_FACT_RE = re.compile(
    r"(?:^|\s)(?:[-*]\s*)?(?:fact|known fact|remember that)\s*[:\-]?\s*(.+?)" + _ITEM_END,
    re.IGNORECASE | re.DOTALL,
)

_PREFERENCE_RE = re.compile(
    r"(?:^|\s)(?:[-*]\s*)?(?:preference|prefers?|user prefers?|I prefer)\s*[:\-]?\s*(.+?)" + _ITEM_END,
    re.IGNORECASE | re.DOTALL,
)

_INSTRUCTION_RE = re.compile(
    r"(?:^|\s)(?:[-*]\s*)?(?:instruction|instruct|must always|should always)\s*[:\-]?\s*(.+?)" + _ITEM_END,
    re.IGNORECASE | re.DOTALL,
)

_TIMELINE_RE = re.compile(
    r"(?:^|\s)(?:[-*]\s*)?(?:timeline|event)\s*[:\-]?\s*(.+?)" + _ITEM_END,
    re.IGNORECASE | re.DOTALL,
)

_KG_TRIPLE_RE = re.compile(
    r"(?:^|\s)(?:[-*]\s*)?(?:kg triple|triple|relationship)\s*[:\-]?\s*"
    r"(.+?)\s*(?:→|->|[-=\u2014]+\u003e)\s*(.+?)\s*(?:→|->|[-=\u2014]+\u003e)\s*(.+?)(?=\s*(?:[-*]\s*)?(?:kg triple|triple|relationship)\s*[:\-]|\Z)",
    re.IGNORECASE | re.DOTALL,
)

def _clean(text: str) -> str:
    """Collapse whitespace and strip bullets from extracted items."""
    text = text.strip()
    text = re.sub(r"\n\s+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def extract_from_text(text: str) -> dict:
    """Extract MEMORIA-style structures from raw text.

    Returns a dict with lists: ``facts``, ``preferences``,
    ``instructions``, ``timelines``, ``kg_triples``.
    """
    facts: list[str] = []
    preferences: list[str] = []
    instructions: list[str] = []
    timelines: list[dict] = []
    kg_triples: list[tuple[str, str, str]] = []

    for m in _FACT_RE.finditer(text):
        facts.append(_clean(m.group(1)))
    for m in _PREFERENCE_RE.finditer(text):
        preferences.append(_clean(m.group(1)))
    for m in _INSTRUCTION_RE.finditer(text):
        instructions.append(_clean(m.group(1)))
    for m in _TIMELINE_RE.finditer(text):
        item = _clean(m.group(1))
        # Try to split "when — what" or "when: what"
        when = ""
        what = item
        if " — " in item:
            when, what = item.split(" — ", 1)
        elif ":" in item:
            when, what = item.split(":", 1)
            when, what = when.strip(), what.strip()
        timelines.append({"when": when, "what": what})
    for m in _KG_TRIPLE_RE.finditer(text):
        kg_triples.append(
            (_clean(m.group(1)), _clean(m.group(2)), _clean(m.group(3)))
        )

    return {
        "facts": facts,
        "preferences": preferences,
        "instructions": instructions,
        "timelines": timelines,
        "kg_triples": kg_triples,
    }


async def persist_extraction(
    store: MemoryStore,
    text: str,
    source: str = "reflection",
) -> dict:
    """Extract MEMORIA structures and store them in memlife tables.

    Facts, preferences, instructions, and timelines become facts with
    appropriate annotations.  KG triples become temporal triples anchored to a
    synthetic fact that records the triple text.

    If ``store.config.memorias_extraction`` is False, this returns empty lists
    without writing anything.
    """
    if not store.config.memorias_extraction:
        return {
            "facts": [],
            "preferences": [],
            "instructions": [],
            "timelines": [],
            "kg_triples": [],
        }

    extracted = extract_from_text(text)
    stored = {
        "facts": [],
        "preferences": [],
        "instructions": [],
        "timelines": [],
        "kg_triples": [],
    }

    for item in extracted["facts"]:
        fid = await store.store_fact(item, source=source, confidence=0.75)
        stored["facts"].append((fid, item))

    for item in extracted["preferences"]:
        fid = await store.store_fact(item, source="user", confidence=0.85)
        store.annotate_fact(fid, "preference")
        stored["preferences"].append((fid, item))

    for item in extracted["instructions"]:
        fid = await store.store_fact(item, source=source, confidence=0.8)
        store.annotate_fact(fid, "instruction")
        stored["instructions"].append((fid, item))

    for item in extracted["timelines"]:
        fact = f"Timeline event ({item['when']}): {item['what']}"
        fid = await store.store_fact(fact, source=source, confidence=0.7)
        store.annotate_fact(fid, "timeline")
        stored["timelines"].append((fid, fact))

    for subj, pred, obj in extracted["kg_triples"]:
        fact_text = f"KG triple: {subj} {pred} {obj}"
        fid = await store.store_fact(fact_text, source=source, confidence=0.7)
        tid = store.store_fact_triple(fid, subj, pred, obj, confidence=0.75)
        stored["kg_triples"].append((tid, subj, pred, obj))

    return stored
