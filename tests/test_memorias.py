"""Tests for MV2-I003 structured MEMORIA extraction."""

import pytest

from memlife import MemoryConfig, MemoryStore, DummyEmbedder, memorias


@pytest.mark.asyncio
async def test_extract_and_persist_facts(tmp_path):
    db = tmp_path / "memoria.db"
    cfg = MemoryConfig(db_path=str(db), memorias_extraction=True)
    store = MemoryStore(config=cfg, embedder=DummyEmbedder())

    text = """
    Fact: James drives the Melbourne-Brisbane route.
    Preference: James prefers concise answers.
    Instruction: Always check traffic before planning a route.
    Timeline event: 2025-12-25 — Christmas delivery completed.
    KG triple: James → drives → Melbourne-Brisbane
    """
    result = await memorias.persist_extraction(store, text)

    assert len(result["facts"]) == 1
    assert len(result["preferences"]) == 1
    assert len(result["instructions"]) == 1
    assert len(result["timelines"]) == 1
    assert len(result["kg_triples"]) == 1

    # Verify annotations
    pref = store.fact_by_id(result["preferences"][0][0])
    assert pref is not None
    assert "preference" in pref.annotations

    instr = store.fact_by_id(result["instructions"][0][0])
    assert instr is not None
    assert "instruction" in instr.annotations

    store.conn.close()


def test_extract_from_text_empty():
    assert memorias.extract_from_text("") == {
        "facts": [],
        "preferences": [],
        "instructions": [],
        "timelines": [],
        "kg_triples": [],
    }


def test_extract_clean_collapse():
    text = "Fact: Line one\n  line two\nFact: another fact"
    extracted = memorias.extract_from_text(text)
    assert extracted["facts"][0] == "Line one line two"
    assert extracted["facts"][1] == "another fact"


def test_kg_triple_parsing():
    text = "KG triple: James → works_in → Pakenham"
    extracted = memorias.extract_from_text(text)
    assert extracted["kg_triples"] == [("James", "works_in", "Pakenham")]


@pytest.mark.asyncio
async def test_memorias_disabled_config_does_nothing(tmp_path):
    db = tmp_path / "memoria_off.db"
    cfg = MemoryConfig(db_path=str(db), memorias_extraction=False)
    store = MemoryStore(config=cfg, embedder=DummyEmbedder())

    text = "Fact: James drives trucks."
    result = await memorias.persist_extraction(store, text)
    assert len(result["facts"]) == 0
    store.conn.close()
