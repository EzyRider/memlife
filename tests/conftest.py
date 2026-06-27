"""Test fixtures for memlife tests."""

from pathlib import Path

import pytest

from memlife import MemoryConfig, MemoryStore, DummyEmbedder


@pytest.fixture
def temp_db(tmp_path):
    """A temporary SQLite database path."""
    db_path = str(tmp_path / "test.db")
    yield db_path
    for ext in ("", "-wal", "-shm"):
        p = Path(db_path + ext)
        if p.exists():
            p.unlink()


@pytest.fixture
def config(temp_db):
    """A MemoryConfig pointed at the temp DB."""
    return MemoryConfig(db_path=temp_db)


@pytest.fixture
def store(config):
    """A MemoryStore with DummyEmbedder."""
    s = MemoryStore(config=config, embedder=DummyEmbedder())
    s.embedding_model_name = "dummy"
    yield s
    s.close()