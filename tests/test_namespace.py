"""Tests for namespace isolation helpers."""

from __future__ import annotations

import pytest

from memlife import (
    DummyEmbedder,
    MemoryConfig,
    MemoryStore,
    NamespaceError,
    list_namespaces,
    validate_namespace,
)
from memlife.namespace import warn_if_cloud_sync_path


def test_valid_namespace():
    assert validate_namespace("ingrid") == "ingrid"
    assert validate_namespace("  openclaw  ") == "openclaw"
    assert validate_namespace("user_42-X") == "user_42-x"


def test_namespace_case_normalized():
    assert validate_namespace("Julie") == "julie"
    assert validate_namespace("InGrid") == "ingrid"
    assert validate_namespace("USER_42") == "user_42"


def test_invalid_namespaces():
    for bad in [
        "",
        "   ",
        ".",
        "..",
        "a/../b",
        "a/b",
        "a\\b",
        "user\x00",
        "user\x01",
        "user name",
        "user\x7f",
    ]:
        with pytest.raises(NamespaceError):
            validate_namespace(bad)


def test_namespace_path_traversal_blocked(tmp_path):
    cfg = MemoryConfig(
        data_dir=str(tmp_path), namespace="../../../../tmp/escaped"
    )
    with pytest.raises(NamespaceError):
        MemoryStore(config=cfg)


def test_list_namespaces(tmp_path):
    (tmp_path / "ingrid").mkdir()
    (tmp_path / "openclaw").mkdir()
    (tmp_path / "bad.dir").mkdir()  # should be ignored
    assert list_namespaces(tmp_path) == ["ingrid", "openclaw"]


def test_list_namespaces_missing_dir(tmp_path):
    missing = tmp_path / "does_not_exist"
    assert list_namespaces(missing) == []


def test_list_namespaces_case_collision(tmp_path, caplog):
    """Mixed-case directories that map to the same namespace are deduped."""
    (tmp_path / "Ingrid").mkdir()
    (tmp_path / "ingrid").mkdir()
    with caplog.at_level("WARNING", logger="memlife.namespace"):
        result = list_namespaces(tmp_path)
    assert result == ["ingrid"]
    assert any("mixed-case namespace directory" in r.message for r in caplog.records)


def test_warn_if_cloud_sync_path(tmp_path, caplog):
    """A data_dir under a known cloud-sync folder triggers a warning."""
    sync_root = tmp_path / "OneDrive" / "Documents"
    sync_root.mkdir(parents=True)
    with caplog.at_level("WARNING", logger="memlife.namespace"):
        warn_if_cloud_sync_path(sync_root)
    assert any("cloud-sync folder" in r.message for r in caplog.records)


def test_cloud_sync_warning_on_store_init(tmp_path, caplog):
    """MemoryStore warns when data_dir resolves under a cloud-sync folder."""
    data_dir = tmp_path / "Dropbox" / "memlife"
    data_dir.mkdir(parents=True)
    with caplog.at_level("WARNING", logger="memlife.namespace"):
        store = MemoryStore(config=MemoryConfig(data_dir=str(data_dir)))
    try:
        assert any("cloud-sync folder" in r.message for r in caplog.records)
    finally:
        store.close()


def test_switch_namespace(tmp_path):
    s1 = MemoryStore(
        config=MemoryConfig(data_dir=str(tmp_path), namespace="a"),
        embedder=DummyEmbedder(),
    )
    s2 = s1.switch_namespace("b")
    assert s2.config.namespace == "b"
    assert s2.db_path.endswith(f"{tmp_path}/b/memlife.db")
    assert s2.embedder is s1.embedder
    s1.close()
    s2.close()


def test_switch_namespace_rejects_invalid(tmp_path):
    s1 = MemoryStore(
        config=MemoryConfig(data_dir=str(tmp_path), namespace="a"),
        embedder=DummyEmbedder(),
    )
    with pytest.raises(NamespaceError):
        s1.switch_namespace("a/b")
    s1.close()


def test_switch_namespace_rejects_model_mismatch(tmp_path):
    s1 = MemoryStore(
        config=MemoryConfig(
            data_dir=str(tmp_path), namespace="a", embedding_model="model-a"
        ),
        embedder=DummyEmbedder(),
    )
    s2 = s1.switch_namespace("b")
    assert s2.config.embedding_model == "model-a"
    s1.close()
    s2.close()
