"""Tests for namespace isolation and db_path resolution."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from memlife import MemoryConfig, MemoryStore


pytestmark = pytest.mark.anyio


class TestNamespaceResolution:
    """db_path is computed from data_dir + namespace when not explicit."""

    def test_default_namespace(self):
        config = MemoryConfig()
        store = MemoryStore(config=config)
        assert store.db_path.endswith(
            os.path.join("memlife_data", "_default", "memlife.db")
        )

    def test_custom_namespace(self):
        config = MemoryConfig(namespace="openclaw")
        store = MemoryStore(config=config)
        assert store.db_path.endswith(
            os.path.join("memlife_data", "openclaw", "memlife.db")
        )

    def test_explicit_db_path_overrides_namespace(self):
        with tempfile.TemporaryDirectory() as tmp:
            explicit = os.path.join(tmp, "custom.db")
            config = MemoryConfig(db_path=explicit, namespace="ignored")
            store = MemoryStore(config=config)
            assert store.db_path == explicit

    def test_data_dir_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["MEMLIFE_DATA_DIR"] = tmp
            os.environ["MEMLIFE_NAMESPACE"] = "zeroclaw"
            try:
                config = MemoryConfig.from_env()
                store = MemoryStore(config=config)
                assert store.db_path == os.path.join(tmp, "zeroclaw", "memlife.db")
            finally:
                os.environ.pop("MEMLIFE_DATA_DIR", None)
                os.environ.pop("MEMLIFE_NAMESPACE", None)


class TestNamespaceIsolation:
    """Facts written to one namespace do not leak into another."""

    @pytest.fixture
    def tmp_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            yield tmp

    async def test_facts_isolated_by_namespace(self, tmp_root):
        store_a = MemoryStore(
            config=MemoryConfig(data_dir=tmp_root, namespace="a")
        )
        store_b = MemoryStore(
            config=MemoryConfig(data_dir=tmp_root, namespace="b")
        )

        await store_a.store_fact("namespace a only", source="test")

        facts_a = await store_a.recall_facts("namespace a only")
        facts_b = await store_b.recall_facts("namespace a only")

        assert len(facts_a) == 1
        assert facts_a[0].content == "namespace a only"
        assert len(facts_b) == 0

    def test_db_files_are_distinct(self, tmp_root):
        store_a = MemoryStore(
            config=MemoryConfig(data_dir=tmp_root, namespace="a")
        )
        store_b = MemoryStore(
            config=MemoryConfig(data_dir=tmp_root, namespace="b")
        )
        assert store_a.db_path != store_b.db_path
        assert Path(store_a.db_path).parent != Path(store_b.db_path).parent

    async def test_default_namespace_isolated(self, tmp_root):
        default_store = MemoryStore(
            config=MemoryConfig(data_dir=tmp_root, namespace="_default")
        )
        other_store = MemoryStore(
            config=MemoryConfig(data_dir=tmp_root, namespace="other")
        )
        await default_store.store_fact("default fact", source="test")
        assert len(await other_store.recall_facts("default fact")) == 0
