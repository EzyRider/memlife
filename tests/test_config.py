"""Tests for MemoryConfig."""

import logging

import pytest

from memlife import MemoryConfig


class TestVectorBackendResolution:
    """Precedence and ambiguity handling for vector backend selection."""

    def test_explicit_vector_backend_wins_over_legacy_flags(self):
        cfg = MemoryConfig(
            vector_backend="json",
            use_sqlite_vec=True,
            use_binary_vectors=True,
        )
        assert cfg.resolved_vector_backend() == "json"

    def test_binary_wins_when_both_legacy_flags_true(self, caplog):
        cfg = MemoryConfig(
            vector_backend=None,
            use_sqlite_vec=True,
            use_binary_vectors=True,
        )
        with caplog.at_level(logging.WARNING):
            backend = cfg.resolved_vector_backend()
        assert backend == "binary"
        assert "both use_binary_vectors and use_sqlite_vec" in caplog.text

    def test_sqlite_vec_used_when_only_legacy_flag_set(self):
        cfg = MemoryConfig(
            vector_backend=None,
            use_sqlite_vec=True,
            use_binary_vectors=False,
        )
        assert cfg.resolved_vector_backend() == "sqlite_vec"

    def test_binary_used_when_only_legacy_flag_set(self):
        cfg = MemoryConfig(
            vector_backend=None,
            use_sqlite_vec=False,
            use_binary_vectors=True,
        )
        assert cfg.resolved_vector_backend() == "binary"

    def test_json_default_when_no_flags_set(self):
        cfg = MemoryConfig(
            vector_backend=None,
            use_sqlite_vec=False,
            use_binary_vectors=False,
        )
        assert cfg.resolved_vector_backend() == "json"

    @pytest.mark.parametrize("value", ["sqlite-vec", "sqlite_vec"])
    def test_sqlite_vec_normalisation(self, value):
        cfg = MemoryConfig(vector_backend=value)
        assert cfg.resolved_vector_backend() == "sqlite_vec"

    def test_unknown_vector_backend_raises(self):
        cfg = MemoryConfig(vector_backend="magic_vec")
        with pytest.raises(ValueError, match="unknown vector_backend"):
            cfg.validate()
        # resolved_vector_backend() does not validate; it just normalises.
        assert cfg.resolved_vector_backend() == "magic_vec"
