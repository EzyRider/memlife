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

    def test_validate_runs_all_checks_when_vector_backend_is_none(self):
        """Regression: default vector_backend=None must not short-circuit validate()."""
        cfg = MemoryConfig(
            vector_backend=None,
            sqlite_busy_timeout_ms=-500,
            recency_halflife_days=-1,
            fact_merge_threshold=5.0,
        )
        with pytest.raises(ValueError):
            cfg.validate()

    def test_validate_accepts_valid_default_config(self):
        """Default config with vector_backend=None should validate cleanly."""
        cfg = MemoryConfig()
        cfg.validate()
        assert cfg.resolved_vector_backend() == "json"


class TestPragmaValidation:
    """HF-003: PRAGMA names and values are validated at config time."""

    @pytest.mark.parametrize("value", ["WAL", "wal", "DELETE", "OFF"])
    def test_valid_journal_mode_passes(self, value):
        cfg = MemoryConfig(sqlite_journal_mode=value)
        cfg.validate()
        assert cfg.sqlite_journal_mode == value

    @pytest.mark.parametrize("value", ["WAL; DROP TABLE facts; --", "", "wal2"])
    def test_invalid_journal_mode_raises(self, value):
        cfg = MemoryConfig(sqlite_journal_mode=value)
        with pytest.raises(ValueError, match="invalid value for PRAGMA journal_mode"):
            cfg.validate()

    def test_validate_pragma_rejects_bad_name(self):
        from memlife.config import MemoryConfig

        with pytest.raises(ValueError, match="unsupported PRAGMA name"):
            MemoryConfig._validate_pragma("journal_mode2", "WAL")

    def test_validate_pragma_accepts_integer_pragmas(self):
        from memlife.config import MemoryConfig

        MemoryConfig._validate_pragma("busy_timeout", 5000)
        MemoryConfig._validate_pragma("cache_size", -2000)
        MemoryConfig._validate_pragma("mmap_size", 0)

    def test_validate_pragma_rejects_string_for_integer_pragma(self):
        from memlife.config import MemoryConfig

        with pytest.raises(ValueError, match="PRAGMA busy_timeout requires an integer"):
            MemoryConfig._validate_pragma("busy_timeout", "5000")

    def test_validate_pragma_accepts_foreign_keys_bool(self):
        from memlife.config import MemoryConfig

        MemoryConfig._validate_pragma("foreign_keys", True)
        MemoryConfig._validate_pragma("foreign_keys", 1)

    def test_validate_pragma_rejects_bad_synchronous(self):
        from memlife.config import MemoryConfig

        with pytest.raises(ValueError, match="invalid value for PRAGMA synchronous"):
            MemoryConfig._validate_pragma("synchronous", "BROKEN")


class TestFromEnv:
    """Environment variable loading must validate before returning."""

    def test_from_env_validates(self, monkeypatch):
        monkeypatch.setenv("MEMLIFE_NAMESPACE", "valid-ns")
        cfg = MemoryConfig.from_env()
        cfg.validate()  # does not raise
        assert cfg.namespace == "valid-ns"

    def test_from_env_rejects_invalid_namespace(self, monkeypatch):
        monkeypatch.setenv("MEMLIFE_NAMESPACE", "../bad")
        with pytest.raises(ValueError):
            MemoryConfig.from_env()

    def test_from_env_rejects_invalid_journal_mode(self, monkeypatch):
        monkeypatch.setenv("MEMLIFE_SQLITE_JOURNAL_MODE", "WAL; DROP TABLE facts; --")
        with pytest.raises(ValueError, match="invalid value for PRAGMA journal_mode"):
            MemoryConfig.from_env()

    def test_from_env_rejects_negative_busy_timeout(self, monkeypatch):
        monkeypatch.setenv("MEMLIFE_SQLITE_BUSY_TIMEOUT_MS", "-1")
        with pytest.raises(ValueError, match="sqlite_busy_timeout_ms must be >= 0"):
            MemoryConfig.from_env()
