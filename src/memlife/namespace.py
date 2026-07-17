"""Namespace isolation helpers."""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path


logger = logging.getLogger(__name__)


class NamespaceError(ValueError):
    """Raised when a namespace identifier is invalid or unsafe."""


_NAMESPACE_RE = re.compile(r"^[a-zA-Z0-9_-]+$")
_MAX_NAMESPACE_LEN = 64
_FORBIDDEN = ("/", "\\", "..", "\x00")
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")

# Reserved namespace names that cannot be used for user namespaces.
_RESERVED_NAMESPACES = frozenset({"_default"})

# Known cloud-sync / network-share roots that fight SQLite WAL. The warning is
# advisory; we do not block the path because some users deliberately run on a
# network share and accept the trade-off.
_CLOUD_SYNC_PATHS = (
    "onedrive",
    "dropbox",
    "google drive",
    "icloud",
    "box",
    "nextcloud",
    "owncloud",
    "syncthing",
)


def validate_namespace(name: str | None) -> str:
    """Return a safe, canonical namespace string.

    Raises NamespaceError if the name is empty, too long, contains path
    separators, control characters, '..' or any character outside the
    allowed set.

    Namespaces are normalized to lowercase so that identifiers like ``"Julie"``
    and ``"julie"`` map to the same directory on case-sensitive filesystems,
    matching the behaviour on case-insensitive filesystems (macOS, Windows).
    """
    if name is None:
        raise NamespaceError("namespace cannot be None")
    name = name.strip().lower()
    if not name:
        raise NamespaceError("namespace cannot be empty")
    if len(name) > _MAX_NAMESPACE_LEN:
        raise NamespaceError(
            f"namespace too long ({len(name)} > {_MAX_NAMESPACE_LEN})"
        )
    for bad in _FORBIDDEN:
        if bad in name:
            raise NamespaceError(
                f"namespace contains forbidden sequence: {bad!r}"
            )
    if _CONTROL_CHARS.search(name):
        raise NamespaceError("namespace contains control characters")
    if name in (".", ".."):
        raise NamespaceError("namespace cannot be '.' or '..'")
    if not _NAMESPACE_RE.match(name):
        raise NamespaceError("namespace must match ^[a-zA-Z0-9_-]+$")
    return name


def list_namespaces(data_dir: str | Path) -> list[str]:
    """Return the names of existing namespace directories under data_dir.

    Directories are validated and normalized to lowercase. On case-insensitive
    filesystems (Windows, macOS) two directories that differ only in case would
    resolve to the same namespace database; in that case we keep the lowercase
    canonical name and warn/ignore the mixed-case duplicate.
    """
    root = Path(data_dir)
    if not root.is_dir():
        return []
    seen: set[str] = set()
    result: list[str] = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        name = p.name
        if not _NAMESPACE_RE.match(name):
            continue
        canonical = name.lower()
        if canonical in seen:
            # On case-insensitive filesystems this is a collision. Warn once.
            logger.warning(
                "ignoring mixed-case namespace directory %r: maps to already-seen %r",
                name,
                canonical,
            )
            continue
        seen.add(canonical)
        result.append(canonical)
    return sorted(result)


def warn_if_cloud_sync_path(path: str | Path) -> None:
    """Log a warning if path lies under a known cloud-sync folder.

    SQLite WAL mode keeps ``-wal`` and ``-shm`` sidecar files next to the
    database that are constantly rewritten. Cloud-sync clients and real-time
    antivirus scanners can lock or corrupt those files, causing "database is
    locked" or checksum errors. The warning is advisory only.
    """
    p = Path(path).resolve()
    lowered_parts = [part.lower() for part in p.parts]
    for marker in _CLOUD_SYNC_PATHS:
        if marker in lowered_parts:
            logger.warning(
                "data_dir %s appears to be under a cloud-sync folder (%s). "
                "SQLite WAL sidecar files may be locked or corrupted by sync "
                "clients. Use a local, non-synced directory, or set "
                "sqlite_journal_mode='DELETE' to disable WAL mode.",
                p,
                marker,
            )
            break


def is_windows_case_insensitive_filesystem() -> bool:
    """Return True when running on a case-insensitive-by-default filesystem.

    This is a heuristic: we treat Windows and macOS as case-insensitive for
    namespace directory purposes. Linux is assumed case-sensitive. The function
    is conservative: when in doubt it returns True.
    """
    return sys.platform in ("win32", "cygwin", "darwin")
