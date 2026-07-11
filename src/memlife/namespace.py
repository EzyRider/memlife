"""Namespace isolation helpers."""

from __future__ import annotations

import re
from pathlib import Path


class NamespaceError(ValueError):
    """Raised when a namespace identifier is invalid or unsafe."""


_NAMESPACE_RE = re.compile(r"^[a-zA-Z0-9_-]+$")
_MAX_NAMESPACE_LEN = 64
_FORBIDDEN = ("/", "\\", "..", "\x00")
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")


def validate_namespace(name: str | None) -> str:
    """Return a safe, canonical namespace string.

    Raises NamespaceError if the name is empty, too long, contains path
    separators, control characters, '..' or any character outside the
    allowed set.
    """
    if name is None:
        raise NamespaceError("namespace cannot be None")
    name = name.strip()
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
    """Return the names of existing namespace directories under data_dir."""
    root = Path(data_dir)
    if not root.is_dir():
        return []
    return sorted(
        p.name
        for p in root.iterdir()
        if p.is_dir() and _NAMESPACE_RE.match(p.name)
    )
