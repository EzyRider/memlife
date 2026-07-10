"""Small shared helpers used across mixins.

Extracted from store.py as part of the mixin refactor.
"""

from __future__ import annotations

import json


def _parse_annotations(raw: str | None) -> list[str]:
    """Safely parse a JSON annotations column into a list of labels."""
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    return []
