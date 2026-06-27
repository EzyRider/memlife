"""Vector math and scoring utilities."""

from __future__ import annotations

import math
import time


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def recency_weight(created_at: float, halflife_days: float = 14.0) -> float:
    """Exponential decay weight based on age. 1.0 for now, 0.5 at halflife."""
    age_days = max(0.0, (time.time() - created_at) / 86400.0)
    return math.pow(0.5, age_days / max(1e-6, halflife_days))