"""Binary vector compression (MV2-I002).

Compresses float32 embeddings into compact bit-packed representations.
Each dimension becomes one bit (positive = 1, negative = 0), giving a
~32x storage reduction.  Hamming distance is used for similarity.
"""

from __future__ import annotations


def binarize(vec: list[float]) -> bytes:
    """Convert a float vector into a compact bit-packed byte string."""
    if not vec:
        return b""
    out = bytearray()
    byte = 0
    for i, val in enumerate(vec):
        if val >= 0:
            byte |= 1 << (7 - (i % 8))
        if (i + 1) % 8 == 0:
            out.append(byte)
            byte = 0
    if len(vec) % 8:
        out.append(byte)
    return bytes(out)


def debinarize(data: bytes, dim: int) -> list[float]:
    """Reconstruct a float vector from a bit-packed byte string.

    Each bit expands to +1.0 or -1.0; this is only useful for distance
    comparisons, not for downstream math.
    """
    if not data or dim <= 0:
        return []
    vec = []
    for i in range(dim):
        byte_idx = i // 8
        bit_idx = 7 - (i % 8)
        if byte_idx >= len(data):
            break
        bit = (data[byte_idx] >> bit_idx) & 1
        vec.append(1.0 if bit else -1.0)
    return vec


def hamming_distance(a: bytes, b: bytes) -> int:
    """Count of differing bits between two packed vectors."""
    n = min(len(a), len(b))
    dist = 0
    for i in range(n):
        dist += (a[i] ^ b[i]).bit_count()
    # If lengths differ, count the extra bits as maximum disagreement.
    longer = max(len(a), len(b))
    dist += (longer - n) * 8
    return dist


def hamming_similarity(a: bytes, b: bytes, dim: int) -> float:
    """Hamming similarity normalised to [0, 1]. 1.0 = identical."""
    if dim <= 0:
        return 0.0
    dist = hamming_distance(a, b)
    return max(0.0, 1.0 - dist / dim)


def cosine_from_binary(a: bytes, b: bytes, dim: int) -> float:
    """Approximate cosine similarity from binary vectors.

    The expected cosine between two random unit vectors with independent
    binarized dimensions is ``1 - 2 * (hamming_distance / dim)``.
    """
    if dim <= 0:
        return 0.0
    dist = hamming_distance(a, b)
    return max(-1.0, 1.0 - 2.0 * (dist / dim))


def compress_for_storage(vec: list[float]) -> bytes:
    """Alias for binarize; returns bytes suitable for ``embedding_json``."""
    return binarize(vec)


def decompress_for_query(data: bytes, dim: int) -> list[float]:
    """Alias for debinarize; returns float vector for distance math."""
    return debinarize(data, dim)
