"""Deterministic, zero-LLM entity extraction from free-form text.

The extractor is intentionally conservative. It recognises:

* Capitalised word sequences (proper-noun-like phrases).
* Known technical terms supplied by the caller (allowlist).
* Short uppercase acronyms (2-5 chars).

It blocks a small set of very common English words and stop-words so random
sentence fragments don't become entities.

Output is a list of ``(canonical_name, alias)`` pairs. The canonical name is
lower-cased; the alias preserves the original casing/variant seen in text so
it can be registered in ``entity_aliases``.
"""

from __future__ import annotations

import re


# Default blocklist: common words that are capitalised by accident (start of
# sentence) or are too generic to be useful entities.
DEFAULT_BLOCKLIST: frozenset[str] = frozenset({
    "i", "me", "my", "myself", "we", "our", "ours", "you", "your", "yours",
    "he", "him", "his", "she", "her", "hers", "it", "its", "they", "them",
    "their", "theirs", "this", "that", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "will", "would", "could", "should", "may", "might", "must", "shall",
    "can", "need", "dare", "ought", "used", "a", "an", "the", "and", "but",
    "or", "yet", "so", "if", "because", "although", "though", "while", "where",
    "when", "after", "before", "until", "unless", "since", "although", "however",
    "therefore", "thus", "hence", "moreover", "furthermore", "nevertheless",
    "meanwhile", "otherwise", "instead", "besides", "also", "too", "very",
    "just", "only", "even", "still", "already", "yet", "once", "twice",
    "here", "there", "everywhere", "somewhere", "nowhere", "anywhere",
    "today", "tomorrow", "yesterday", "now", "then", "soon", "later",
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "first", "second", "third", "last", "next", "previous", "such",
    "what", "which", "who", "whom", "whose", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "many", "several",
    "no", "not", "none", "any", "every", "each", "own", "same", "different",
    # Common sentence-start words that are not entities.
    "plan", "build", "create", "make", "use", "work", "run", "test", "fix",
    "add", "update", "remove", "delete", "write", "read", "check", "get",
    "set", "put", "take", "give", "look", "see", "find", "know", "think",
    "say", "tell", "ask", "try", "want", "like", "love", "help", "start",
    "end", "finish", "done", "doing",
})

# Default allowlist: domain terms that may appear lower-case but are still
# meaningful entities for memlife users. Keep it small and project-relevant.
DEFAULT_ALLOWLIST: frozenset[str] = frozenset({
    "memlife", "ingrid", "hermes", "openclaw", "nanobot", "zeroclaw",
    "ollama", "anthropic", "claude", "mcp", "sqlite", "python", "pytest",
    "github", "pypi", "docker", "tailwind", "react", "vite", "nextcloud",
})


def _token_is_acronym(token: str) -> bool:
    """True for short all-caps tokens that look like acronyms."""
    return (
        2 <= len(token) <= 5
        and token.isupper()
        and token.isalpha()
        and token not in DEFAULT_BLOCKLIST
    )


def _normalise_token_sequence(tokens: list[str]) -> str:
    """Return a canonical lower-case name for a token sequence."""
    return " ".join(tokens).strip(".,!?;:\"'()[]{}").lower()


def extract_entities(
    text: str,
    *,
    allowlist: set[str] | frozenset[str] | None = None,
    blocklist: set[str] | frozenset[str] | None = None,
) -> list[tuple[str, str]]:
    """Extract canonical entity mentions from ``text``.

    Returns a list of ``(canonical_name, alias)`` pairs without duplicates.
    ``canonical_name`` is lower-cased; ``alias`` preserves the casing seen in
    the text. If the canonical form and the alias are identical, both are still
    returned so callers can decide whether to store an alias.

    The extractor uses regex heuristics only; no LLM is involved.
    """
    if not text:
        return []

    # Normalise internal whitespace and collapse multiple newlines so the regex
    # cannot bridge across sentence boundaries.
    text = re.sub(r"\s+", " ", text).strip()

    allow = set(allowlist) if allowlist is not None else set(DEFAULT_ALLOWLIST)
    block = set(blocklist) if blocklist is not None else set(DEFAULT_BLOCKLIST)

    results: list[tuple[str, str]] = []
    seen: set[str] = set()

    # 1. Capitalised phrases: one or more Title-Case or ALL-CAPS words.
    #    This catches "James", "Julie", "OpenClaw", "GitHub", "MCP Server".
    for match in re.finditer(
        r"\b[A-Z][a-zA-Z0-9_]*(?:\s+[A-Z][a-zA-Z0-9_]*){0,3}\b", text
    ):
        raw = match.group(0).strip()
        # Drop common words that happen to be capitalised inside the phrase.
        tokens = [
            t for t in raw.split()
            if _normalise_token_sequence([t]) not in block
        ]
        if not tokens:
            continue
        raw = " ".join(tokens)
        canonical = _normalise_token_sequence(tokens)
        if not canonical or canonical in block:
            continue
        # Drop pure-number or single-letter fragments.
        if len(canonical) < 2:
            continue
        # Drop if the first token is a common word capitalised by sentence start.
        first = canonical.split()[0]
        if first in block:
            continue
        if canonical not in seen:
            seen.add(canonical)
            results.append((canonical, raw))

    # 2. Known allowlist terms (case-insensitive) that may appear in plain text.
    lower_text = text.lower()
    for term in allow:
        if term in lower_text and term not in block:
            if term not in seen:
                seen.add(term)
                # Recover original casing from the text for the alias.
                m = re.search(
                    re.escape(term),
                    text,
                    flags=re.IGNORECASE,
                )
                alias = m.group(0) if m else term
                results.append((term, alias))

    # 3. Standalone acronyms not already captured by the capitalised-phrase pass.
    for match in re.finditer(r"\b[A-Z]{2,5}\b", text):
        raw = match.group(0)
        canonical = raw.lower()
        if canonical in block:
            continue
        if canonical in seen:
            continue
        # Skip if it looks like a Roman numeral.
        if re.fullmatch(r"[IVXLCDM]+", raw):
            continue
        seen.add(canonical)
        results.append((canonical, raw))

    return results
