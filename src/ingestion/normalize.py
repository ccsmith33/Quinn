"""Plaintext normalization + content hashing (S3.3 / S4.2).

The same normalizer is the source of truth for:
  - `filings.content_hash` (S3.3 AC-5): SHA-256 over normalized text
  - the similarity prefilter (S4.2): exact-match dedupe, MinHash inputs

Keeping these aligned is non-negotiable; if S4.2 needs a different
canonicalization, that surfaces as a story-level change here.
"""

from __future__ import annotations

import hashlib
import re

_WHITESPACE = re.compile(r"\s+")


def normalize_text(s: str) -> str:
    """Lowercase + whitespace-collapse.

    Two strings normalize to the same value iff they differ only in
    whitespace runs and case. Pragmatic baseline; extensions (e.g.
    Unicode NFKC, accent stripping) belong in S4.2 if measurement
    shows they pay off.
    """
    return _WHITESPACE.sub(" ", s).strip().lower()


def content_hash(s: str) -> str:
    """Hex SHA-256 of `normalize_text(s)` — deterministic, 64 chars."""
    return hashlib.sha256(normalize_text(s).encode("utf-8")).hexdigest()
