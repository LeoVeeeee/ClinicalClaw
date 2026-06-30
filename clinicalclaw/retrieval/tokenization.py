"""Small text normalization helpers for toy retrieval."""

from __future__ import annotations

import re


_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric terms."""

    return _TOKEN_RE.findall(text.lower())
