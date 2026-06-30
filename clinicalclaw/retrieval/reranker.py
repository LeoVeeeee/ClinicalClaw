"""Reranker placeholder."""

from __future__ import annotations

from clinicalclaw.models import RetrievalResult


class StubReranker:
    """Pass-through reranker for the initial scaffold.

    TODO: reranker
    """

    def rerank(self, query: str, results: list[RetrievalResult]) -> list[RetrievalResult]:
        """Return results unchanged while preserving the future reranker interface."""

        return results
