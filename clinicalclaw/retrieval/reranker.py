"""Reranker interface for retrieval experiments."""

from __future__ import annotations

from clinicalclaw.models import RetrievalResult


class PassThroughReranker:
    """Pass-through reranker for controlled baseline experiments.

    TODO: reranker
    """

    def rerank(
        self, query: str, results: list[RetrievalResult]
    ) -> list[RetrievalResult]:
        """Return results unchanged while preserving the future reranker interface."""

        return results
