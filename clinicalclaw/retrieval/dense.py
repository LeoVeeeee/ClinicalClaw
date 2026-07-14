"""Deterministic dense retriever baseline."""

from __future__ import annotations

import hashlib

from clinicalclaw.models import Document, RetrievalResult
from clinicalclaw.retrieval.tokenization import tokenize


class DeterministicDenseRetriever:
    """A deterministic baseline that stands in for embedding search.

    TODO: real embedding model
    TODO: FAISS/Chroma integration
    """

    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        scored = [
            RetrievalResult(
                document=document,
                score=self.score(query, document),
                rank=0,
                retriever="deterministic_dense",
                components={"dense": self.score(query, document)},
            )
            for document in self.documents
        ]
        scored.sort(key=lambda result: (-result.score, result.document.doc_id))
        return [
            RetrievalResult(
                document=result.document,
                score=result.score,
                rank=rank,
                retriever=result.retriever,
                components=result.components,
            )
            for rank, result in enumerate(scored[:top_k], start=1)
        ]

    def score(self, query: str, document: Document) -> float:
        """Return a stable pseudo-semantic score in the range [0, 1]."""

        query_tokens = set(tokenize(query))
        document_tokens = set(tokenize(document.text))
        if not query_tokens or not document_tokens:
            overlap = 0.0
        else:
            overlap = len(query_tokens & document_tokens) / len(
                query_tokens | document_tokens
            )
        return min(1.0, overlap + self._stable_tiebreaker(document.doc_id) * 0.01)

    def _stable_tiebreaker(self, value: str) -> float:
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
        return int(digest[:8], 16) / 0xFFFFFFFF
