"""Hybrid retrieval by combining sparse and deterministic dense scores."""

from __future__ import annotations

from clinicalclaw.models import Document, RetrievalResult
from clinicalclaw.retrieval.bm25 import BM25Retriever
from clinicalclaw.retrieval.dense import DeterministicDenseRetriever


class HybridRetriever:
    """Combine BM25 and dense retrieval with weighted score fusion."""

    def __init__(
        self,
        documents: list[Document],
        bm25_weight: float = 0.65,
        dense_weight: float = 0.35,
    ) -> None:
        if bm25_weight < 0 or dense_weight < 0:
            raise ValueError("Hybrid retrieval weights must be non-negative")
        if bm25_weight == 0 and dense_weight == 0:
            raise ValueError("At least one hybrid retrieval weight must be positive")

        self.documents = documents
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.bm25 = BM25Retriever(documents)
        self.dense = DeterministicDenseRetriever(documents)

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Search with both retrievers and return weighted combined scores."""

        candidate_k = max(top_k, len(self.documents))
        bm25_results = self.bm25.search(query, top_k=candidate_k)
        dense_results = self.dense.search(query, top_k=candidate_k)
        bm25_scores = {result.document.doc_id: result.score for result in bm25_results}
        dense_scores = {
            result.document.doc_id: result.score for result in dense_results
        }
        documents = {document.doc_id: document for document in self.documents}

        normalized_bm25 = _normalize_scores(bm25_scores)
        normalized_dense = _normalize_scores(dense_scores)

        fused: list[RetrievalResult] = []
        for doc_id, document in documents.items():
            bm25_score = normalized_bm25.get(doc_id, 0.0)
            dense_score = normalized_dense.get(doc_id, 0.0)
            combined = self.bm25_weight * bm25_score + self.dense_weight * dense_score
            if combined > 0:
                fused.append(
                    RetrievalResult(
                        document=document,
                        score=combined,
                        rank=0,
                        retriever="hybrid",
                        components={
                            "bm25": bm25_scores.get(doc_id, 0.0),
                            "dense": dense_scores.get(doc_id, 0.0),
                            "bm25_normalized": bm25_score,
                            "dense_normalized": dense_score,
                        },
                    )
                )

        fused.sort(key=lambda result: (-result.score, result.document.doc_id))
        return [
            RetrievalResult(
                document=result.document,
                score=result.score,
                rank=rank,
                retriever=result.retriever,
                components=result.components,
            )
            for rank, result in enumerate(fused[:top_k], start=1)
        ]


def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    maximum = max(scores.values())
    if maximum <= 0:
        return {key: 0.0 for key in scores}
    return {key: value / maximum for key, value in scores.items()}
