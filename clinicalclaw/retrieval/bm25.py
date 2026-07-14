"""Minimal BM25 retriever implemented with the standard library."""

from __future__ import annotations

import math
from collections import Counter

from clinicalclaw.models import Document, RetrievalResult
from clinicalclaw.retrieval.tokenization import tokenize


class BM25Retriever:
    """A compact BM25 implementation for baseline retrieval experiments."""

    def __init__(
        self, documents: list[Document], k1: float = 1.5, b: float = 0.75
    ) -> None:
        self.documents = documents
        self.k1 = k1
        self.b = b
        self._tokenized = [tokenize(document.text) for document in documents]
        self._term_frequencies = [Counter(tokens) for tokens in self._tokenized]
        self._doc_lengths = [len(tokens) for tokens in self._tokenized]
        self._average_doc_length = (
            sum(self._doc_lengths) / len(self._doc_lengths)
            if self._doc_lengths
            else 0.0
        )
        self._document_frequency = self._build_document_frequency()

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Return the highest-scoring documents for ``query``."""

        scored = [
            RetrievalResult(
                document=document,
                score=self.score(query, index),
                rank=0,
                retriever="bm25",
                components={"bm25": self.score(query, index)},
            )
            for index, document in enumerate(self.documents)
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

    def score(self, query: str, document_index: int) -> float:
        """Compute a BM25 score for one indexed document."""

        if not self.documents:
            return 0.0

        score = 0.0
        query_terms = tokenize(query)
        term_frequencies = self._term_frequencies[document_index]
        doc_length = self._doc_lengths[document_index]
        for term in query_terms:
            frequency = term_frequencies.get(term, 0)
            if frequency == 0:
                continue
            score += self._idf(term) * self._term_weight(frequency, doc_length)
        return score

    def _build_document_frequency(self) -> Counter[str]:
        frequency: Counter[str] = Counter()
        for tokens in self._tokenized:
            frequency.update(set(tokens))
        return frequency

    def _idf(self, term: str) -> float:
        document_count = len(self.documents)
        matching_count = self._document_frequency.get(term, 0)
        return math.log(
            1 + (document_count - matching_count + 0.5) / (matching_count + 0.5)
        )

    def _term_weight(self, frequency: int, doc_length: int) -> float:
        if self._average_doc_length == 0:
            return 0.0
        denominator = frequency + self.k1 * (
            1 - self.b + self.b * doc_length / self._average_doc_length
        )
        return frequency * (self.k1 + 1) / denominator
