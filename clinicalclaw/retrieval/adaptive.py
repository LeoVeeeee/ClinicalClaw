"""Adaptive retrieval strategy baseline."""

from __future__ import annotations

from clinicalclaw.models import EnhancedQuery, QueryPlan, RetrievalResult
from clinicalclaw.retrieval.hybrid import HybridRetriever


class AdaptiveRetriever:
    """Select retrieval behavior from a research-aligned query plan."""

    def __init__(
        self,
        documents,
        bm25_weight: float = 0.65,
        dense_weight: float = 0.35,
    ) -> None:
        self.hybrid = HybridRetriever(
            documents,
            bm25_weight=bm25_weight,
            dense_weight=dense_weight,
        )

    def search(
        self,
        plan: QueryPlan,
        enhanced_query: EnhancedQuery | None = None,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """Run the retrieval strategy selected by ``plan.route``."""

        if plan.route == "parametric_memory":
            return []

        if plan.route == "atomic":
            query = _primary_query(plan, enhanced_query)
            return _rerank_with_strategy(
                self.hybrid.search(query, top_k=top_k),
                strategy="atomic",
                source_query=query,
            )

        queries = _multi_queries(plan, enhanced_query)
        if plan.route == "reasoning":
            queries.append(
                f"clinical reasoning evidence for {_primary_query(plan, enhanced_query)}"
            )
        return self._multi_search(queries, top_k=top_k, strategy=plan.route)

    def _multi_search(
        self,
        queries: list[str],
        top_k: int,
        strategy: str,
    ) -> list[RetrievalResult]:
        by_doc_id: dict[str, RetrievalResult] = {}
        for query in queries:
            for result in self.hybrid.search(query, top_k=top_k):
                previous = by_doc_id.get(result.document.doc_id)
                if previous is None or result.score > previous.score:
                    by_doc_id[result.document.doc_id] = _with_strategy(
                        result, strategy, query
                    )

        ranked = sorted(
            by_doc_id.values(),
            key=lambda result: (-result.score, result.document.doc_id),
        )
        return [
            RetrievalResult(
                document=result.document,
                score=result.score,
                rank=rank,
                retriever=f"adaptive_{strategy}",
                components=result.components,
            )
            for rank, result in enumerate(ranked[:top_k], start=1)
        ]


def _primary_query(plan: QueryPlan, enhanced_query: EnhancedQuery | None) -> str:
    if enhanced_query is not None:
        return enhanced_query.rewritten_query
    return plan.subqueries[0] if plan.subqueries else plan.original_question


def _multi_queries(plan: QueryPlan, enhanced_query: EnhancedQuery | None) -> list[str]:
    queries = list(plan.subqueries)
    if enhanced_query is not None and enhanced_query.rewritten_query not in queries:
        queries.insert(0, enhanced_query.rewritten_query)
    return _deduplicate_texts(queries)


def _deduplicate_texts(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        normalized = value.strip().lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            deduped.append(value.strip())
    return deduped


def _rerank_with_strategy(
    results: list[RetrievalResult],
    strategy: str,
    source_query: str,
) -> list[RetrievalResult]:
    return [
        RetrievalResult(
            document=result.document,
            score=result.score,
            rank=result.rank,
            retriever=f"adaptive_{strategy}",
            components={
                **result.components,
                "strategy": strategy,
                "source_query": source_query,
            },
        )
        for result in results
    ]


def _with_strategy(
    result: RetrievalResult, strategy: str, source_query: str
) -> RetrievalResult:
    return RetrievalResult(
        document=result.document,
        score=result.score,
        rank=result.rank,
        retriever=f"adaptive_{strategy}",
        components={
            **result.components,
            "strategy": strategy,
            "source_query": source_query,
        },
    )
