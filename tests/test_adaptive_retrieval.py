from __future__ import annotations

from clinicalclaw.data import load_pubmedqa
from clinicalclaw.models import QueryPlan
from clinicalclaw.query_enhancement import ClinicalQueryEnhancer
from clinicalclaw.retrieval import AdaptiveRetriever


def _documents():
    return [
        document for example in load_pubmedqa() for document in example.to_documents()
    ]


def test_parametric_memory_skips_retrieval() -> None:
    plan = QueryPlan(
        original_question="hello",
        subqueries=["hello"],
        route="parametric_memory",
    )

    results = AdaptiveRetriever(_documents()).search(plan, top_k=3)

    assert results == []


def test_atomic_retrieval_returns_top_k_results() -> None:
    query = "Does aspirin reduce platelet aggregation?"
    enhanced = ClinicalQueryEnhancer().enhance(query)
    plan = QueryPlan(
        original_question=query,
        subqueries=[enhanced.rewritten_query],
        route="atomic",
    )

    results = AdaptiveRetriever(_documents()).search(plan, enhanced, top_k=2)

    assert len(results) == 2
    assert results[0].retriever == "adaptive_atomic"


def test_associative_retrieval_deduplicates_multi_query_results() -> None:
    plan = QueryPlan(
        original_question="aspirin and thromboxane",
        subqueries=["aspirin platelet", "aspirin thromboxane", "aspirin platelet"],
        route="associative",
    )

    results = AdaptiveRetriever(_documents()).search(plan, top_k=4)
    doc_ids = [result.document.doc_id for result in results]

    assert len(doc_ids) == len(set(doc_ids))
    assert results[0].retriever == "adaptive_associative"


def test_adaptive_retrieval_keeps_auditable_score_components() -> None:
    plan = QueryPlan(
        original_question="aspirin platelet",
        subqueries=["aspirin platelet"],
        route="atomic",
    )

    result = AdaptiveRetriever(_documents()).search(plan, top_k=1)[0]

    assert "bm25" in result.components
    assert "dense" in result.components
    assert result.components["strategy"] == "atomic"
    assert result.components["source_query"] == "aspirin platelet"
