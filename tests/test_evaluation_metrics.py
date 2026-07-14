from __future__ import annotations

from clinicalclaw.evaluation import (
    average_precision,
    claim_support_rate,
    f1_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from clinicalclaw.models import Document, RetrievalResult, VerificationResult


def _result(doc_id: str, rank: int) -> RetrievalResult:
    return RetrievalResult(
        document=Document(doc_id=doc_id, text=f"Document {doc_id}"),
        score=1 / rank,
        rank=rank,
        retriever="test",
    )


def test_retrieval_metrics_on_fixed_ranking() -> None:
    results = [_result("a", 1), _result("b", 2), _result("c", 3)]
    relevant = {"b", "c"}

    assert precision_at_k(results, relevant, 2) == 0.5
    assert recall_at_k(results, relevant, 2) == 0.5
    assert f1_at_k(results, relevant, 2) == 0.5
    assert reciprocal_rank(results, relevant) == 0.5
    assert average_precision(results, relevant) == ((1 / 2) + (2 / 3)) / 2


def test_claim_support_rate() -> None:
    verifications = [
        VerificationResult("claim-1", "supported", "ok"),
        VerificationResult("claim-2", "not_enough_evidence", "missing"),
    ]

    assert claim_support_rate(verifications) == 0.5
