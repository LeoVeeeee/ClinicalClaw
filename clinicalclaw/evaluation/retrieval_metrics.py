"""Retrieval metrics for ClinicalClaw experiments."""

from __future__ import annotations

from clinicalclaw.models import RetrievalResult


def precision_at_k(
    results: list[RetrievalResult], relevant_doc_ids: set[str], k: int
) -> float:
    """Compute precision@k."""

    if k <= 0:
        return 0.0
    retrieved = [result.document.doc_id for result in results[:k]]
    if not retrieved:
        return 0.0
    return sum(1 for doc_id in retrieved if doc_id in relevant_doc_ids) / k


def recall_at_k(
    results: list[RetrievalResult], relevant_doc_ids: set[str], k: int
) -> float:
    """Compute recall@k."""

    if not relevant_doc_ids:
        return 0.0
    retrieved = {result.document.doc_id for result in results[:k]}
    return len(retrieved & relevant_doc_ids) / len(relevant_doc_ids)


def f1_at_k(
    results: list[RetrievalResult], relevant_doc_ids: set[str], k: int
) -> float:
    """Compute F1@k from precision and recall."""

    precision = precision_at_k(results, relevant_doc_ids, k)
    recall = recall_at_k(results, relevant_doc_ids, k)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def reciprocal_rank(
    results: list[RetrievalResult], relevant_doc_ids: set[str]
) -> float:
    """Compute reciprocal rank for the first relevant result."""

    for index, result in enumerate(results, start=1):
        if result.document.doc_id in relevant_doc_ids:
            return 1 / index
    return 0.0


def average_precision(
    results: list[RetrievalResult], relevant_doc_ids: set[str]
) -> float:
    """Compute average precision across the ranked list."""

    if not relevant_doc_ids:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for index, result in enumerate(results, start=1):
        if result.document.doc_id in relevant_doc_ids:
            hits += 1
            precision_sum += hits / index
    return precision_sum / len(relevant_doc_ids)
