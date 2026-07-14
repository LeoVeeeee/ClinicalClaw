"""Evaluation metric skeletons for ClinicalClaw."""

from __future__ import annotations

from clinicalclaw.models import Claim, RetrievalResult, VerificationResult


def retrieval_recall_at_k(
    results: list[RetrievalResult], relevant_doc_ids: set[str], k: int
) -> float:
    """Compute recall@k for retrieved documents."""

    if not relevant_doc_ids:
        return 0.0
    retrieved = {result.document.doc_id for result in results[:k]}
    return len(retrieved & relevant_doc_ids) / len(relevant_doc_ids)


def citation_coverage(claims: list[Claim]) -> float:
    """Return the fraction of extracted claims that include citations."""

    if not claims:
        return 0.0
    cited = sum(1 for claim in claims if claim.cited_doc_ids)
    return cited / len(claims)


def claim_faithfulness_baseline(verifications: list[VerificationResult]) -> float:
    """Return supported-claim fraction as a baseline faithfulness score.

    TODO: claim-level faithfulness metrics
    """

    if not verifications:
        return 0.0
    supported = sum(1 for result in verifications if result.status == "supported")
    return supported / len(verifications)


def medqa_medmcqa_extension_point() -> None:
    """Extension point for clinical QA benchmark integrations.

    TODO: MedQA and MedMCQA evaluation
    """

    return None
