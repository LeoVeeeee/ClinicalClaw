"""Claim-level evidence verifier baseline."""

from __future__ import annotations

from clinicalclaw.models import Claim, Document, VerificationResult
from clinicalclaw.retrieval.tokenization import tokenize


class EvidenceVerifier:
    """Verify claims with simple lexical overlap.

    TODO: real NLI verifier
    """

    def verify(
        self, claims: list[Claim], evidence: list[Document]
    ) -> list[VerificationResult]:
        """Return one verification result per claim."""

        evidence_by_id = {document.doc_id: document for document in evidence}
        return [self._verify_one(claim, evidence_by_id) for claim in claims]

    def _verify_one(
        self, claim: Claim, evidence_by_id: dict[str, Document]
    ) -> VerificationResult:
        if not claim.cited_doc_ids:
            return VerificationResult(
                claim_id=claim.claim_id,
                status="not_enough_evidence",
                rationale="No citation was attached to this claim.",
                evidence_doc_ids=[],
                score=0.0,
            )

        cited_documents = [
            evidence_by_id[doc_id]
            for doc_id in claim.cited_doc_ids
            if doc_id in evidence_by_id
        ]
        if not cited_documents:
            return VerificationResult(
                claim_id=claim.claim_id,
                status="not_enough_evidence",
                rationale="The cited evidence was not found in the retrieved evidence set.",
                evidence_doc_ids=[],
                score=0.0,
            )

        claim_terms = set(tokenize(claim.text))
        evidence_terms = set(
            tokenize(" ".join(document.text for document in cited_documents))
        )
        overlap = len(claim_terms & evidence_terms)
        score = overlap / len(claim_terms) if claim_terms else 0.0
        status = "supported" if score >= 0.2 else "not_enough_evidence"
        rationale = (
            "The cited evidence has lexical overlap with the claim."
            if status == "supported"
            else "The cited evidence has too little lexical overlap with the claim."
        )
        return VerificationResult(
            claim_id=claim.claim_id,
            status=status,
            rationale=rationale,
            evidence_doc_ids=[document.doc_id for document in cited_documents],
            score=score,
        )
