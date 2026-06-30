from __future__ import annotations

from clinicalclaw.models import Document
from clinicalclaw.verification import ClaimExtractor, EvidenceVerifier


def test_claim_extraction_schema() -> None:
    extractor = ClaimExtractor()

    claims = extractor.extract("Aspirin reduces platelet aggregation [doc-1].")

    assert len(claims) == 1
    assert claims[0].claim_id == "claim-1"
    assert claims[0].text == "Aspirin reduces platelet aggregation."
    assert claims[0].cited_doc_ids == ["doc-1"]


def test_verifier_output_schema() -> None:
    extractor = ClaimExtractor()
    verifier = EvidenceVerifier()
    evidence = [Document(doc_id="doc-1", text="Aspirin reduces platelet aggregation.")]
    claims = extractor.extract("Aspirin reduces platelet aggregation [doc-1].")

    results = verifier.verify(claims, evidence)

    assert len(results) == 1
    assert results[0].claim_id == "claim-1"
    assert results[0].status in {"supported", "contradicted", "not_enough_evidence"}
    assert results[0].evidence_doc_ids == ["doc-1"]
    assert 0.0 <= results[0].score <= 1.0
