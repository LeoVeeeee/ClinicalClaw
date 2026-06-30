"""Command-line demo for ClinicalClaw."""

from __future__ import annotations

from clinicalclaw.data import load_pubmedqa
from clinicalclaw.models import SAFETY_DISCLAIMER
from clinicalclaw.pipeline import ClinicalClawPipeline


def build_demo_pipeline() -> ClinicalClawPipeline:
    """Create a pipeline backed by the built-in tiny sample."""

    examples = load_pubmedqa()
    documents = [document for example in examples for document in example.to_documents()]
    return ClinicalClawPipeline(documents=documents, top_k=3)


def main() -> None:
    """Run a minimal end-to-end demo."""

    question = "Does aspirin reduce platelet aggregation?"
    pipeline = build_demo_pipeline()
    final = pipeline.run(question)

    print("ClinicalClaw Demo")
    print("=================")
    print(SAFETY_DISCLAIMER)
    print()
    print(f"Question: {final.question}")
    print(f"Safety: {final.safety.action} ({final.safety.risk_level})")
    print(f"Answer: {final.answer}")
    print(f"Citations: {', '.join(final.citations) if final.citations else 'none'}")
    print("Verification:")
    for result in final.verifications:
        print(f"- {result.claim_id}: {result.status} ({result.score:.2f})")


if __name__ == "__main__":
    main()
