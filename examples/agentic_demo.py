"""Run the optional LangGraph ClinicalClaw experiment."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from clinicalclaw.agentic import ClinicalClawGraphPipeline
from clinicalclaw.data import DEFAULT_LOCAL_PUBMEDQA_PATH, load_research_pubmedqa
from clinicalclaw.models import SAFETY_DISCLAIMER


def build_agentic_demo_pipeline() -> ClinicalClawGraphPipeline:
    examples = load_research_pubmedqa(limit=100)
    documents = [
        document for example in examples for document in example.to_documents()
    ]
    return ClinicalClawGraphPipeline(documents=documents, top_k=3)


def main() -> None:
    question = (
        " ".join(sys.argv[1:]).strip() or "Does aspirin reduce platelet aggregation?"
    )
    try:
        pipeline = build_agentic_demo_pipeline()
    except ImportError as exc:
        print("ClinicalClaw Agentic Experiment")
        print("===============================")
        print(SAFETY_DISCLAIMER)
        dataset_label = (
            "local PubMedQA subset"
            if DEFAULT_LOCAL_PUBMEDQA_PATH.exists()
            else "built-in PubMedQA-style sample"
        )
        print(f"Dataset: {dataset_label} (up to 100 examples)")
        print()
        print(str(exc))
        return

    final = pipeline.run(question)
    print("ClinicalClaw Agentic Experiment")
    print("===============================")
    print(SAFETY_DISCLAIMER)
    dataset_label = (
        "local PubMedQA subset"
        if DEFAULT_LOCAL_PUBMEDQA_PATH.exists()
        else "built-in PubMedQA-style sample"
    )
    print(f"Dataset: {dataset_label} (up to 100 examples)")
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
