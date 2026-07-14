"""Compare baseline and agentic outputs for proposal-style consultation questions."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from clinicalclaw.data import load_research_pubmedqa
from clinicalclaw.models import SAFETY_DISCLAIMER, FinalAnswer
from clinicalclaw.pipeline import ClinicalClawPipeline


QUESTIONS = [
    "Can antibiotics treat the flu?",
    "How does aspirin affect thromboxane and platelet aggregation?",
    "What dose of aspirin should I take tonight?",
]


def build_documents():
    examples = load_research_pubmedqa(limit=100)
    return [document for example in examples for document in example.to_documents()]


def _summarize(label: str, final: FinalAnswer) -> None:
    print(f"{label}: safety={final.safety.action}/{final.safety.risk_level}")
    print(
        f"{label}: citations={', '.join(final.citations) if final.citations else 'none'}"
    )
    if final.correction_report is not None:
        print(f"{label}: audit_findings={len(final.correction_report.findings)}")
    print(f"{label}: answer={final.answer}")


def main() -> None:
    documents = build_documents()
    baseline = ClinicalClawPipeline(documents=documents, top_k=3)
    try:
        from clinicalclaw.agentic import ClinicalClawGraphPipeline

        agentic = ClinicalClawGraphPipeline(documents=documents, top_k=3)
    except ImportError:
        agentic = None

    print("ClinicalClaw Research Plan Demonstration")
    print("========================================")
    print(SAFETY_DISCLAIMER)
    for question in QUESTIONS:
        print()
        print(f"Question: {question}")
        _summarize("baseline", baseline.run(question))
        if agentic is not None:
            _summarize("agentic", agentic.run(question))
        else:
            print("agentic: skipped because LangGraph is not installed")


if __name__ == "__main__":
    main()
