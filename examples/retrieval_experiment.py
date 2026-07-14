"""Inspect adaptive retrieval behavior for fixed consultation questions."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from clinicalclaw.data import load_research_pubmedqa
from clinicalclaw.generation import CitationAnswerGenerator
from clinicalclaw.pipeline.router import QueryPlanner
from clinicalclaw.query_enhancement import ClinicalQueryEnhancer
from clinicalclaw.retrieval import AdaptiveRetriever


QUESTIONS = [
    "Does aspirin reduce platelet aggregation?",
    "Can antibiotics treat the flu?",
    "How does aspirin affect thromboxane and platelet aggregation?",
]


def build_documents():
    examples = load_research_pubmedqa(limit=100)
    return [document for example in examples for document in example.to_documents()]


def run_retrieval_experiment(questions: list[str] | None = None, top_k: int = 3):
    """Run adaptive retrieval and return auditable intermediate outputs."""

    documents = build_documents()
    enhancer = ClinicalQueryEnhancer()
    planner = QueryPlanner()
    retriever = AdaptiveRetriever(documents)
    generator = CitationAnswerGenerator()

    rows = []
    for question in questions or QUESTIONS:
        enhanced = enhancer.enhance(question)
        plan = planner.plan(question, enhanced)
        results = retriever.search(plan, enhanced, top_k=top_k)
        generated = generator.generate(question, results)
        rows.append(
            {
                "question": question,
                "enhanced_query": enhanced,
                "query_plan": plan,
                "retrieval_results": results,
                "generated_answer": generated,
            }
        )
    return rows


def main() -> None:
    print("ClinicalClaw Adaptive Retrieval Experiment")
    print("=========================================")
    for row in run_retrieval_experiment():
        plan = row["query_plan"]
        enhanced = row["enhanced_query"]
        print()
        print(f"Question: {row['question']}")
        print(f"Scenario: {plan.scenario}")
        print(f"Route: {plan.route}")
        print(f"Enhanced query: {enhanced.rewritten_query}")
        print(
            f"Expansion terms: {', '.join(enhanced.expansion_terms) if enhanced.expansion_terms else 'none'}"
        )
        print("Retrieved evidence:")
        for result in row["retrieval_results"]:
            print(
                f"- rank={result.rank} doc={result.document.doc_id} score={result.score:.3f} "
                f"bm25={float(result.components.get('bm25', 0.0)):.3f} "
                f"dense={float(result.components.get('dense', 0.0)):.3f}"
            )
            print(f"  text={result.document.text}")
        citations = row["generated_answer"].citations
        print(f"Generated citations: {', '.join(citations) if citations else 'none'}")


if __name__ == "__main__":
    main()
