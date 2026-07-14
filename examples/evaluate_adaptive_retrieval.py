"""Day 5 scaffold: evaluate adaptive retrieval routing and ranking.

This script is intentionally small and explicit. It helps you check whether the
rule-based router/planner/retriever behaves as expected before you move to
larger PubMedQA, MedQA, or MedMCQA experiments.

Run:
    python examples/evaluate_adaptive_retrieval.py

Optional:
    python examples/evaluate_adaptive_retrieval.py --top-k 5
    python examples/evaluate_adaptive_retrieval.py --cases data/adaptive_retrieval_eval.jsonl
    python examples/evaluate_adaptive_retrieval.py --use-llm

TODO[Day5-Data]: Add 20-50 hand-labeled questions after you understand this
script. Keep each label small: retrieval route, scenario, and relevant
document IDs.

TODO[Eval]: Track retrieval quality separately from generation quality.
Retrieval metrics answer "did we find the right evidence?" Generation metrics
answer "did the final answer use the evidence correctly?"
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from clinicalclaw.data import load_pubmedqa
from clinicalclaw.evaluation import precision_at_k, recall_at_k, reciprocal_rank
from clinicalclaw.pipeline.router import QueryPlanner
from clinicalclaw.query_enhancement import ClinicalQueryEnhancer
from clinicalclaw.retrieval import AdaptiveRetriever
from clinicalclaw.safety import SafetyPolicy


DEFAULT_CASES_PATH = PROJECT_ROOT / "data" / "adaptive_retrieval_eval.jsonl"
DEFAULT_PUBMEDQA_PATH = PROJECT_ROOT / "data" / "PubMedQA" / "ori_pqaa.json"
ADAPTIVE_ROUTES = {"parametric_memory", "atomic", "associative", "reasoning"}


@dataclass(frozen=True)
class AdaptiveRetrievalCase:
    """One labeled question for Day 5 evaluation.

    Beginner note:
    - expected_route checks the retrieval route.
    - expected_scenario checks the clinical situation label.
    - relevant_doc_ids are the evidence IDs that should appear in top-k.
    """

    case_id: str
    question: str
    expected_route: str | None = None
    expected_scenario: str | None = None
    relevant_doc_ids: list[str] = field(default_factory=list)
    expected_safety_action: str | None = None


def load_cases(path: Path, limit: int | None = None) -> list[AdaptiveRetrievalCase]:
    """Load JSONL evaluation cases.

    TODO[Day5-Validation]: If this file grows, add stronger validation for
    allowed route, scenario, and safety labels.
    """

    cases: list[AdaptiveRetrievalCase] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            cases.append(_case_from_payload(payload, line_number))
            if limit is not None and len(cases) >= limit:
                break
    return cases


def _case_from_payload(
    payload: dict[str, Any], line_number: int
) -> AdaptiveRetrievalCase:
    if "question" not in payload:
        raise ValueError(f"Case on line {line_number} is missing 'question'")

    expected_route = _expected_route_from_payload(payload, line_number)
    return AdaptiveRetrievalCase(
        case_id=str(payload.get("case_id", f"case-{line_number}")),
        question=str(payload["question"]),
        expected_route=expected_route,
        expected_scenario=_optional_str(payload.get("expected_scenario")),
        expected_safety_action=_optional_str(payload.get("expected_safety_action")),
        relevant_doc_ids=[
            str(doc_id) for doc_id in payload.get("relevant_doc_ids", [])
        ],
    )


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _expected_route_from_payload(
    payload: dict[str, Any], line_number: int
) -> str | None:
    """Read the adaptive retrieval route label from a case payload.

    New evaluation files should use ``expected_route`` with one of the values
    in ``ADAPTIVE_ROUTES``.
    """

    expected_route = _optional_str(payload.get("expected_route"))
    route = expected_route
    if route is not None and route not in ADAPTIVE_ROUTES:
        raise ValueError(
            f"Case on line {line_number} has expected_route={route!r}; "
            f"expected one of {sorted(ADAPTIVE_ROUTES)}"
        )
    return route


def build_documents(
    pubmedqa_path: Path | None = None, pubmedqa_limit: int | None = None
):
    """Build the retrieval corpus used by this evaluation.

        By default, this uses ``data/PubMedQA/ori_pqaa.json`` when that file is
        present, because the Day 5 evaluation cases reference real PubMedQA IDs.
        If the file is absent, it falls back to the tiny built-in sample.

    TODO[Day5-PubMedQA]: Expand the current real-data slice with 20-50 additional
    hand-labeled questions and held-out relevant document IDs.
    """

    examples = (
        load_pubmedqa(pubmedqa_path, limit=pubmedqa_limit)
        if pubmedqa_path
        else load_pubmedqa()
    )
    return [document for example in examples for document in example.to_documents()]


def evaluate_cases(
    cases: list[AdaptiveRetrievalCase],
    top_k: int = 3,
    pubmedqa_path: Path | None = None,
    pubmedqa_limit: int | None = None,
    use_llm: bool = False,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Run planning, safety, and adaptive retrieval over cases.

    The default uses deterministic planning so reported baseline metrics are
    reproducible even when a local LLM key is configured. Pass ``use_llm`` to
    compare the optional configured LLM planner.
    """

    documents = build_documents(
        pubmedqa_path=pubmedqa_path, pubmedqa_limit=pubmedqa_limit
    )
    enhancer = ClinicalQueryEnhancer()
    planner = QueryPlanner(enable_llm="auto" if use_llm else "never")
    retriever = AdaptiveRetriever(documents)
    safety_policy = SafetyPolicy()

    rows: list[dict[str, Any]] = []
    for case in cases:
        enhanced = enhancer.enhance(case.question)
        plan = planner.plan(case.question, enhanced)
        retrieval_results = retriever.search(plan, enhanced, top_k=top_k)
        safety_decision = safety_policy.evaluate(case.question)

        relevant_doc_ids = set(case.relevant_doc_ids)
        row = {
            "case_id": case.case_id,
            "question": case.question,
            "route": plan.route,
            "expected_route": case.expected_route,
            "scenario": plan.scenario,
            "expected_scenario": case.expected_scenario,
            "safety_action": safety_decision.action,
            "expected_safety_action": case.expected_safety_action,
            "rewritten_query": enhanced.rewritten_query,
            "retrieved_doc_ids": [
                result.document.doc_id for result in retrieval_results
            ],
            "relevant_doc_ids": sorted(relevant_doc_ids),
            "precision_at_k": (
                precision_at_k(retrieval_results, relevant_doc_ids, top_k)
                if relevant_doc_ids
                else None
            ),
            "recall_at_k": (
                recall_at_k(retrieval_results, relevant_doc_ids, top_k)
                if relevant_doc_ids
                else None
            ),
            "mrr": (
                reciprocal_rank(retrieval_results, relevant_doc_ids)
                if relevant_doc_ids
                else None
            ),
        }
        rows.append(row)

    return summarize(rows), rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate Day 5 metrics.

    TODO[Day5-Report]: Save this summary as JSON or Markdown under reports/
    when you start comparing multiple retrieval strategies.
    """

    summary = {
        "routing_accuracy": _accuracy(rows, "route", "expected_route"),
        "scenario_accuracy": _accuracy(rows, "scenario", "expected_scenario"),
        "mean_precision_at_k": _mean_metric(rows, "precision_at_k"),
        "mean_recall_at_k": _mean_metric(rows, "recall_at_k"),
        "mean_mrr": _mean_metric(rows, "mrr"),
    }
    safety_accuracy = _accuracy(rows, "safety_action", "expected_safety_action")
    if safety_accuracy is not None:
        summary["safety_action_accuracy"] = safety_accuracy
    return summary


def _accuracy(
    rows: list[dict[str, Any]], actual_key: str, expected_key: str
) -> float | None:
    labeled = [row for row in rows if row[expected_key] is not None]
    if not labeled:
        return None
    return sum(1 for row in labeled if row[actual_key] == row[expected_key]) / len(
        labeled
    )


def _mean_metric(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if row[key] is not None]
    if not values:
        return 0.0
    return sum(values) / len(values)


def print_report(
    summary: dict[str, float], rows: list[dict[str, Any]], top_k: int
) -> None:
    print("ClinicalClaw Day 5: Adaptive Retrieval Evaluation")
    print("=================================================")
    print(f"Cases: {len(rows)}")
    print(f"Top-k: {top_k}")
    print()
    print("Summary metrics")
    for metric_name, value in summary.items():
        print(f"- {metric_name}: {value:.3f}")

    print()
    print("Case details")
    for row in rows:
        status = "PASS" if _row_passed(row) else "CHECK"
        print(f"- {row['case_id']} [{status}] {row['question']}")
        print(
            "  route={route} "
            "scenario={scenario} safety={safety_action}".format(**row)
        )
        mismatches = _mismatches(row)
        if mismatches:
            print(f"  mismatches={'; '.join(mismatches)}")
        print(f"  rewritten_query={row['rewritten_query']}")
        print(f"  retrieved={row['retrieved_doc_ids'] or 'none'}")
        if row["relevant_doc_ids"]:
            print(
                f"  relevant={row['relevant_doc_ids']} "
                f"precision@{top_k}={row['precision_at_k']:.3f} "
                f"recall@{top_k}={row['recall_at_k']:.3f} "
                f"mrr={row['mrr']:.3f}"
            )
        else:
            print("  relevant=none; retrieval ranking metrics skipped for this case")


def _row_passed(row: dict[str, Any]) -> bool:
    checks = [
        row["expected_route"] is None or row["route"] == row["expected_route"],
        row["expected_scenario"] is None or row["scenario"] == row["expected_scenario"],
        row["expected_safety_action"] is None
        or row["safety_action"] == row["expected_safety_action"],
    ]
    return all(checks)


def _mismatches(row: dict[str, Any]) -> list[str]:
    """Return human-readable label mismatches for beginner debugging."""

    pairs = [
        ("route", "expected_route"),
        ("scenario", "expected_scenario"),
        ("safety_action", "expected_safety_action"),
    ]
    messages: list[str] = []
    for actual_key, expected_key in pairs:
        expected = row[expected_key]
        if expected is not None and row[actual_key] != expected:
            messages.append(f"{actual_key}: expected {expected}, got {row[actual_key]}")
    return messages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        type=Path,
        default=DEFAULT_CASES_PATH,
        help="Path to JSONL eval cases.",
    )
    parser.add_argument(
        "--case-limit",
        type=int,
        default=None,
        help="Optional number of cases to evaluate.",
    )
    parser.add_argument(
        "--top-k", type=int, default=3, help="Number of retrieved documents to score."
    )
    parser.add_argument(
        "--pubmedqa-path",
        type=Path,
        default=DEFAULT_PUBMEDQA_PATH if DEFAULT_PUBMEDQA_PATH.exists() else None,
        help="Optional PubMedQA file/directory. Default uses data/PubMedQA/ori_pqaa.json when present.",
    )
    parser.add_argument(
        "--pubmedqa-limit",
        type=int,
        default=30,
        help="Optional limit when loading a larger PubMedQA path.",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use the configured LLM planner instead of deterministic rules.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = load_cases(args.cases, limit=args.case_limit)
    summary, rows = evaluate_cases(
        cases,
        top_k=args.top_k,
        pubmedqa_path=args.pubmedqa_path,
        pubmedqa_limit=args.pubmedqa_limit,
        use_llm=args.use_llm,
    )
    print_report(summary, rows, top_k=args.top_k)


if __name__ == "__main__":
    main()
