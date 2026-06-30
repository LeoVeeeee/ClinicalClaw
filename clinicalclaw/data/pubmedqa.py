"""PubMedQA loader stub.

The real PubMedQA dataset has richer fields and multiple splits. This module
normalizes tiny JSONL-style samples into one beginner-friendly shape.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from clinicalclaw.models import PubMedQAExample


_TINY_SAMPLE = [
    PubMedQAExample(
        example_id="pubmedqa-sample-1",
        question="Does aspirin reduce platelet aggregation?",
        contexts=[
            "Aspirin irreversibly inhibits cyclooxygenase activity in platelets.",
            "This reduces thromboxane A2 formation and platelet aggregation.",
        ],
        answer="yes",
    ),
    PubMedQAExample(
        example_id="pubmedqa-sample-2",
        question="Can antibiotics treat viral influenza?",
        contexts=[
            "Influenza is caused by influenza viruses.",
            "Antibiotics target bacteria and are not effective against uncomplicated viral influenza.",
        ],
        answer="no",
    ),
]


def load_pubmedqa(path: str | Path | None = None, limit: int | None = None) -> list[PubMedQAExample]:
    """Load a tiny PubMedQA-style dataset.

    If ``path`` is omitted, a built-in sample is returned so the demo works
    without downloads. If ``path`` is provided, it must point to a JSONL file
    where each line has at least a question and context field.

    TODO: MedQA and MedMCQA evaluation should live in separate dataset adapters.
    """

    if path is None:
        examples = list(_TINY_SAMPLE)
        return examples[:limit] if limit is not None else examples

    jsonl_path = Path(path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"PubMedQA file not found: {jsonl_path}")

    examples: list[PubMedQAExample] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            examples.append(_normalize_record(payload, line_number))
            if limit is not None and len(examples) >= limit:
                break
    return examples


def _normalize_record(payload: dict[str, Any], line_number: int) -> PubMedQAExample:
    question = payload.get("question") or payload.get("QUESTION")
    if not question:
        raise ValueError(f"PubMedQA record on line {line_number} is missing a question")

    contexts = _normalize_contexts(payload.get("context") or payload.get("CONTEXTS") or payload.get("contexts"))
    if not contexts:
        raise ValueError(f"PubMedQA record on line {line_number} is missing context text")

    return PubMedQAExample(
        example_id=str(payload.get("id") or payload.get("pubid") or f"record-{line_number}"),
        question=str(question),
        contexts=contexts,
        answer=_normalize_answer(payload),
    )


def _normalize_contexts(raw_context: Any) -> list[str]:
    if raw_context is None:
        return []
    if isinstance(raw_context, str):
        return [raw_context]
    if isinstance(raw_context, list):
        return [str(item) for item in raw_context if str(item).strip()]
    if isinstance(raw_context, dict):
        values: list[str] = []
        for value in raw_context.values():
            if isinstance(value, list):
                values.extend(str(item) for item in value if str(item).strip())
            elif str(value).strip():
                values.append(str(value))
        return values
    return [str(raw_context)]


def _normalize_answer(payload: dict[str, Any]) -> str | None:
    answer = payload.get("answer") or payload.get("final_decision") or payload.get("long_answer")
    return str(answer) if answer is not None else None
