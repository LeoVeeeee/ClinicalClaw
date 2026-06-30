from __future__ import annotations

import json

import pytest

from clinicalclaw.data import load_pubmedqa


def test_load_builtin_pubmedqa_sample_returns_normalized_examples() -> None:
    examples = load_pubmedqa(limit=1)

    assert len(examples) == 1
    assert examples[0].example_id == "pubmedqa-sample-1"
    assert examples[0].question
    assert examples[0].contexts
    assert examples[0].to_documents()[0].source == "pubmedqa"


def test_load_pubmedqa_jsonl_normalizes_records(tmp_path) -> None:
    sample_path = tmp_path / "sample.jsonl"
    sample_path.write_text(
        json.dumps(
            {
                "id": "x1",
                "question": "Does intervention help?",
                "context": {"contexts": ["Intervention improved the outcome."]},
                "answer": "yes",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    examples = load_pubmedqa(sample_path)

    assert examples[0].example_id == "x1"
    assert examples[0].contexts == ["Intervention improved the outcome."]
    assert examples[0].answer == "yes"


def test_load_pubmedqa_missing_path_raises_file_not_found(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="PubMedQA file not found"):
        load_pubmedqa(tmp_path / "missing.jsonl")
