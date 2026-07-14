from __future__ import annotations

import json

import pytest

from clinicalclaw.data import load_pubmedqa, load_research_pubmedqa


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


def test_load_pubmedqa_json_object_keyed_by_pubmed_id(tmp_path) -> None:
    sample_path = tmp_path / "ori_pqa_sample.json"
    sample_path.write_text(
        json.dumps(
            {
                "25429730": {
                    "QUESTION": "Are ILC2s increased in chronic rhinosinusitis?",
                    "CONTEXTS": [
                        "ILC2s were identified in sinus mucosa.",
                        "ILC2 frequencies were associated with nasal polyps.",
                    ],
                    "LABELS": ["BACKGROUND", "RESULTS"],
                    "LONG_ANSWER": "ILC2s are elevated in patients with CRSwNP.",
                    "final_decision": "yes",
                }
            }
        ),
        encoding="utf-8",
    )

    examples = load_pubmedqa(sample_path, limit=1)

    assert examples[0].example_id == "25429730"
    assert examples[0].question == "Are ILC2s increased in chronic rhinosinusitis?"
    assert examples[0].contexts == [
        "ILC2s were identified in sinus mucosa.",
        "ILC2 frequencies were associated with nasal polyps.",
    ]
    assert examples[0].answer == "yes"


def test_load_pubmedqa_json_object_respects_limit(tmp_path) -> None:
    sample_path = tmp_path / "ori_pqa_sample.json"
    sample_path.write_text(
        json.dumps(
            {
                "1": {
                    "QUESTION": "Question 1?",
                    "CONTEXTS": ["Context 1"],
                    "final_decision": "yes",
                },
                "2": {
                    "QUESTION": "Question 2?",
                    "CONTEXTS": ["Context 2"],
                    "final_decision": "no",
                },
            }
        ),
        encoding="utf-8",
    )

    examples = load_pubmedqa(sample_path, limit=1)

    assert len(examples) == 1
    assert examples[0].example_id == "1"


def test_load_pubmedqa_directory_loads_ori_pqa_json_files(tmp_path) -> None:
    (tmp_path / "ori_pqaa.json").write_text(
        json.dumps({"a": {"QUESTION": "Question A?", "CONTEXTS": ["Context A"]}}),
        encoding="utf-8",
    )
    (tmp_path / "ori_pqau.json").write_text(
        json.dumps({"u": {"QUESTION": "Question U?", "CONTEXTS": ["Context U"]}}),
        encoding="utf-8",
    )

    examples = load_pubmedqa(tmp_path, limit=2)

    assert [example.example_id for example in examples] == ["a", "u"]


def test_load_pubmedqa_missing_path_raises_file_not_found(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="PubMedQA file not found"):
        load_pubmedqa(tmp_path / "missing.jsonl")


def test_load_research_pubmedqa_uses_explicit_bounded_path(tmp_path) -> None:
    sample_path = tmp_path / "ori_pqaa.json"
    sample_path.write_text(
        json.dumps(
            {
                "local-1": {
                    "QUESTION": "Local question?",
                    "CONTEXTS": ["Local context."],
                },
                "local-2": {
                    "QUESTION": "Second question?",
                    "CONTEXTS": ["Second context."],
                },
            }
        ),
        encoding="utf-8",
    )

    examples = load_research_pubmedqa(sample_path, limit=1)

    assert len(examples) == 1
    assert examples[0].example_id == "local-1"


def test_load_research_pubmedqa_explicit_missing_path_is_not_silently_ignored(
    tmp_path,
) -> None:
    with pytest.raises(FileNotFoundError, match="PubMedQA file not found"):
        load_research_pubmedqa(tmp_path / "missing.json", limit=1)
