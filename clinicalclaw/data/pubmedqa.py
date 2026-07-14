"""PubMedQA loader baseline.

The real PubMedQA release uses large JSON files keyed by PubMed ID. This module
also supports compact JSONL samples used by local tests.
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

DEFAULT_LOCAL_PUBMEDQA_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "PubMedQA" / "ori_pqaa.json"
)


def load_pubmedqa(
    path: str | Path | None = None, limit: int | None = None
) -> list[PubMedQAExample]:
    """Load PubMedQA-style examples.

    If ``path`` is omitted, a built-in sample is returned so experiments run
    without downloads.

    ``path`` may point to:
    - a JSONL file with one record per line;
    - a PubMedQA JSON object keyed by PubMed ID, such as ``ori_pqau.json``;
    - a JSON list of records;
    - a directory, in which case ``ori_pqa*.json`` files are loaded in sorted order.

    Use ``limit`` for large PubMedQA files. JSON object/list files are streamed
    record-by-record so small limits do not require loading the whole file.

    TODO: MedQA and MedMCQA evaluation should live in separate dataset adapters.
    """

    if path is None:
        examples = list(_TINY_SAMPLE)
        return examples[:limit] if limit is not None else examples

    json_path = Path(path)
    if any(char in str(json_path) for char in "*?[]"):
        examples: list[PubMedQAExample] = []
        for matched_path in sorted(json_path.parent.glob(json_path.name)):
            examples.extend(
                _load_pubmedqa_file(matched_path, _remaining_limit(limit, examples))
            )
            if limit is not None and len(examples) >= limit:
                break
        if not examples and not list(json_path.parent.glob(json_path.name)):
            raise FileNotFoundError(f"PubMedQA file not found: {json_path}")
        return examples[:limit] if limit is not None else examples

    if not json_path.exists():
        raise FileNotFoundError(f"PubMedQA file not found: {json_path}")

    if json_path.is_dir():
        examples: list[PubMedQAExample] = []
        files = sorted(json_path.glob("ori_pqa*.json"))
        if not files:
            files = sorted(json_path.glob("*.json"))
        for file_path in files:
            examples.extend(
                _load_pubmedqa_file(file_path, _remaining_limit(limit, examples))
            )
            if limit is not None and len(examples) >= limit:
                break
        return examples[:limit] if limit is not None else examples

    return _load_pubmedqa_file(json_path, limit)


def load_research_pubmedqa(
    path: str | Path | None = None,
    limit: int = 100,
) -> list[PubMedQAExample]:
    """Load a bounded local PubMedQA subset for demos and experiments.

    The repository does not ship the large original dataset. When the local
    release is present, this helper reads it incrementally; otherwise it falls
    back to the tiny built-in sample so demos remain runnable after a clean
    checkout.
    """

    if path is not None:
        return load_pubmedqa(Path(path), limit=limit)
    if DEFAULT_LOCAL_PUBMEDQA_PATH.exists():
        return load_pubmedqa(DEFAULT_LOCAL_PUBMEDQA_PATH, limit=limit)
    return load_pubmedqa(limit=limit)


def _remaining_limit(limit: int | None, examples: list[PubMedQAExample]) -> int | None:
    if limit is None:
        return None
    return max(0, limit - len(examples))


def _load_pubmedqa_file(path: Path, limit: int | None) -> list[PubMedQAExample]:
    if limit == 0:
        return []
    if path.suffix.lower() == ".jsonl":
        return _load_jsonl(path, limit)
    if path.suffix.lower() == ".json":
        return _load_json(path, limit)
    raise ValueError(f"Unsupported PubMedQA file extension: {path.suffix}")


def _load_jsonl(path: Path, limit: int | None) -> list[PubMedQAExample]:
    examples: list[PubMedQAExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            examples.append(_normalize_record(payload, line_number))
            if limit is not None and len(examples) >= limit:
                break
    return examples


def _load_json(path: Path, limit: int | None) -> list[PubMedQAExample]:
    examples: list[PubMedQAExample] = []
    for index, (record_id, payload) in enumerate(_iter_json_records(path), start=1):
        examples.append(_normalize_record(payload, index, record_id=record_id))
        if limit is not None and len(examples) >= limit:
            break
    return examples


def _normalize_record(
    payload: dict[str, Any],
    line_number: int,
    record_id: str | None = None,
) -> PubMedQAExample:
    question = payload.get("question") or payload.get("QUESTION")
    if not question:
        raise ValueError(f"PubMedQA record on line {line_number} is missing a question")

    contexts = _normalize_contexts(
        payload.get("context") or payload.get("CONTEXTS") or payload.get("contexts")
    )
    if not contexts:
        raise ValueError(
            f"PubMedQA record on line {line_number} is missing context text"
        )

    return PubMedQAExample(
        example_id=str(
            record_id
            or payload.get("id")
            or payload.get("pubid")
            or f"record-{line_number}"
        ),
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
    answer = (
        payload.get("answer")
        or payload.get("final_decision")
        or payload.get("LONG_ANSWER")
        or payload.get("long_answer")
    )
    return str(answer) if answer is not None else None


class _JSONStream:
    """Small incremental JSON reader for large PubMedQA object files."""

    def __init__(self, path: Path, chunk_size: int = 65536) -> None:
        self._handle = path.open("r", encoding="utf-8")
        self._decoder = json.JSONDecoder()
        self._buffer = ""
        self._position = 0
        self._eof = False
        self._chunk_size = chunk_size

    def close(self) -> None:
        self._handle.close()

    def skip_whitespace(self) -> None:
        while True:
            self._ensure_chars(1)
            while (
                self._position < len(self._buffer)
                and self._buffer[self._position].isspace()
            ):
                self._position += 1
            self._trim_buffer()
            if self._position < len(self._buffer) or self._eof:
                return

    def peek(self) -> str:
        self.skip_whitespace()
        self._ensure_chars(1)
        if self._position >= len(self._buffer):
            raise ValueError("Unexpected end of JSON")
        return self._buffer[self._position]

    def consume(self, expected: str) -> None:
        actual = self.peek()
        if actual != expected:
            raise ValueError(f"Expected {expected!r}, found {actual!r}")
        self._position += 1
        self._trim_buffer()

    def decode_next(self) -> Any:
        self.skip_whitespace()
        while True:
            try:
                value, end = self._decoder.raw_decode(self._buffer, self._position)
            except json.JSONDecodeError:
                if self._eof:
                    raise
                self._read_more()
                continue
            self._position = end
            self._trim_buffer()
            return value

    def _ensure_chars(self, count: int) -> None:
        while not self._eof and len(self._buffer) - self._position < count:
            self._read_more()

    def _read_more(self) -> None:
        chunk = self._handle.read(self._chunk_size)
        if chunk:
            self._buffer += chunk
        else:
            self._eof = True

    def _trim_buffer(self) -> None:
        if self._position > self._chunk_size:
            self._buffer = self._buffer[self._position :]
            self._position = 0


def _iter_json_records(path: Path):
    stream = _JSONStream(path)
    try:
        first = stream.peek()
        if first == "{":
            yield from _iter_json_object_records(stream)
            return
        if first == "[":
            yield from _iter_json_array_records(stream)
            return
        raise ValueError(f"Unsupported JSON root in PubMedQA file: {path}")
    finally:
        stream.close()


def _iter_json_object_records(stream: _JSONStream):
    stream.consume("{")
    if stream.peek() == "}":
        stream.consume("}")
        return
    while True:
        key = stream.decode_next()
        if not isinstance(key, str):
            raise ValueError("Expected PubMedQA JSON object keys to be strings")
        stream.consume(":")
        value = stream.decode_next()
        if not isinstance(value, dict):
            raise ValueError(f"Expected PubMedQA record {key!r} to be an object")
        yield key, value
        separator = stream.peek()
        if separator == ",":
            stream.consume(",")
            continue
        if separator == "}":
            stream.consume("}")
            return
        raise ValueError(
            f"Expected ',' or '}}' in PubMedQA JSON object, found {separator!r}"
        )


def _iter_json_array_records(stream: _JSONStream):
    stream.consume("[")
    if stream.peek() == "]":
        stream.consume("]")
        return
    index = 1
    while True:
        value = stream.decode_next()
        if not isinstance(value, dict):
            raise ValueError(f"Expected PubMedQA array record {index} to be an object")
        yield None, value
        separator = stream.peek()
        if separator == ",":
            stream.consume(",")
            index += 1
            continue
        if separator == "]":
            stream.consume("]")
            return
        raise ValueError(
            f"Expected ',' or ']' in PubMedQA JSON array, found {separator!r}"
        )
