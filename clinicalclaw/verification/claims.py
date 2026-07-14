"""Claim extraction baseline."""

from __future__ import annotations

import re

from clinicalclaw.models import Claim, GeneratedAnswer


_CITATION_RE = re.compile(r"\[([^\[\]]+)\]")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


class ClaimExtractor:
    """Extract simple sentence-level claims from generated answers."""

    def extract(self, answer: GeneratedAnswer | str) -> list[Claim]:
        """Return claims with citation IDs parsed from bracketed citations.

        TODO: claim-level faithfulness metrics should consume richer claim spans.
        """

        answer_text = answer.answer if isinstance(answer, GeneratedAnswer) else answer
        sentences = [
            sentence.strip()
            for sentence in _SENTENCE_SPLIT_RE.split(answer_text)
            if sentence.strip()
        ]
        claims: list[Claim] = []
        for index, sentence in enumerate(sentences, start=1):
            cited_doc_ids = _CITATION_RE.findall(sentence)
            clean_text = _clean_claim_text(_CITATION_RE.sub("", sentence))
            if clean_text:
                claims.append(
                    Claim(
                        claim_id=f"claim-{index}",
                        text=clean_text,
                        cited_doc_ids=cited_doc_ids,
                    )
                )
        return claims


def _clean_claim_text(text: str) -> str:
    text = re.sub(r"\s+([.,!?])", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
