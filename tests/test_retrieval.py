from __future__ import annotations

from clinicalclaw.models import Document
from clinicalclaw.retrieval import BM25Retriever, HybridRetriever


def test_bm25_ranks_matching_clinical_text_first() -> None:
    documents = [
        Document(
            doc_id="heart",
            text="Aspirin reduces platelet aggregation in cardiovascular disease.",
        ),
        Document(
            doc_id="space", text="The telescope observed distant galaxies and stars."
        ),
    ]
    retriever = BM25Retriever(documents)

    results = retriever.search("aspirin platelet aggregation", top_k=2)

    assert results[0].document.doc_id == "heart"
    assert results[0].score > results[1].score


def test_hybrid_score_combination_respects_weights() -> None:
    documents = [
        Document(doc_id="aspirin", text="Aspirin reduces platelet aggregation."),
        Document(doc_id="influenza", text="Influenza is a viral infection."),
    ]
    query = "aspirin platelet"

    sparse_only = HybridRetriever(documents, bm25_weight=1.0, dense_weight=0.0).search(
        query, top_k=2
    )
    dense_only = HybridRetriever(documents, bm25_weight=0.0, dense_weight=1.0).search(
        query, top_k=2
    )
    blended = HybridRetriever(documents, bm25_weight=0.25, dense_weight=0.75).search(
        query, top_k=2
    )

    sparse_doc = sparse_only[0]
    dense_doc = dense_only[0]
    blended_doc = blended[0]

    assert sparse_doc.score == sparse_doc.components["bm25_normalized"]
    assert dense_doc.score == dense_doc.components["dense_normalized"]
    expected = (
        0.25 * blended_doc.components["bm25_normalized"]
        + 0.75 * blended_doc.components["dense_normalized"]
    )
    assert blended_doc.score == expected
    assert blended_doc.document.doc_id == "aspirin"
