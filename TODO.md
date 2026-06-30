# ClinicalClaw 4-Week Development Plan

## Week 1: Naive Medical RAG

- Build a tiny PubMedQA-style dataset loader and inspect normalized examples.
- Implement BM25 retrieval and citation-based answer generation.
- Add a minimal demo that prints evidence, citations, and the safety disclaimer.
- Write baseline tests for loader behavior, retrieval ranking, and answer shape.

## Week 2: Agentic Retrieval

- Add query planning with multiple subqueries.
- TODO: real embedding model for dense retrieval.
- TODO: FAISS/Chroma integration for persistent vector search.
- TODO: reranker to reorder hybrid retrieval candidates.
- TODO: LangGraph workflow to replace the current linear scaffold.

## Week 3: Evidence Verification

- Improve claim extraction with structured claim IDs and citation links.
- TODO: real NLI verifier for supported, contradicted, and not-enough-evidence labels.
- Add verification summaries to final answers.
- TODO: medical prompt injection tests for unsafe or instruction-overriding prompts.

## Week 4: Evaluation and Research Report

- TODO: MedQA and MedMCQA evaluation adapters.
- TODO: claim-level faithfulness metrics.
- Compare naive RAG, hybrid RAG, reranked RAG, and verified RAG.
- Write findings, limitations, and future work in `reports/clinicalclaw_report.md`.
