# ClinicalClaw Project Summary

ClinicalClaw is a research prototype for clinical agentic RAG with citation-based generation, claim-level evidence checks, output auditing, and safety guardrails. It is not a medical product and must not be used for diagnosis, treatment, dosing, or emergency decisions.

The repository contains two comparable execution modes:

1. The pure-Python baseline keeps BM25, deterministic dense scoring, hybrid fusion, adaptive retrieval, generation, verification, auditing, and safety decisions readable and testable.
2. The optional LangGraph mode reuses those components as graph nodes. LangGraph provides orchestration and state transitions; it does not make the underlying retrieval or medical reasoning clinically reliable.

The default demo and experiments use up to 100 records from the local `data/PubMedQA/ori_pqaa.json` file when it exists. The large dataset is intentionally ignored by Git. A tiny built-in fallback keeps the repository runnable without private or downloaded data.

Read the code in this order:

`clinicalclaw/models.py` -> `clinicalclaw/data/pubmedqa.py` -> `clinicalclaw/query_enhancement.py` -> `clinicalclaw/pipeline/router.py` -> `clinicalclaw/retrieval/adaptive.py` -> `clinicalclaw/pipeline/workflow.py` -> `clinicalclaw/agentic/workflow.py`.

The current system is suitable for controlled experiments and learning-by-inspection. It still needs real embeddings, vector indexing, reranking, NLI verification, clinical correction models, stronger safety tests, benchmark adapters, and clinician-reviewed evaluation before any pilot discussion.
