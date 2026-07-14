# ClinicalClaw Research Roadmap

## Track 1: Adaptive Retrieval

- Maintain deterministic clinical-scenario classification and retrieval-route selection as the reproducible baseline; query complexity remains an internal route-decision signal.
- TODO: compare rule fallback, optional LLM planning, and a trained MedBERT/lightweight classifier on the same annotated route/scenario cases.
- TODO: implement FAISS/Chroma indexing and compare BM25, dense, hybrid, and adaptive retrieval.
- TODO: add a reranker and measure effects on citation relevance.

## Track 2: Query Enhancement

- Maintain terminology mapping for colloquial patient language, such as flu to influenza.
- TODO: connect MeSH/UMLS terminology expansion.
- TODO: add query rewriting for abbreviated or context-limited patient consultation queries.
- TODO: evaluate query enhancement impact on recall, MRR, MAP, and latency.

## Track 3: Output Verification And Correction

- Maintain citation verification and deterministic output audit as the reproducible baseline.
- TODO: replace lexical verification with an NLI verifier.
- TODO: add clinical error detection, localization, and correction inspired by MEDEC/MEDIQA-CORR.
- TODO: add multi-model or multi-sample voting for correction candidates.
- TODO: add medical prompt injection and unsafe-advice tests.

## Track 4: Clinical Evaluation

- Maintain precision@k, recall@k, F1@k, MRR, MAP, claim support rate, unsafe-output flag rate, and latency metrics.
- TODO: add PubMedQA, MedQA, MedMCQA, and MMLU-Med adapters.
- TODO: add clinician review rubrics for relevance, safety, clarity, and workflow fit.
- TODO: add optional LangSmith experiment tracking for RAG and verifier runs.

## Track 5: Pilot Readiness

- Keep all outputs clearly marked as research-prototype output, not clinical advice.
- TODO: define de-identification, ethics/IRB, logging, and clinician-review requirements before any real-world pilot.
- TODO: update `reports/clinicalclaw_report.md` after each reproducible experiment batch with findings, limitations, and next steps.
