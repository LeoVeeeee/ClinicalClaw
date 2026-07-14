# ClinicalClaw Research Report

## Scope And Safety

ClinicalClaw is a research prototype for studying clinical RAG workflow design. It is not a medical product, clinical decision support system, diagnosis tool, or substitute for a qualified clinician. The outputs below are software experiment observations, not medical conclusions.

## Current Architecture

The baseline pipeline is:

`question -> query enhancement -> QueryPlanner -> AdaptiveRetriever -> reranker interface -> citation generator -> claim extractor -> evidence verifier -> output auditor -> safety policy -> FinalAnswer`.

The optional agentic implementation uses LangGraph 1.x to orchestrate the same components through explicit state. It adds graph nodes for guardrail routing, query enhancement, planning, adaptive retrieval, generation, verification, correction, and final safety handling. The local environment used for this report had LangGraph `1.1.6`, `langchain-core` `1.2.28`, and OpenAI-compatible client `2.45.0`. The top-level `langchain` package remains optional and was not installed; LangGraph and the OpenAI-compatible planner still ran successfully.

## Data Verification

The local original PubMedQA file was at the project-relative path `data/PubMedQA/ori_pqaa.json`. Its JSON root is an object keyed by PubMed ID. The incremental loader successfully read the first 100 records without loading the full 533 MB file into memory. The first loaded record was PubMed ID `25429730`.

The evaluation runner used a bounded subset of the same local file with deterministic planning. The full original dataset is ignored by Git because it is large and local; `data/pubmedqa_tiny.jsonl` remains available as a small tracked fallback. The evaluator has an explicit `--use-llm` mode for a separate comparison, so the baseline report is not changed by a local API key.

## Acceptance Runs

The following commands completed successfully from the project root:

```bash
pytest
python examples/demo.py
python examples/agentic_demo.py
python examples/retrieval_experiment.py
python examples/research_plan_demo.py
python examples/evaluate_adaptive_retrieval.py
python examples/evaluate_adaptive_retrieval.py --use-llm --case-limit 1
```

Both demos reported that they used the local PubMedQA subset when the local release was available. The baseline and agentic demo produced the same deterministic answer for the same evidence, which is expected because the graph mode currently changes orchestration, not the core models.

The three-step LLM planning smoke test also completed successfully with the local `.env` configuration. It returned `route=reasoning`, `scenario=biomedical_research`, and a public query plan containing only `original_question`, `route`, `scenario`, and `subqueries`. No API key value was printed or stored in the repository.

## Evaluation Result

The adaptive retrieval evaluation ran 10 labeled cases using the local PubMedQA corpus bounded to 30 records and `top_k=3`:

| Metric | Result |
| --- | ---: |
| Routing accuracy | 1.000 |
| Scenario accuracy | 1.000 |
| Mean precision@3 | 0.815 |
| Mean recall@3 | 0.806 |
| Mean MRR | 1.000 |

These values measure the current deterministic routing and retrieval scaffold on a small labeled slice. They do not demonstrate clinical correctness.

## Findings

- The planner correctly routed the included fixed cases for the current rule set.
- Query enhancement rewrote colloquial `flu` to `influenza` in the tested consultation example.
- The LangGraph workflow completed with the configured OpenAI-compatible client and preserved the same final answer contract as the baseline. Unit tests explicitly disable dotenv loading so they remain offline and deterministic.
- The optional LLM planner completed successfully, but it did not always choose the same route as the deterministic rules. For example, one relationship query was classified as `reasoning` by the LLM and `associative` by the baseline. These results must be evaluated as separate experiment conditions.
- Retrieval on the first 100 PubMedQA records can return unrelated evidence for an arbitrary aspirin question. The deterministic generator then summarizes retrieved text rather than answering with clinically validated reasoning. This is an expected limitation of the baseline, not evidence of safe medical QA.
- The safety and audit modules flag medication wording and unsupported claims, but they are deterministic heuristics and can over- or under-flag output.

## Next Experiments

1. Add a real embedding model and compare dense, BM25, hybrid, and adaptive retrieval.
2. Add a reranker and measure citation relevance on a held-out PubMedQA subset.
3. Replace lexical verification with an entailment/contradiction model.
4. Add claim-level faithfulness, answer correctness, latency, and unsafe-output evaluations.
5. Add MedQA, MedMCQA, MMLU-Med, prompt-injection cases, and clinician review only after the local baseline is reproducible.

## Reproducibility Note

Keep API keys in local environment variables or an ignored `.env`/`configs/.env`; never commit them. Keep the large PubMedQA, MedQA, and MedMCQA files outside version control. All repository examples use project-relative paths and bounded reads.
