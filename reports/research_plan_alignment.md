# ClinicalClaw Research Plan Alignment

This note maps the proposal, "A Clinical RAG-LLM Framework for Patient Consultation: Enhancing Retrieval Efficiency and Output Safety," to the current ClinicalClaw implementation.

## RQ1: Scenario-And-Complexity-Aware Adaptive Retrieval

ClinicalClaw now represents clinical scenario and retrieval route directly in `QueryPlan`. Query complexity is still used internally to choose the route, but it is not part of the public query plan consumed by retrieval. The deterministic baseline supports four retrieval routes: `parametric_memory`, `atomic`, `associative`, and `reasoning`.

Current implementation:
- `clinicalclaw.pipeline.router.QueryPlanner`
- `clinicalclaw.retrieval.adaptive.AdaptiveRetriever`
- `examples/retrieval_experiment.py`
- `configs/routing_rules.json`

Next research step: compare deterministic rules, optional LLM planning, and a MedBERT or lightweight classifier trained on scenario, complexity, and retrieval-route labels.

## RQ2: Clinical Query Enhancement And Output Safety

ClinicalClaw adds pre-retrieval terminology mapping through `ClinicalQueryEnhancer`, then audits generated output with `ClinicalOutputAuditor`.

Current implementation:
- `clinicalclaw.query_enhancement.ClinicalQueryEnhancer`
- `clinicalclaw.correction.output_auditor.ClinicalOutputAuditor`
- `clinicalclaw.safety.policy.SafetyPolicy`

Next research step: connect terminology normalization to MeSH/UMLS and replace deterministic output auditing with clinical error detection, localization, and correction models.

## RQ3: Clinical-Specific Evaluation

ClinicalClaw expands metrics beyond basic recall to include precision@k, recall@k, F1@k, MRR, MAP, claim support rate, unsafe-output flag rate, and latency.

Current implementation:
- `clinicalclaw.evaluation.retrieval_metrics`
- `clinicalclaw.evaluation.safety_metrics`

Next research step: add MedQA, MedMCQA, MMLU-Med, clinician review forms, and LangSmith experiment tracking.

## RQ4: Clinical Validation And Pilot Readiness

ClinicalClaw remains a research prototype and does not conduct real clinical validation. The current code prepares auditable intermediate artifacts needed before any pilot study.

Current implementation:
- baseline and LangGraph experiment runners
- structured correction reports
- safety disclaimer in outputs

Next research step: define clinician feedback rubrics, IRB/ethics constraints, and a de-identified pilot protocol before any real-world testing.
