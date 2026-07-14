# ClinicalClaw 20 天开发计划

本计划面向项目编码和 LangChain/LangGraph 初学者。目标是先把 ClinicalClaw 的确定性研究基线理解清楚，再逐步进入自适应检索、LangGraph 多智能体编排、证据验证、安全评估和研究报告。

## 每日固定流程

每天开始编码前先运行：

```bash
pytest
python examples/demo.py
python examples/agentic_demo.py
```

每天完成编码后再次运行：

```bash
pytest
python examples/demo.py
python examples/agentic_demo.py
```

如果当天涉及检索或研究计划实验，还运行：

```bash
python examples/retrieval_experiment.py
python examples/research_plan_demo.py
```

## Week 1: 理解并测量基线 RAG

### Day 1: 项目地图

阅读：

- `README.md`
- `clinicalclaw/models.py`
- `clinicalclaw/pipeline/workflow.py`

实现：

- 在 `TODO.md` 中补充每个主要模块的用途说明。

验收：

- 你能口头解释完整流程：question -> retrieval -> answer -> verification -> safety。

TODO 注释：

```text
TODO[Day1]: Understand the baseline pipeline before adding real models.
```

### Day 2: 数据集与 Document

阅读：

- `clinicalclaw/data/pubmedqa.py`
- `data/pubmedqa_tiny.jsonl`

实现：

- 新增 3 条 PubMedQA 风格的小样例。

验收：

- `pytest` 通过。
- `python examples/demo.py` 正常运行。

TODO 注释：

```text
TODO[Data]: Expand the real PubMedQA loader experiments beyond the bounded local subset.
```

### Day 3: BM25 检索

阅读：

- `clinicalclaw/retrieval/bm25.py`
- `clinicalclaw/retrieval/tokenization.py`

实现：

- 给 BM25 代码补充注释，解释 IDF、term frequency、ranking。

验收：

- 你能解释为什么某个文档排在另一个文档前面。

TODO 注释：

```text
TODO[Retrieval-BM25]: Compare BM25 against dense retrieval on the same questions.
```

### Day 4: 自适应检索评估集

阅读：

- `clinicalclaw/pipeline/router.py`
- `configs/routing_rules.json`
- `clinicalclaw/retrieval/adaptive.py`

实现：

- 创建 `data/adaptive_retrieval_eval.jsonl`。
- 写 6-8 条带标签的问题。

每条样例建议包含：

```json
{
  "question": "...",
  "expected_route": "atomic | associative | reasoning | parametric_memory",
  "expected_scenario": "...",
  "relevant_doc_ids": ["..."]
}
```

验收：

- 每条样例都有 expected route、scenario 和 relevant docs，其中 route 表示检索策略。complexity 只是 planner 内部判断 route 的中间信号，不写入 `QueryPlan`。

TODO 注释：

```text
TODO[RQ1]: Replace deterministic scenario and retrieval-route rules with a trained classifier.
```

### Day 5: 自适应检索评估脚本

目标：

- 把 Day 4 的“手工标注小样例”变成一个可以运行的评估脚本。
- 先评估检索链路，不评估最终回答质量。也就是说，Day 5 只关心系统有没有把临床场景分对、选对 retrieval route、找回正确 evidence。
- 你不需要训练模型，也不需要大规模数据。今天的重点是学会写一个最小可复现实验。

你要理解的概念：

- `routing accuracy`: `QueryPlanner` 是否选择了预期的 retrieval route，即 `parametric_memory` / `atomic` / `associative` / `reasoning`。
- `scenario accuracy`: `QueryPlanner` 是否选择了预期 clinical scenario。
- `recall@3`: 正确 evidence 是否出现在前 3 个检索结果里。
- `MRR`: 第一个正确 evidence 出现得越靠前，分数越高。

实现：

- 添加 `examples/evaluate_adaptive_retrieval.py`。
- 添加或继续扩展 `data/adaptive_retrieval_eval.jsonl`。
- 每条 JSONL 样例包含：
  - `question`
  - `expected_route`
  - `expected_scenario`
  - `relevant_doc_ids`
  - 可选 `expected_safety_action`
- 输出 routing accuracy、scenario accuracy、recall@3、MRR。

推荐步骤：

1. 先运行脚本，不改代码，读懂输出。
2. 打开 `data/adaptive_retrieval_eval.jsonl`，新增 3 条你自己写的问题。
3. 预测它们的 expected labels。
4. 再运行脚本，看哪些样例是 `CHECK`。
5. 如果 `CHECK` 合理，修改样例标签；如果 `CHECK` 暴露系统问题，再修改 router/planner 规则。

验收：

```bash
python examples/evaluate_adaptive_retrieval.py
python examples/evaluate_adaptive_retrieval.py --top-k 5
pytest
```

完成标准：

- 脚本可以运行并打印 summary metrics。
- 至少 7 条 evaluation cases 可以被加载。
- 你能解释一个 `PASS` 样例为什么通过。
- 你能解释一个 `CHECK` 样例是数据标签错了，还是系统规则错了。
- 你能说清楚 `recall@3` 和 `MRR` 的区别。

TODO 注释：

```text
TODO[Eval]: Track retrieval quality separately from generation quality.
TODO[Day5-Data]: Add 20-50 hand-labeled adaptive retrieval evaluation cases.
TODO[Day5-Report]: Save evaluation summaries under reports/ for experiment comparison.
```

## Week 2: LangGraph 与 Agentic Retrieval

### Day 6: LangGraph 基础

阅读：

- `clinicalclaw/agentic/state.py`
- `clinicalclaw/agentic/workflow.py`

实现：

- 给每个 LangGraph node 补充注释，解释输入 state 和输出 state。

验收：

- 你能解释每个 agent node 接收什么、返回什么。

TODO 注释：

```text
TODO[LangGraph]: Keep graph nodes thin; put research logic in reusable modules.
```

### Day 7: 对比 Baseline 与 Agentic

阅读：

- `examples/research_plan_demo.py`

实现：

- 新增 1 个问题，用于比较 baseline 和 agentic 输出。

验收：

```bash
python examples/research_plan_demo.py
```

TODO 注释：

```text
TODO[Experiment]: Record cases where graph orchestration changes output behavior.
```

### Day 8: Query Enhancement

阅读：

- `clinicalclaw/query_enhancement.py`

实现：

- 新增 5 个术语映射，例如：
  - `heartburn`
  - `sugar disease`
  - `blood thinner`
  - `high cholesterol`
  - `shortness of breath`

验收：

- 测试至少证明 2 个新增映射有效。

TODO 注释：

```text
TODO[RQ2]: Replace built-in terminology map with MeSH/UMLS terminology expansion.
```

### Day 9: Associative Retrieval

阅读：

- `clinicalclaw/retrieval/adaptive.py`

实现：

- 新增测试，证明 multi-query retrieval 会去重文档。

验收：

- associative retrieval 返回的 doc IDs 没有重复。

TODO 注释：

```text
TODO[Adaptive]: Evaluate whether associative retrieval improves recall but hurts precision.
```

### Day 10: Week 2 研究记录

实现：

- 创建 `reports/week2_agentic_retrieval_notes.md`。

验收：

- 报告包含：
  - what worked
  - what failed
  - next experiment

## Week 3: Verification 与 Safety

### Day 11: Claim Extraction

阅读：

- `clinicalclaw/verification/claims.py`

实现：

- 添加两个测试：
  - two-sentence answer
  - uncited claim

验收：

- 抽取出的 claims 有稳定的 `claim_id`。

TODO 注释：

```text
TODO[Claims]: Link claim spans to exact answer sentence offsets.
```

### Day 12: Evidence Verification

阅读：

- `clinicalclaw/verification/verifier.py`

实现：

- 添加 `not_enough_evidence` 的测试。

验收：

- unsupported claims 会被标记出来。

TODO 注释：

```text
TODO[RQ2]: Replace lexical overlap verifier with NLI entailment model.
```

### Day 13: Output Auditor

阅读：

- `clinicalclaw/correction/output_auditor.py`

实现：

- 新增一条规则，用于检测绝对化医疗表述，例如：
  - `always`
  - `never`
  - `guaranteed`

验收：

- 测试证明 absolute wording 会被 flag。

TODO 注释：

```text
TODO[Correction]: Add error detection, localization, and correction stages.
```

### Day 14: Safety Policy

阅读：

- `clinicalclaw/safety/policy.py`

实现：

- 增加更多 emergency/advice phrases。

验收：

- emergency prompts -> `refuse`
- dosage prompts -> `caution`

TODO 注释：

```text
TODO[Safety]: Build medical prompt-injection test set.
```

### Day 15: Safety Report

实现：

- 创建 `reports/week3_safety_verification_notes.md`。

验收：

- 至少包含 3 个 unsafe prompt examples 和系统行为观察。

## Week 4: Evaluation 与 Research Report

### Day 16: Retrieval Metrics

阅读：

- `clinicalclaw/evaluation/retrieval_metrics.py`

实现：

- 为 MAP 和 MRR 增加 edge case 测试。

验收：

- 空结果返回 `0.0`。

TODO 注释：

```text
TODO[Eval-Retrieval]: Report precision, recall, MRR, MAP for each retrieval route.
```

### Day 17: Safety Metrics

阅读：

- `clinicalclaw/evaluation/safety_metrics.py`

实现：

- 添加一个 correction report evaluator。

验收：

- 能输出 unsafe-output flag rate。

TODO 注释：

```text
TODO[Eval-Safety]: Separate unsafe-answer detection from answer correctness.
```

### Day 18: Benchmark Planning

实现：

- 创建 `reports/benchmark_plan.md`。

报告包含：

- PubMedQA
- MedQA
- MedMCQA
- MMLU-Med
- clinician review

TODO 注释：

```text
TODO[Benchmarks]: Add dataset adapters without changing pipeline APIs.
```

### Day 19: Final Research Report Draft

实现：

- 创建 `reports/clinicalclaw_report.md`。

报告章节：

- Objective
- Method
- Current System
- Experiments
- Limitations
- Next Steps

验收：

- 报告明确对应 RQ1-RQ4。

### Day 20: Cleanup And Presentation

实现：

- 检查 README、TODO、reports。
- 运行所有脚本。

验收：

```bash
pytest
python examples/demo.py
python examples/agentic_demo.py
python examples/retrieval_experiment.py
python examples/research_plan_demo.py
```

最终 TODO：

```text
TODO[Next Phase]: Start real embedding + vector database integration only after baseline evaluation is stable.
```

## 核心原则

在确定性 baseline 可以被稳定评估之前，不要急着加入真实 LangChain model calls、FAISS、Chroma 或 NLI verifier。

当前阶段最重要的问题是：

```text
Does the deterministic baseline behave correctly and measurably?
```
