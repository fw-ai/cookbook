# Moved: see `sft_judge_alignment`

This case study measured how well a prompted LLM judge reproduces your 1–5 rubric scores
(Cohen's kappa over a small set of hand-written traces). It has been replaced by
[`../sft_judge_alignment`](../sft_judge_alignment), which keeps that meta-eval and goes further:

- traces come from a **real LangGraph agent** instead of hand-written templates,
- gold labels come from a **three-annotator panel**, which yields a human–human agreement
  **ceiling** to read every kappa against,
- calibration is measured (ECE, reliability diagrams, Brier decomposition), not just agreement,
- the judge is then **fine-tuned** and compared on agreement, calibration, latency, and cost.

The structured-output schema and the quadratic-weighted kappa report from the old notebook live on
in `sft_judge_alignment/judge_client.py` (`rubric_schema`) and `judge_metrics.py` (`kappa_report`).
