#!/usr/bin/env python3

from __future__ import annotations

import json
import unittest
from pathlib import Path


SKILL_ROOT = Path(__file__).resolve().parents[1]
REFERENCE = (SKILL_ROOT / "references" / "cost-estimation.md").read_text(encoding="utf-8")
SKILL = (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")
CASES = json.loads(
    (Path(__file__).parent / "fixtures" / "cost-estimation-cases.json").read_text(
        encoding="utf-8"
    )
)


class CostEstimationContractTest(unittest.TestCase):
    def test_representative_routes_are_covered(self) -> None:
        routes = {case["name"]: case["expected_action"] for case in CASES}
        self.assertEqual(
            routes,
            {
                "managed-sft": "calculate",
                "managed-dpo": "calculate",
                "serverless-sft": "calculate",
                "serverless-dpo": "calculate-with-reference-sampler",
                "dedicated-sft": "not-calculated",
                "dedicated-dpo": "not-calculated",
                "training-api-rl": "contact-training",
                "managed-rft": "excluded",
            },
        )
        self.assertIn("Serverless LoRA SFT or DPO", REFERENCE)
        self.assertIn("reference through a\nsampling client", REFERENCE)
        self.assertIn("Managed RFT | Excluded", REFERENCE)
        self.assertIn("Dedicated SFT or DPO | Do not calculate", REFERENCE)

    def test_formulas_include_run_multipliers_and_serverless_reference_cost(self) -> None:
        self.assertIn("billable_tokens_per_run = rendered_dataset_tokens × epochs", REFERENCE)
        self.assertIn("total_cost = cost_per_run × candidate_runs", REFERENCE)
        self.assertIn(
            "trainer_tokens_per_run = tokens sent to forward or forward_backward × optimizer passes",
            REFERENCE,
        )
        self.assertIn("reference_cost_per_run =", REFERENCE)
        self.assertIn("cache_effective_reference =", REFERENCE)
        self.assertIn("× candidate_runs", REFERENCE)
        self.assertIn("once per unique pair for each candidate run", REFERENCE)
        self.assertNotIn("dpo_saturated_baseline", REFERENCE)
        self.assertNotIn("policy_bundled_rate", REFERENCE)

    def test_output_contract_is_complete(self) -> None:
        for label in (
            "Recommended path",
            "Cost range",
            "Rate certainty",
            "Usage certainty",
            "Supplied inputs",
            "Inferred inputs",
            "Assumptions",
            "Next action",
        ):
            self.assertIn(f"**{label}", REFERENCE)

    def test_skill_routes_to_reference_and_preserves_confirmation(self) -> None:
        self.assertIn("references/cost-estimation.md", SKILL)
        self.assertIn("Do not calculate Dedicated SFT or DPO", SKILL)
        self.assertIn("does not replace the mandatory final-plan", SKILL)

    def test_dedicated_is_not_calculated_and_does_not_expose_private_coefficients(
        self,
    ) -> None:
        self.assertIn("Do not calculate Dedicated SFT or DPO", REFERENCE)
        self.assertIn("Cost range:** `Not calculated`", REFERENCE)
        self.assertIn("docs.fireworks.ai/fine-tuning/cost-estimator", REFERENCE)
        self.assertNotIn("policyEffectiveRatePerM", REFERENCE)
        self.assertNotIn("tokens per second", REFERENCE.lower())
        self.assertNotIn("model flops utilization", REFERENCE.lower())
        self.assertNotIn("policy lower bound", REFERENCE.lower())


if __name__ == "__main__":
    unittest.main()
