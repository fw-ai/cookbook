#!/usr/bin/env python3

from __future__ import annotations

import json
import unittest
from pathlib import Path


SKILL_ROOT = Path(__file__).resolve().parents[1]
REFERENCE = (SKILL_ROOT / "references" / "cost-estimation.md").read_text(
    encoding="utf-8"
)
SKILL = (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")
OUTPUT_TEMPLATE = (SKILL_ROOT / "references" / "output-template.md").read_text(
    encoding="utf-8"
)
RUN_STATE = (
    SKILL_ROOT.parent
    / "fireworks-training"
    / "references"
    / "run-state-and-reporting.md"
).read_text(encoding="utf-8")
CASES = json.loads(
    (Path(__file__).parent / "fixtures" / "cost-estimation-cases.json").read_text(
        encoding="utf-8"
    )
)
VECTORS = json.loads(
    (Path(__file__).parent / "fixtures" / "cost-estimation-vectors.json").read_text(
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
                "managed-orpo": "not-calculated",
                "serverless-sft": "calculate",
                "serverless-dpo": "calculate-unpadded-baseline",
                "dedicated-sft": "not-calculated",
                "dedicated-dpo": "not-calculated",
                "vision-sft": "not-calculated",
                "training-api-rl": "contact-training",
                "embedding": "contact-training",
                "igpo": "contact-training",
                "distillation": "contact-training",
                "managed-rft": "excluded",
            },
        )
        self.assertIn("Serverless LoRA SFT", REFERENCE)
        self.assertIn("Serverless LoRA DPO", REFERENCE)
        self.assertIn("reference through a\nsampling client", REFERENCE)
        self.assertIn("Managed RFT | Excluded", REFERENCE)
        self.assertIn("Dedicated SFT or DPO | Do not calculate", REFERENCE)
        self.assertIn("Managed ORPO | Do not calculate", REFERENCE)
        self.assertIn("Embedding, IGPO, or distillation", REFERENCE)
        self.assertIn("Vision SFT or DPO | Do not calculate", REFERENCE)

    def test_formulas_include_run_multipliers_and_serverless_reference_cost(
        self,
    ) -> None:
        self.assertIn(
            "billable_tokens_per_run = rendered_dataset_tokens × epochs", REFERENCE
        )
        self.assertIn("rates are USD per\n1 million billed tokens", REFERENCE)
        self.assertIn("total_cost = cost_per_run × candidate_runs", REFERENCE)
        self.assertIn("planning_range = total_cost × [0.7, 1.3]", REFERENCE)
        self.assertIn(
            "trainer_tokens_per_run = tokens sent to forward or forward_backward × optimizer passes",
            REFERENCE,
        )
        self.assertIn(
            "serverless_sft_planning_range = total_trainer_cost × [0.7, 1.3]",
            REFERENCE,
        )
        self.assertIn("unpadded_baseline_per_run =", REFERENCE)
        self.assertIn("all_uncached_reference_tokens =", REFERENCE)
        self.assertIn("group_by_length=True", REFERENCE)
        self.assertIn("must not be recommended as a Serverless replacement", REFERENCE)
        self.assertIn("longest rendered sequence", REFERENCE)
        self.assertNotIn("cache_effective_reference =", REFERENCE)
        self.assertIn("× candidate_runs", REFERENCE)
        self.assertIn("once per unique pair", REFERENCE)
        self.assertNotIn("dpo_saturated_baseline", REFERENCE)
        self.assertNotIn("policy_bundled_rate", REFERENCE)

    def test_output_contract_is_complete(self) -> None:
        for label in (
            "Recommended path",
            "Estimate type",
            "Cost result",
            "Rate certainty",
            "Usage certainty",
            "Supplied inputs",
            "Inferred inputs",
            "Assumptions",
            "Next action",
        ):
            self.assertIn(f"**{label}", REFERENCE)
        self.assertIn("| Estimate type |", OUTPUT_TEMPLATE)
        self.assertIn("| Cost result |", OUTPUT_TEMPLATE)
        self.assertNotIn("| Cost range |", OUTPUT_TEMPLATE)
        self.assertIn("Unpadded baseline", OUTPUT_TEMPLATE)
        self.assertIn("two decimals below $100", REFERENCE)
        self.assertIn("`<$0.01` for a positive sub-cent result", REFERENCE)
        self.assertIn("<$0.01", OUTPUT_TEMPLATE)

    def test_manifest_uses_canonical_estimated_cost_key(self) -> None:
        self.assertIn("estimated_cost:", REFERENCE)
        self.assertIn("estimated_cost:", RUN_STATE)
        self.assertNotIn("cost_estimate", REFERENCE)

    def test_skill_routes_to_reference_and_preserves_confirmation(self) -> None:
        self.assertIn("references/cost-estimation.md", SKILL)
        self.assertIn("unpadded", SKILL.lower())
        self.assertIn("baseline", SKILL.lower())
        self.assertIn("all-uncached reference", SKILL.lower())
        self.assertIn("Do not calculate Dedicated SFT or DPO", SKILL)
        self.assertIn("Do not calculate Managed ORPO", SKILL)
        self.assertIn("RL, embedding, IGPO, and distillation", SKILL)
        self.assertIn("Do not calculate vision SFT or DPO", SKILL)
        self.assertIn("does not replace the mandatory final-plan", SKILL)

    def test_dedicated_is_not_calculated_and_does_not_expose_private_coefficients(
        self,
    ) -> None:
        self.assertIn("Do not calculate Dedicated SFT or DPO", REFERENCE)
        self.assertIn("Cost result:** `Not calculated`", REFERENCE)
        self.assertIn("docs.fireworks.ai/fine-tuning/cost-estimator", REFERENCE)
        self.assertNotIn("docs.fireworks.ai/fine-tuning/cost-estimator.md", REFERENCE)
        self.assertIn("if the estimator URL is not live", REFERENCE)
        self.assertNotIn("policyEffectiveRatePerM", REFERENCE)
        self.assertNotIn("tokens per second", REFERENCE.lower())
        self.assertNotIn("model flops utilization", REFERENCE.lower())
        self.assertNotIn("policy lower bound", REFERENCE.lower())

    def test_shared_cost_vectors(self) -> None:
        vectors = {vector["name"]: vector for vector in VECTORS}

        sft = vectors["managed-sft"]
        sft_tokens = (
            sft["items"]
            * sft["epochs"]
            * (sft["prompt_tokens"] + sft["response_tokens"])
        )
        sft_center = sft_tokens / 1_000_000 * sft["rate_per_million"]
        self.assertEqual(sft_tokens, sft["expected_tokens"])
        self.assertEqual(sft_center, sft["expected_center"])
        self.assertEqual(sft_center * 0.7, sft["expected_low"])
        self.assertEqual(sft_center * 1.3, sft["expected_high"])

        managed_dpo = vectors["managed-dpo"]
        managed_pair_tokens = (
            2 * managed_dpo["prompt_tokens"]
            + managed_dpo["chosen_tokens"]
            + managed_dpo["rejected_tokens"]
        )
        managed_dpo_tokens = (
            managed_dpo["items"] * managed_dpo["epochs"] * managed_pair_tokens
        )
        managed_dpo_center = (
            managed_dpo_tokens / 1_000_000 * managed_dpo["rate_per_million"]
        )
        self.assertEqual(managed_dpo_tokens, managed_dpo["expected_tokens"])
        self.assertEqual(managed_dpo_center, managed_dpo["expected_center"])
        self.assertAlmostEqual(managed_dpo_center * 0.7, managed_dpo["expected_low"])
        self.assertAlmostEqual(managed_dpo_center * 1.3, managed_dpo["expected_high"])

        serverless_sft = vectors["serverless-sft"]
        serverless_sft_tokens = (
            serverless_sft["items"]
            * serverless_sft["epochs"]
            * (serverless_sft["prompt_tokens"] + serverless_sft["response_tokens"])
        )
        serverless_sft_center = (
            serverless_sft_tokens / 1_000_000 * serverless_sft["train_rate_per_million"]
        )
        self.assertEqual(serverless_sft_tokens, serverless_sft["expected_tokens"])
        self.assertAlmostEqual(serverless_sft_center, serverless_sft["expected_center"])
        self.assertAlmostEqual(
            serverless_sft_center * 0.7, serverless_sft["expected_low"]
        )
        self.assertAlmostEqual(
            serverless_sft_center * 1.3, serverless_sft["expected_high"]
        )

        for name in ("serverless-dpo", "serverless-dpo-repeated-subset"):
            with self.subTest(name=name):
                serverless_dpo = vectors[name]
                pair_tokens = (
                    2 * serverless_dpo["prompt_tokens"]
                    + serverless_dpo["chosen_tokens"]
                    + serverless_dpo["rejected_tokens"]
                )
                policy_tokens = serverless_dpo["policy_pair_occurrences"] * pair_tokens
                reference_tokens = (
                    serverless_dpo["unique_reference_pairs"] * pair_tokens
                )
                baseline = (
                    policy_tokens / 1_000_000 * serverless_dpo["train_rate_per_million"]
                    + reference_tokens
                    / 1_000_000
                    * serverless_dpo["prefill_rate_per_million"]
                )
                self.assertEqual(
                    policy_tokens, serverless_dpo["expected_policy_tokens"]
                )
                self.assertEqual(
                    reference_tokens, serverless_dpo["expected_reference_tokens"]
                )
                self.assertAlmostEqual(
                    baseline, serverless_dpo["expected_unpadded_baseline"]
                )


if __name__ == "__main__":
    unittest.main()
