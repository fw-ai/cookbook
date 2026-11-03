"""Offline checks for the lesson's evidence, reader data, and paid-run boundary."""

import ast
import contextlib
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
import types
import os
import importlib.util
from pathlib import Path
from unittest.mock import patch

from train_walkthrough import make_plan, validate_dataset, execute, main
from evaluate_walkthrough import run_evaluation

HERE = Path(__file__).resolve().parent


class LessonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lesson = json.loads((HERE / "walkthrough/recorded.json").read_text())

    def test_evidence_complete_and_pack_bound(self):
        self.assertEqual(len(self.lesson["cases"]), 99)
        self.assertEqual(len({c["id"] for c in self.lesson["cases"]}), 99)
        for case in self.lesson["cases"]:
            self.assertEqual(set(case["arms"]), set(self.lesson["arms"]))
        for split in ("train", "test"):
            actual = hashlib.sha256(
                (HERE / f"pack/humanizer_{split}.jsonl").read_bytes()
            ).hexdigest()
            self.assertEqual(actual, self.lesson["provenance"][f"{split}_sha256"])

    def test_guided_cases_represent_claimed_outcomes(self):
        cases = {c["id"]: c for c in self.lesson["cases"]}
        self.assertTrue(cases["hz-0142"]["arms"]["adapter_s29"]["pass"])
        self.assertEqual(
            cases["hz-0430"]["source"], cases["hz-0430"]["arms"]["adapter_s29"]["text"]
        )
        failure = cases["hz-0233"]["arms"]["adapter_s29"]
        self.assertTrue(failure["preserve"])
        self.assertFalse(failure["pass"])

    def test_export_has_no_account_or_deployment_identifiers(self):
        raw = json.dumps(self.lesson)
        self.assertNotIn("accounts/pyroworks", raw)
        self.assertNotIn("/deployments/", raw)

    def test_invalid_reader_data_has_actionable_errors(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "input.jsonl"
            for content, message in [
                ("[]", "system, user"),
                ('{"messages": [1, 2, 3]}', "object"),
                ("broken", "Line 1"),
                ("", "no training rows"),
            ]:
                p.write_text(content)
                with self.assertRaisesRegex(ValueError, message):
                    validate_dataset(p)
            row = self.lesson["training_example"]
            p.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "repeated input"):
                validate_dataset(p)

    def test_smoke_plan_is_bounded_and_has_no_side_effects(self):
        with tempfile.TemporaryDirectory() as d:
            target = Path(d) / "not-created"
            plan = make_plan(HERE / "pack/humanizer_train.jsonl", "smoke", None, target)
            self.assertEqual(plan["training_rows"], 8)
            self.assertEqual(plan["estimated_steps"], 2)
            self.assertFalse(target.exists())

    def test_cli_presets_use_separate_directories(self):
        plans = []
        for preset in ("smoke", "tutorial"):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                main(["--preset", preset])
            plan, _ = json.JSONDecoder().raw_decode(output.getvalue())
            plans.append(plan)
            self.assertEqual(Path(plan["run_dir"]).name, f"lesson-{preset}")
        self.assertNotEqual(plans[0]["run_dir"], plans[1]["run_dir"])

    def test_paid_training_contract_without_sdk_or_network(self):
        class Config:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        seen = []

        def fake_train(config):
            seen.append(config)
            return {"steps": 2, "job_id": "local-test-job"}

        training = types.ModuleType("training")
        recipes = types.ModuleType("training.recipes")
        utils = types.ModuleType("training.utils")
        recipes.sft_loop = types.SimpleNamespace(Config=Config, main=fake_train)
        utils.RunnerConfig = Config
        modules = {
            "training": training,
            "training.recipes": recipes,
            "training.utils": utils,
        }
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "smoke"
            plan = make_plan(HERE / "pack/humanizer_train.jsonl", "smoke", None, target)
            with patch.dict(
                os.environ, {"FIREWORKS_API_KEY": "local-test-placeholder"}
            ), patch.dict(sys.modules, modules):
                result = execute(plan)
                with self.assertRaisesRegex(ValueError, "not empty"):
                    execute(plan)
            self.assertEqual(len(seen), 1)
            cfg = seen[0]
            self.assertTrue(cfg.serverless)
            self.assertEqual((cfg.max_examples, cfg.epochs, cfg.batch_size), (8, 1, 4))
            self.assertEqual(cfg.output_model_id, None)
            self.assertEqual(Path(cfg.runner.status_file), target.resolve() / "status.json")
            self.assertEqual(
                json.loads((target / "lesson-result.json").read_text()), result
            )

    def test_evaluation_reruns_preserve_evidence(self):
        commands = []

        def fake_run(command, **kwargs):
            commands.append(command)
            output = Path(command[command.index("--out") + 1])
            (output / "samples.jsonl").write_text("retained evidence")

        with tempfile.TemporaryDirectory() as directory, patch(
            "subprocess.run", side_effect=fake_run
        ):
            with contextlib.redirect_stdout(io.StringIO()):
                paths = [
                    run_evaluation(
                        "accounts/example/models/test",
                        Path(directory),
                        split=split,
                        max_inputs=0 if split == "test" else 5,
                        idempotence=split == "test",
                    )
                    for split in ("dev", "test", "test")
                ]
            self.assertEqual(len(set(paths)), 3)
            for path in paths:
                self.assertEqual(
                    (path / "samples.jsonl").read_text(), "retained evidence"
                )
            self.assertTrue(paths[0].name.startswith("eval-dev-"))
            self.assertTrue(paths[1].name.startswith("eval-test-"))
            self.assertIn("--no-idem", commands[0])
            self.assertNotIn("--no-idem", commands[1])

    def test_exporter_rejects_new_experiments_before_writing(self):
        spec = importlib.util.spec_from_file_location(
            "lesson_exporter", HERE / "walkthrough/export_recorded.py"
        )
        exporter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(exporter)
        with tempfile.TemporaryDirectory() as directory:
            sample = Path(directory) / "samples.jsonl"
            output = Path(directory) / "recorded.json"
            sample.write_text('{"arm": "adapter_s29", "id": "hz-0142"}\n')
            output.write_text("existing evidence")
            with self.assertRaisesRegex(
                ValueError, "not the historical lesson snapshot"
            ):
                exporter.export(HERE, sample, output)
            self.assertEqual(output.read_text(), "existing evidence")

    def test_notebook_default_execution_never_runs_paid_commands(self):
        nb = json.loads((HERE / "humanizer_walkthrough.ipynb").read_text())
        scope = {"__name__": "__main__"}
        real_run = subprocess.run
        calls = []

        def local_only(command, **kwargs):
            calls.append(command)
            self.assertNotIn("--execute", command)
            self.assertNotIn("evaluate_pack.py", " ".join(command))
            return real_run(command, **kwargs, capture_output=True, text=True)

        with patch(
            "subprocess.run", side_effect=local_only
        ), contextlib.redirect_stdout(io.StringIO()):
            for cell in nb["cells"]:
                if cell["cell_type"] == "code":
                    source = "".join(cell["source"])
                    ast.parse(source)
                    exec(compile(source, "<notebook>", "exec"), scope)
        self.assertEqual(len(calls), 1)
        self.assertFalse(scope["RUN_TRAINING"])
        self.assertFalse(scope["RUN_EVALUATION"])


if __name__ == "__main__":
    unittest.main()
