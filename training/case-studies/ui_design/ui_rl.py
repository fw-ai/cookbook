"""GRPO on rendered pages: the lipogram serverless loop with the reward replaced by reward.score.

Score tuple mapping onto the shared loop's (reward, correct, answered, rate):
  reward   = reward.score()["reward"]
  correct  = rendered and functional >= 0.5           (shared metric name "e_free")
  answered = fidelity >= 0.5 (<= 2 unsupported policy claims)   ("answered")
  rate     = design in [-1, 1]                         ("e_rate")
Phase ramp: reference = the brief's best PLAIN page ("ramp") until eval success >= strict_trigger, then the best SKILL
page ("strict"). Success = correct and answered and not truncated. Every component and the judge's evidence strings are
logged per sample in the run dir so reward exploitation can be audited.

    python ui_rl.py --steps 20 --prompt-groups-per-step 8 --group-size 4 --run-dir runs/ui-rl-pilot
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import httpx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "rl_lipogram"))
import reward as R  # noqa: E402
import lipogram_rl as L  # noqa: E402


def _carve_dev(rows, eval_prompt_groups, eval_seed):
    """Validation is the rows marked group_id == DEV (the held-out dev briefs), not a random carve-out."""
    train = [r for r in rows if json.loads(r["ground_truth"])["group_id"] != "DEV"]
    dev = [r for r in rows if json.loads(r["ground_truth"])["group_id"] == "DEV"]
    idx = [i for i, r in enumerate(rows) if json.loads(r["ground_truth"])["group_id"] == "DEV"]
    return train, dev, idx


L._carve_out_eval_rows = _carve_dev


class UIDesignRL(L.ServerlessAntislopRL):
    def __init__(self, cfg: L.Config) -> None:
        super().__init__(cfg)
        self.score_log = (self.run_dir / "scores.jsonl").open("a")

    def _score_one(self, content: str, gt: dict[str, Any]):
        if not content.strip():
            return 0.0, False, False, -1.0
        brief_path = HERE / gt["brief_path"]
        brief = json.loads(brief_path.read_text())
        ref = HERE / (gt["ref_plain"] if self.phase == "ramp" else gt["ref_skill"])
        work = self.run_dir / "work" / gt["source_id"] / str(abs(hash(content)) % 10**10)
        try:
            with httpx.Client() as client:
                s = R.score(content, brief, brief_path, ref, workdir=work, client=client)
        except Exception as e:  # noqa: BLE001  (a harness error is not a verdict)
            print("score error:", str(e)[:160], flush=True)
            return None
        if s.get("rendered") and s.get("design_error"):
            return None
        self.score_log.write(json.dumps({"source_id": gt["source_id"], "phase": self.phase, **{k: v for k, v in s.items() if k != "workdir"}}) + "\n"); self.score_log.flush()
        import shutil
        shutil.rmtree(work / "render" / "dist", ignore_errors=True)   # keep shots and record, drop the build
        correct = bool(s["rendered"]) and s["functional"] >= 0.5
        answered = bool(s["rendered"]) and s["fidelity"] >= 0.5
        return float(s["reward"]), correct, answered, float(s["design"])


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    cfg = L.Config()
    for name, default in (("steps", 20), ("prompt_groups_per_step", 8), ("group_size", 4), ("eval_group_size", 2), ("max_sample_tokens", 10000),
                          ("learning_rate", 2e-5), ("temperature", 1.0), ("score_concurrency", 6), ("prompt_concurrency", 4), ("strict_trigger", 0.5)):
        parser.add_argument("--" + name.replace("_", "-"), type=type(default), default=default)
    parser.add_argument("--run-dir", default="runs/ui-rl-pilot")
    parser.add_argument("--reward-phase", default="auto")
    parser.add_argument("--train-efforts", default="low")
    parser.add_argument("--eval-effort", default="low")
    a = parser.parse_args(argv)
    cfg.dataset = str(HERE / "data/rl_all.jsonl"); cfg.eval_prompt_groups = 1   # any nonzero: the carve-out ignores the count
    for k, v in vars(a).items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    cfg.run_dir = str(HERE / a.run_dir) if hasattr(cfg, "run_dir") else cfg.run_dir
    cfg.max_seq_len = 16384
    print({k: getattr(cfg, k) for k in ("steps", "prompt_groups_per_step", "group_size", "eval_group_size", "max_sample_tokens", "learning_rate", "reward_phase", "train_efforts")})
    UIDesignRL(cfg).run()


if __name__ == "__main__":
    main()
