"""Export a portable, account-free lesson from a recorded four-arm evaluation.

No API calls. Re-run only when intentionally replacing the lesson's evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARMS = {
    "base_plain": "Base · plain prompt",
    "base_short": "Base · editor instruction",
    "skill": "Base · full skill",
    "adapter_s29": "Adapter · editor instruction",
}
FIELDS = (
    "text",
    "finish",
    "similarity",
    "length_ratio",
    "problems",
    "judge",
    "preserve",
    "pass",
    "over_edit",
    "idem_similarity",
)


# This lesson and its editorial analysis describe one immutable experiment.
EXPECTED_HASHES = {
    "samples_sha256": "6ab1261d8d6066656d5db4053da18b7724860d21ab8124d60d4e4180d664ce5a",
    "test_sha256": "a37d07c7637af1dbda3512d9ac85bd8ef06275fe067627bad314d60561863c2a",
    "train_sha256": "7ee6ca3e500cc268bdd1e277a765ee1554ff87b6e40ecdaf707d704956202f01",
}


def verify_snapshot(study, samples):
    paths = {
        "samples_sha256": samples,
        "test_sha256": study / "pack/humanizer_test.jsonl",
        "train_sha256": study / "pack/humanizer_train.jsonl",
    }
    for key, path in paths.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != EXPECTED_HASHES[key]:
            raise ValueError(
                f"{path.name}: not the historical lesson snapshot. Create a new lesson with reviewed metadata and commentary for a new experiment."
            )


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def export(study, samples, output):
    verify_snapshot(study, samples)
    fixtures = read_rows(study / "pack/humanizer_test.jsonl")
    rows = read_rows(samples)
    by_id = {r["id"]: {} for r in fixtures}
    for row in rows:
        if row["arm"] not in ARMS:
            continue
        if row["id"] not in by_id or row["arm"] in by_id[row["id"]]:
            raise ValueError("Expected exactly one recorded draw per fixture and arm")
        by_id[row["id"]][row["arm"]] = {k: row[k] for k in FIELDS}
    if any(set(arms) != set(ARMS) for arms in by_id.values()):
        raise ValueError("Incomplete four-arm evaluation")
    cases = [
        {
            **{
                k: f[k]
                for k in (
                    "id",
                    "cls",
                    "genre",
                    "audience",
                    "source",
                    "must_preserve",
                    "protected",
                )
            },
            "arms": by_id[f["id"]],
        }
        for f in fixtures
    ]
    training = read_rows(study / "pack/humanizer_train.jsonl")
    example = next(r for r in training if r["meta"]["input_id"] == "hz-0001")
    manifest = json.loads((study / "pack/manifest.json").read_text())
    result = {
        "schema_version": 1,
        "recorded_date": "2026-10-08",
        "seed": 29,
        "base_model": manifest["base_model"],
        "judge_model": manifest["judge_model"],
        "decoding": "One greedy draw per passage; low reasoning effort",
        "provenance": {
            "samples_sha256": hashlib.sha256(samples.read_bytes()).hexdigest(),
            "test_sha256": hashlib.sha256(
                (study / "pack/humanizer_test.jsonl").read_bytes()
            ).hexdigest(),
            "train_sha256": hashlib.sha256(
                (study / "pack/humanizer_train.jsonl").read_bytes()
            ).hexdigest(),
            "note": "Account and deployment identifiers omitted. Text and per-sample scores preserved. Sources are synthetic fictional passages. Historical scores predate the later calibration gate.",
        },
        "arms": ARMS,
        "cases": cases,
        "training_example": example,
        "pack_counts": {
            "train": len(training),
            "dev": len(read_rows(study / "pack/humanizer_dev.jsonl")),
            "test": len(fixtures),
        },
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"Exported {len(cases)} passages × {len(ARMS)} arms to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=HERE.parent)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=HERE / "recorded.json")
    args = parser.parse_args()
    export(args.study_root, args.samples, args.out)
