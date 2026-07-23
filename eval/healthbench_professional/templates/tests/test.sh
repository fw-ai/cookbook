#!/usr/bin/env bash
set -euo pipefail

readonly trajectory_path="/logs/agent/trajectory.json"
readonly answer_path="/workspace/answer.txt"

# Never trust verifier outputs that may have existed before verifier execution.
# Both files are consumed after the run: Harbor reads reward.json, while the
# summary and RL exporter read healthbench_result.json.
/bin/rm -f \
  /logs/verifier/reward.json \
  /logs/verifier/healthbench_result.json

test -s "${trajectory_path}"
test -f "${answer_path}"

# Validate the exact token-in/token-out and aligned-logprob contract before
# spending a judge call. Bind the exact validated trace to both the immutable
# benchmark prompt and the answer that the judge will score.
/usr/local/bin/python - <<'PY'
import json
import sys
from pathlib import Path

sys.path.insert(0, "/tests")
from trajectory import load_atif_trajectory

trajectory_path = Path("/logs/agent/trajectory.json")
source_path = Path("/tests/source.json")
answer_path = Path("/workspace/answer.txt")

exact = load_atif_trajectory(trajectory_path)
trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
source = json.loads(source_path.read_text(encoding="utf-8"))

if trajectory["extra"]["messages"] != source["messages"]:
    raise RuntimeError("trajectory prompt does not match the benchmark source")
if answer_path.read_bytes() != exact.visible_text.encode("utf-8"):
    raise RuntimeError("scored answer does not match trajectory visible_text")
PY

/usr/local/bin/python /tests/judge.py
/usr/local/bin/python -c \
  'import json; from pathlib import Path; reward = json.loads(Path("/logs/verifier/reward.json").read_text()); details = json.loads(Path("/logs/verifier/healthbench_result.json").read_text()); assert set(reward) == {"reward"} and isinstance(reward["reward"], (int, float)) and isinstance(details, dict) and details'
