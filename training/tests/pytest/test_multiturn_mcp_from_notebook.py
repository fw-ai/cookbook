
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from eval_protocol import evaluation_test
from eval_protocol.models import EvaluationRow, InputMetadata

try:
    from eval_protocol.pytest import MCPGymRolloutProcessor as _RolloutProcessor
except Exception:
    from eval_protocol.pytest import default_mcp_gym_rollout_processor as _RolloutProcessor

DATASET_PATH = Path(r"/Users/sinan/cookbook/training/examples/benchmarks/data/airline_dataset.jsonl")
SERVER_SCRIPT_PATH = Path(r"/Users/sinan/cookbook/training/examples/rl/eval_protocol_chat/remote_server/server.py")

assert DATASET_PATH.exists(), f"Dataset path does not exist: {DATASET_PATH}"
assert SERVER_SCRIPT_PATH.exists(), f"Server path does not exist: {SERVER_SCRIPT_PATH}"

# Ensure eval-protocol's server subprocess ("python ...server.py") resolves to
# this same interpreter environment (where deps are installed).
_shim_dir = Path(__file__).resolve().parent / ".python_shim"
_shim_dir.mkdir(parents=True, exist_ok=True)
_shim_path = _shim_dir / "python"
_shim_path.write_text(
    f"#!/usr/bin/env bash\nexec \"{sys.executable}\" \"$@\"\n",
    encoding="utf-8",
)
os.chmod(_shim_path, 0o755)
os.environ["PATH"] = f"{_shim_dir}:{os.environ.get('PATH', '')}"

def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def dataset_to_evaluation_row(data: List[Dict[str, Any]]) -> List[EvaluationRow]:
    rows: List[EvaluationRow] = []
    for i, row in enumerate(data):
        rows.append(EvaluationRow(
            messages=[{"role": "system", "content": "You are a helpful agent. Use available tools when needed."}],
            input_metadata=InputMetadata(
                row_id=row.get("id", f"row-{i}"),
                dataset_info=dict(row),
            ),
        ))
    return rows

@evaluation_test(
    input_dataset=[str(DATASET_PATH)],
    dataset_adapter=dataset_to_evaluation_row,
    completion_params=[{"model": "fireworks_ai/accounts/fireworks/models/glm-5p2", "temperature": 0.0, "max_tokens": 4096}],
    rollout_processor=_RolloutProcessor() if callable(_RolloutProcessor) else _RolloutProcessor,
    passed_threshold=0.4,
    num_runs=1,
    mode="pointwise",
    max_concurrent_rollouts=16,
    server_script_path=str(SERVER_SCRIPT_PATH),
)
def test_multiturn_mcp(row: EvaluationRow) -> EvaluationRow:
    return row
