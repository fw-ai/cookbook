# E2E Resume Tests

End-to-end tests for DCP checkpoint save/load across SFT, GRPO, and DPO recipes.

## Prerequisites

```bash
pip install -e "training[dev]"
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `FIREWORKS_API_KEY` | Yes | Shared dev key: `fw_3ZkNBrXgLw1EJ4y77kqSMBU5` |
| `FIREWORKS_ACCOUNT_ID` | Yes | `pyroworks-dev` |
| `FIREWORKS_BASE_URL` | Yes | `https://dev.api.fireworks.ai` |

## Training Shapes (pyroworks-dev)

| Shape | Nodes | GPUs | Model | Trainer Tag | Region |
|-------|-------|------|-------|-------------|--------|
| `ts-qwen3-30b-a3b-policy` | 1 | 8xB200 | qwen3-30b-a3b | 0.33.0 | US_OHIO_1 |
| `ts-qwen3-30b-a3b-128k-2node` | 2 | 16xB200 | qwen3-30b-a3b | 0.0.0-dev-chengxili-dcp-ci-v3 | US_OHIO_1 |

## Running Tests

### SFT Resume

```bash
FIREWORKS_API_KEY="fw_3ZkNBrXgLw1EJ4y77kqSMBU5" \
FIREWORKS_ACCOUNT_ID="pyroworks-dev" \
FIREWORKS_BASE_URL="https://dev.api.fireworks.ai" \
python -m pytest \
  "training/tests/e2e/test_sft_resume_e2e.py::TestSFTResumeE2E::test_sft_resume_from_checkpoint" \
  -v -s
```

Phase 1 trains SFT on synthetic data (20 examples, 5 steps), saves DCP checkpoints.
Phase 2 uses `init_from_checkpoint` to load DCP from phase 1 and trains 5 more steps.
Verifies model weights are preserved (loss starts low, not from scratch).

Expected runtime: ~15-25 minutes (2x job creation + model download + training).

### GRPO Resume

```bash
FIREWORKS_API_KEY="fw_3ZkNBrXgLw1EJ4y77kqSMBU5" \
FIREWORKS_ACCOUNT_ID="pyroworks-dev" \
FIREWORKS_BASE_URL="https://dev.api.fireworks.ai" \
python -m pytest \
  "training/tests/e2e/test_grpo_resume_e2e.py::TestGRPOResumeE2E::test_grpo_resume_from_checkpoint" \
  -v -s
```

### DPO Resume

```bash
FIREWORKS_API_KEY="fw_3ZkNBrXgLw1EJ4y77kqSMBU5" \
FIREWORKS_ACCOUNT_ID="pyroworks-dev" \
FIREWORKS_BASE_URL="https://dev.api.fireworks.ai" \
python -m pytest \
  "training/tests/e2e/test_dpo_resume_e2e.py::TestDPOResumeE2E::test_dpo_resume_from_checkpoint" \
  -v -s
```

## Cleanup

Tests automatically delete trainer jobs on completion. If a test is interrupted,
clean up stale jobs manually:

```bash
# List running jobs
curl -s "https://dev.api.fireworks.ai/v1/accounts/pyroworks-dev/rlorTrainerJobs" \
  -H "Authorization: Bearer fw_3ZkNBrXgLw1EJ4y77kqSMBU5" | python3 -m json.tool

# Delete a specific job
curl -s -X DELETE \
  "https://dev.api.fireworks.ai/v1/accounts/pyroworks-dev/rlorTrainerJobs/<JOB_ID>?ignoreChecks=true" \
  -H "Authorization: Bearer fw_3ZkNBrXgLw1EJ4y77kqSMBU5"
```
