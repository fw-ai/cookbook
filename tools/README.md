# Cookbook Tools

Standalone scripts for customer-operated Fireworks workflows. These use public
`firectl` and Fireworks APIs only — no Fireworks-internal credentials or GCP
access required.

## Prerequisites

Before running any script in this directory:

| Requirement | Details |
|---|---|
| **Python** | **3.10 or newer** (3.11+ recommended). Check with `python3 --version`. |
| **Python packages** | **None.** Scripts here use only the Python standard library. |
| **firectl** | Install the Fireworks CLI: [firectl docs](https://docs.fireworks.ai/tools-sdks/firectl/firectl). Verify with `firectl deployment list --help`. |
| **API key** | A Fireworks API key with permission to list deployments and read deployment metrics for the target account. |

Set your API key once per shell session:

```bash
export FIREWORKS_API_KEY="fw_..."
```

Optional: point at the dev gateway when working against a dev account:

```bash
export FIREWORKS_API_KEY="fw_..."   # dev-scoped key
python3 list_low_token_deployments.py --account my-account-dev --server gateway-dev.fireworks.ai:443
```

## Download without cloning

```bash
curl -fsSL -o list_low_token_deployments.py \
  https://raw.githubusercontent.com/fw-ai/cookbook/main/tools/list_low_token_deployments.py
chmod +x list_low_token_deployments.py
```

Or browse the directory on GitHub:
https://github.com/fw-ai/cookbook/tree/main/tools

## Scripts

### `list_low_token_deployments.py`

Find deployments whose daily **prompt + generated** token usage stayed below a
threshold over a recent window. Useful for spotting idle or forgotten
deployments that may still hold GPU replicas.

The script:

1. Lists deployments with `firectl deployment list`
2. Fetches per-deployment token rates with `firectl deployment-metrics list`
3. Converts Prometheus rates into daily totals and filters by your threshold

#### Quick start

```bash
export FIREWORKS_API_KEY="fw_..."

# Default: last 7 days, flag deployments whose peak day was < 10 tokens
python3 list_low_token_deployments.py --account <ACCOUNT_ID>
```

#### Common examples

```bash
# Only deployments still holding at least one replica (capacity cleanup candidates)
python3 list_low_token_deployments.py --account <ACCOUNT_ID> --min-replicas 1

# Stricter: every day in the window must be below the threshold
python3 list_low_token_deployments.py --account <ACCOUNT_ID> --mode every-day

# Machine-readable output for automation
python3 list_low_token_deployments.py --account <ACCOUNT_ID> --output json

# Custom lookback and threshold
python3 list_low_token_deployments.py \
  --account <ACCOUNT_ID> \
  --days 14 \
  --threshold 100 \
  --min-replicas 1
```

#### Useful flags

| Flag | Default | Description |
|---|---|---|
| `--account` | *(required)* | Fireworks account ID |
| `--days` | `7` | Lookback window in UTC calendar days |
| `--threshold` | `10` | Token count per day to compare against |
| `--mode` | `max-daily` | `max-daily` (peak day), `every-day`, or `avg-daily` |
| `--min-replicas` | `0` | Only include deployments with at least this many replicas |
| `--output` | `text` | `text` (tab-separated table) or `json` |
| `--firectl` | `firectl` | Path to the `firectl` binary if it is not on `PATH` |
| `--server` | *(prod)* | Gateway host, e.g. `gateway-dev.fireworks.ai:443` for dev accounts |
| `--dry-run` | off | Print the underlying `firectl` commands without running them |

#### Output

**Text mode** prints a table with deployment ID, state, replica count, base
model, and token totals.

**JSON mode** includes metadata (`account`, `days`, `threshold`, `mode`) plus a
`deployments` array with per-day breakdowns in `daily_tokens`.

#### Troubleshooting

| Symptom | Fix |
|---|---|
| `firectl not found on PATH` | Install [firectl](https://docs.fireworks.ai/tools-sdks/firectl/firectl) or pass `--firectl /path/to/firectl` |
| `set FIREWORKS_API_KEY or pass --api-key` | Export `FIREWORKS_API_KEY` or pass `--api-key fw_...` |
| `PermissionDenied` | API key lacks access to that account, or you are on the wrong gateway (`--server`) |
| `python3: command not found` | Install Python 3.10+ and invoke with `python3`, not `python` on older systems |

#### Notes

- Token counts come from **deployment metrics** (inference traffic), not billing
  GPU-hours. Dedicated deployments billed by GPU time can show zero tokens while
  still incurring cost.
- The metrics window uses **UTC calendar days**. Today's usage is included.
- Deployments with no metrics data in the window are treated as **0 tokens/day**.

Related agent skill:
[`skills/research/fireworks-auto-tune/SKILL.md`](../skills/research/fireworks-auto-tune/SKILL.md)
