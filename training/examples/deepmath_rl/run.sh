#!/usr/bin/env bash
set -euo pipefail

# Keep the default DeepMath launcher on a validated GRPO shape pair.
# The older qwen3-4b launcher path is not currently runnable for the
# public GRPO flow because it lacks a working policy/reference shape set.

HERE="$(cd "$(dirname "$0")" && pwd)"
exec "$HERE/run_qwen3_30b_a3b.sh" "$@"
