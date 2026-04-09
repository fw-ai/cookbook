# Token-level GSPO on GSM8K

This example launches token-level GSPO through the shared
`training.recipes.rl_loop` recipe. In the shared loss code,
`policy_loss="gspo"` now implements the sequence-level GSPO objective from
Equation 5/10 of the paper, while `policy_loss="gspo-token"` implements the
token-level stop-gradient variant from Equation 13/17. This example uses the
token-level path.

The example uses the checked-in `seed_gsm8k_sample.jsonl` dataset by default,
an exact-match numeric reward, and the token-level GSPO defaults
`clip_ratio_low=3e-4`, `clip_ratio_high=4e-4`. The example also sets
`Config.kl_beta=0.0` so `rl_loop.py` skips provisioning a reference model for
this lightweight run.

Because `rl_loop.py` currently computes one scalar normalized advantage per
completion, this example primarily demonstrates the dedicated `gspo-token`
dispatch path and defaults. It does not demonstrate token-wise advantage
shaping; that behavior is covered by the unit tests for the shared loss code.

Set your Fireworks credentials, then run:

```bash
export FIREWORKS_API_KEY="..."
export FIREWORKS_ACCOUNT_ID="..."

python train_gspo_token.py \
  --training-shape qwen3-4b-minimum-h200 \
  --deployment-id gspo-token-qwen3-4b-$(date +%s)
```

Override `--dataset-path` and `--max-rows` if you want to point the example at a
larger JSONL dataset, but the default seed file is enough to show the full
`rl_loop.py` plus shared-loss wiring.
