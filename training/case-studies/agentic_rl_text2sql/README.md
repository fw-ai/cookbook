# Agentic text-to-SQL RL: a tool-calling SQL agent with serverless GRPO

The goal here is to train **Kimi K3** in the ways of being a multi-turn, **tool-calling text-to-SQL agent** with reinforcement
learning on the Fireworks *serverless* training path. The whole RL loop —
**rollout -> reward -> weight update** — lives here:
[`sql_agent_rl_loop.ipynb`](sql_agent_rl_loop.ipynb).

**Is this you?** You have an agent that answers natural-language questions by *navigating* a
set of tools specific to your use-case: e.g. inspect a database schema, look up data, write SQL, read the result,
correct itself. You want that agent to navigate your toolset and workspace more efficiently.

**The idea.**

1. **Give the model tools.** Three tools define the environment:
   `get_database_schema` (to learn about the db), `look_up_evidence` (retrieve SQL examples from a Chroma
   store), and `run_sql_against_database` (execute and return rows). The agent decides the order.
2. **Roll out multi-turn episodes.** Each episode is up to `MAX_TURNS` sampled tool-calling turns
   against a snapshot of the current LoRA adapter, served through the serverless sampler.
3. **Reward on result-set correctness.** `answer_correctness` compares the rows from the agent's
   last `run_sql_against_database` call to the cached gold rows as an order-/duplicate-aware
   multiset: `1.0` exact match, **Jaccard** partial credit otherwise, `0.0` for wrong/no SQL. The
   partial credit gives GRPO a gradient even when the query is close.
4. **GRPO update.** Standardize rewards within each prompt group to advantages, drop zero-spread
   groups (no signal), turn every assistant turn into an importance-sampling datum, then one
   `forward_backward` + `optim_step`.
5. **Evaluate out-of-distribution.** The held-out eval set is drawn only from **databases held
   out of training entirely** (`HOLDOUT_DBS`), so a rising eval curve reflects better tool/DB
   navigation, not schema memorization.

**Why serverless is short.** One `FiretitanServiceClient` pointed at `.../training/v1/serverless`
hands you both a LoRA training client and per-snapshot sampling clients. No trainer job to
provision, no inference deployment, no hot-load, no tunnel — the sequence is
`create_lora_training_client` -> `save_weights_for_sampler` -> `create_sampling_client` ->
`forward_backward` -> `optim_step`.

**The out-of-distribution holdout.** BIRD dev has 11 databases. Training uses 8; the eval set is
drawn only from the 3 unseen ones in `HOLDOUT_DBS`
(`debit_card_specializing`, `california_schools`, `european_football_2`). In a representative run
the held-out (unseen-DB) solve rate climbed **40% -> 78%**, close to the in-distribution result —
evidence the policy learned to navigate new databases rather than memorize schemas. Edit
`HOLDOUT_DBS` in Section 3 to change the split.

## The data (BIRD)

This case study uses the [BIRD](https://bird-bench.github.io/) text-to-SQL `dev` set. The raw
data is **not** shipped (it is large and separately licensed). Download it and place it in this
folder:

- `dev.json` — the BIRD dev questions (question + gold `SQL` + `evidence` + `db_id`).
- `dev_databases/` — the per-database SQLite files (`<db_id>/<db_id>.sqlite`).

Then generate the derived artifacts (Chroma evidence store + cached gold result sets + the
train/holdout JSONL) with:

```bash
python prepare_data.py            # builds chromadb_text2sql/, text2sql_train.jsonl, text2sql_holdout.jsonl
python prepare_data.py --skip-chroma   # rebuild the JSONL rows only
```

`text2sql_train.sample.jsonl` (a tiny slice showing the eval-protocol row format) ships so you can
see the schema without the full dataset; the real generated JSONL, `dev.json`, `dev_databases/`,
and `chromadb_text2sql/` are all gitignored.

## Run

```bash
pip install --pre "fireworks-ai[training]" tinker-cookbook==0.4.1 transformers python-dotenv matplotlib
export FIREWORKS_API_KEY=fw_...
```

Run [`sql_agent_rl_loop.ipynb`](sql_agent_rl_loop.ipynb) top to bottom from inside the cookbook
repo — it registers the Kimi K3 renderer from `training/renderer/kimi_k3.py` and imports the
sibling `sql_tools.py` / `sql_reward.py`. Kimi K3's tokenizer is loaded with
`trust_remote_code=True`, so `transformers` is required.

## Cost

The serverless Kimi K3 GRPO loop spends money: `STEPS=12`, `PROMPT_GROUPS_PER_STEP=4`,
`GROUP_SIZE=16` (~768 training episodes), each a multi-turn rollout against a large MoE, plus a
fixed held-out eval every few steps. Defaults are sized for a visible learning curve, not a
production model. Lower `STEPS`, `PROMPT_GROUPS_PER_STEP`, and `GROUP_SIZE` for a cheaper smoke run.

## Teardown

There is no deployment to delete — the serverless RL session releases itself when the notebook
exits. Local artifacts (`chromadb_text2sql/`, the JSONL files, `dev_databases/`) can be removed at
any time and regenerated with `prepare_data.py`.
