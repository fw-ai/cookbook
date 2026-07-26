# Technique: embedding fine-tuning for support search

Teach a retriever which help-center article *governs* a situation, not just which one
sounds like it.

**Is this you?** You run search or RAG over a help center, policy library, or internal
knowledge base. Users describe messy real situations; your retriever returns articles that
are topically adjacent but not the one that actually answers them.

**The customer problem.** A host asks about a guest smoking indoors when the house rules
never mentioned smoking. The governing article is *Safety tips for hosts* -- not whichever
article contains the word "smoking" most often. A general-purpose embedding model has no
way to know that; your policy structure is not in its pretraining.

**The data.** [`profoz/airbnb-policy-scenarios`](https://huggingface.co/datasets/profoz/airbnb-policy-scenarios):
997 scenarios (769 train / 228 test) over 877 Airbnb Help Center articles, one governing
article per scenario.

**The model.** Qwen3-Embedding-8B, full-parameter.

**The technique.** Bidirectional InfoNCE over in-batch negatives via
[`training/recipes/embedding_loop.py`](../../recipes/embedding_loop.py), 72 steps.

**What we'll do.** Measure the base model, fine-tune, re-register the checkpoint as an
embedding model, deploy it, and measure again with the identical harness.

## Results

72 optimizer steps (batch 32, 3 epochs) over 769 pairs. Scored on 228 held-out situations
against the full 877-article corpus.

| Metric | Base | Fine-tuned | Gain |
|---|---|---|---|
| MRR | 0.525 | **0.707** | **+35%** |
| nDCG@10 | 0.602 | 0.771 | +28% |
| nDCG@100 | 0.630 | 0.778 | +23% |
| Recall@10 | 0.868 | 0.974 | +12% |
| Recall@100 | 0.991 | 1.000 | +1% |

The correct article moves from rank ~1.9 to ~1.4 on average. 115 queries improved, 88 were
unchanged (mostly already at rank 1), 25 regressed.

**The gains and losses are asymmetric, which is the interesting part.** The biggest wins
were +137, +108, +79, +60 and +58 rank positions; the worst regressions were -4, -4, -6,
-9 and -11. Fine-tuning rescued queries that were unfindable (rank 60-140) and paid for it
by reshuffling a few within the top 13.

The rescued queries are navigational -- "unsure if their building allows this", "where to
find resources about hosting responsibly", "protection options during their stay". They
share almost no vocabulary with the article that answers them, so lexical similarity was
useless and the base model was lost. The regressions cluster on cancellation and inbound
booking messages, where several sibling articles are legitimately near-equivalent.

**It generalized rather than memorized.** Training loss collapses by an order of magnitude
at the first epoch boundary and `in_batch_recall_at_1` starts touching 1.000, which looks
alarming. It is not: scoring the training situations as a control, fine-tuning lifted MRR
**+0.183 on unseen queries versus +0.133 on seen** -- it helped more on data it had never
trained on. (Base scores higher on train than test even before training, so that split is
simply easier; the train/test gap actually *narrowed*, 0.157 to 0.107.)

## Files

| file | what it is |
|---|---|
| `airbnb_policy_embedding.ipynb` | the case study, runnable top to bottom |
| `retrieval_eval.py` | embed via `/v1/embeddings`, cosine top-k, nDCG / Recall / MRR / MAP |

## Screening your own corpus

Run section 2 against your own data and read the gap between **Recall@100** and **MRR**.
High recall with mediocre MRR is the fine-tunable shape -- the model finds the right
neighborhood and misorders it. If MRR is already high there is nothing to win. If
Recall@100 is low, your problem is coverage (chunking, or a corpus gap) and contrastive
training will not fix it.

Section 1b audits corpus hygiene and, when duplicate golds exist, prices them by scoring
twice: strictly, and with byte-identical twins credited. That gap is the ceiling on what
deduping would buy you.

## Four things that will bite you

**1. The query instruction is asymmetric.** Queries get an `Instruct: ...\nQuery: ` prefix,
documents get nothing. It is baked into the trained weights and nothing re-applies it at
inference. `retrieval_eval.py` reads the template from the recipe at call time, so the
notebook's patch keeps training and evaluation in lockstep.

The recipe ships `"Instruct: {}\nQuery:"` with no trailing space, diverging from the Qwen3
convention; the notebook patches the space back in. Self-consistency is what protects the
measured lift, but absolute numbers shift with it.

**2. Three model ids, three different jobs.**

| id | kind | use |
|---|---|---|
| `accounts/pyroworks/models/qwen3-embedding-8b-ft-base` | tunable base | trainer `base_model` |
| `accounts/fireworks/models/qwen3-embedding-8b` | `EMBEDDING_MODEL` | serverless baseline |
| `accounts/<you>/models/<run>-emb` | `EMBEDDING_MODEL` | your fine-tune, after re-registration |

The recipe's *default* `base_model` is the serving id, which the trainer rejects with
`not of kind base model, but EMBEDDING_MODEL`. Tunable bases live under `pyroworks` (they
are public) and need a validated `POLICY_TRAINER` training shape.

**3. The fine-tune must be re-registered before it serves correct embeddings.** The trainer
promotes as `Kind: HF_BASE_MODEL`, a generative base. Deploying that answers
`/v1/embeddings` on the generative path, which skips the `<|endoftext|>` append and
last-token pooling the model trained with -- the vectors are quietly wrong.

Section 4 downloads the checkpoint, re-creates it as `kind="EMBEDDING_MODEL"`, and deploys
with an **embedding deployment shape**. Both halves are required. You cannot shortcut it:
`models.update(kind=...)` silently no-ops on an existing model. The same is true of
`base_model_details` -- omit it at create time and the model registers with
`api_model_type=""`, and deploying fails with `model is missing model_type`. The notebook
derives those fields from `config.json` the way `firectl` does.

The notebook asserts **input-form invariance** (a raw string embeds identically to its own
token ids) before reporting any metric. If that assertion fires, the deployment is on the
wrong serving path and the numbers are meaningless.

**4. Dedicated deployments need `model#deployment` routing.** A model name alone returns
error 1010 ("not available") even with the deployment `READY`. Requests must use the fully
qualified `accounts/x/models/y#accounts/x/deployments/z`; a bare deployment id also fails.

## Data notes

**Only 769 training pairs**, so `batch_size=32` gives 24 steps per epoch and 3 epochs lands
at 72. At batch 64 you would need twelve passes over the same data to reach 150 steps.

**Article length is bimodal, not long-tailed.** Median 3.3k chars, p75 5.7k, but p95 is 45k
and the longest is 654k -- the giants are topic-index pages that concatenate many articles.
Text is capped at 12k chars (~3k tokens), truncating 11% of documents and discarding ~62%
of total corpus characters. The same cap applies at training and eval.

**The corpus was scraped, and it showed.** The original 1,039 documents contained 38 binary
blobs (PNGs and PDFs decoded as text), 122 redundant duplicates (Terms of Service appeared
three times at 243,800 chars under different URL variants), and nav stubs.

Duplicates were the damaging ones: 20% of eval queries had a gold document with a
byte-identical twin, so the model could retrieve the correct *text* and still be scored
wrong -- an artifact worth **0.035 MRR**. Cleaning the dataset moved the baseline from
0.484 to 0.536 MRR without touching the model.

One subtlety survives: `DOC_CHARS` can *manufacture* duplicates. Two documents that diverge
only after character 19,346 become identical when clipped at 12k. Neither is a gold answer
here, so metrics are unaffected, but the audit runs on the truncated corpus for this reason.

**One relevant article per query**, so MRR is the honest headline; nDCG@10 carries the same
signal through a log discount.

**Provenance is undocumented.** The scenarios read LLM-generated from the articles and the
dataset card is empty. If they were generated *from* the article text there is some leakage
risk -- though a base MRR of 0.536 says the task is not trivially solvable.

## Cost and runtime

Dedicated trainer job and dedicated inference deployment; only the base-model baseline is
serverless. Roughly: 3 min trainer provisioning, ~15 min training, ~40 min to download the
16 GB checkpoint and re-upload it, a few min to deploy, and ~20 s per eval pass. Run the
cleanup cell -- the deployment bills while it exists.

## Related

- [`embedding_retrieval_repro`](../embedding_retrieval_repro/) -- the same recipe applied to
  the four legal/clinical benchmarks from the Fireworks embedding fine-tuning post.
