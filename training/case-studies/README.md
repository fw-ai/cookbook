# Open Weight Room: training case studies

Small adapters can change how a model behaves. These studies show the training methods, the recorded outcomes, and the tradeoffs behind six examples. Start with a notebook, inspect its evidence, then adapt the method to your own task.

| Study | Behavior | Method |
|---|---|---|
| [Caveman](sft_caveman/README.md) | Teach concise answers without dropping the facts. | Supervised fine-tuning |
| [Humanizer](sft_humanizer/README.md) | Rewrite naturally while preserving details and leaving good writing alone. | Supervised fine-tuning |
| [E-vasion](lipogram_data/README.md) | Answer useful questions without the letter e. | Supervised fine-tuning followed by reinforcement learning |
| [Anti-slop](dpo_antislop/README.md) | See what preference optimization learns, and what a detector score misses. | Direct preference optimization and RL follow-up |
| [Taste](ui_design/README.md) | Compare visual design and functionality across a base model, a skill prompt, SFT and RL. | Recorded comparison with training source |
| [Ask Before Acting](rl_ask_before_acting/README.md) | Train a simulated agent to ask when an action needs clarification. | Reinforcement learning |

## Choose a starting point

Start with **Caveman** for a compact supervised fine-tuning workflow. Use **Humanizer** to study preservation and overediting, **E-vasion** for measurable output constraints, and **Anti-slop** to understand preference optimization and reward tradeoffs. **Taste** explores visual preference versus functionality. **Ask Before Acting** introduces behavior in simulated agent tasks.

## Read the comparisons correctly

Recorded examples are saved model outputs, not new inference. Each study identifies its evaluation conditions; metrics from different datasets or checkpoints are not interchangeable. A shorter response, a higher detector score, or a preferred screenshot is not by itself proof of better task performance.

## Running the notebooks

Use these folders within the Fireworks cookbook and install its training dependencies. Open each notebook from its own directory. Review the data and recorded sections first. Paid training and inference are separate actions requiring your own account and explicit notebook settings. See each study’s guide for its available reproduction path.

`common/` provides display helpers. `rl_lipogram/` preserves shared training code and earlier constraint experiments used by dependent studies.

## Download the data

Each featured study has [one versioned ZIP](bundles/README.md). Its first notebook cell verifies the bundle and restores the original files. Full repository checkouts already include the ZIPs; no API key is needed to unpack them. Training and evaluation splits remain separate inside each bundle.

## Study layout

Each featured folder has one README and one primary notebook. Training, data preparation, and evaluation scripts stay separate so they can be inspected and run independently. Study-specific tests live in `tests/`; shared data and display utilities live in `common/`. Upstream source and license notices remain in `sources/` or `skill/`.

## Offline checks

Restore the relevant data bundle before running study tests. From `training/case-studies/`, run the standard-library bundle checks with:

```bash
python -m unittest discover -s common/tests -p test_study_data.py -v
```

For an individual study, run its tests from that study directory after installing the dependencies its imports require. For example, after restoring Humanizer data and installing matplotlib, run `python -m unittest discover -s tests -p test_walkthrough.py -v`. Other suites use pytest; none of these commands starts paid training.
