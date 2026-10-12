"""RL policy objectives: one self-contained module per algorithm, no registry.

Every objective module has the same surface, so a fork swaps objectives by
changing one import (``from training.utils.rl.algorithm import cispo as objective``):

- ``<Name>Config``: frozen settings, validated on construction;
- ``LOSS_FN`` and ``loss_fn_config(config)``: the trainer built-in loss, when
  one exists;
- ``build_datums(batch, config)``: datums for that built-in loss;
- ``metrics(batch, result, config)``: ``{PREFIX}_ratio_mean``,
  ``{PREFIX}_clip_frac`` and ``{PREFIX}_kl`` from the returned logprobs;
- ``make_loss_fn(batch, config)``: the same objective as a client loss,
  ``forward_backward_custom(batch.data, make_loss_fn(batch, config))``,
  reporting the same metric names. Client-only modules (``reinforce``,
  ``score_centering``) have no ``LOSS_FN``; their ``build_datums`` returns the
  datums ``make_loss_fn`` runs on;
- ``NORMALIZATION``: the ``grad_accumulation_normalization`` for ``optim_step``.

=====================  ========================  ==========================
Module                 ``LOSS_FN``               Notes
=====================  ========================  ==========================
``grpo``               ``"ppo"``                 metric prefix ``ppo``
``gspo``               ``"gspo"``                sequence ratio; ``num_sequences``
``dapo``               ``"dapo"``                asymmetric / dual clip
``cispo``              ``"cispo"``               detached clipped IS weight
``dro``                ``"dro"``                 quadratic anchor penalty
``dppo``               ``"dppo"``                binary-divergence trust region
``importance_sampling`` ``"importance_sampling"`` unclipped ratio, prefix ``is``
``reinforce``          --                        client loss only
``score_centering``    --                        client loss only, top-k datums
=====================  ========================  ==========================

Two modules are not objectives: ``igpo`` sets ``PolicyBatch.token_advantages``
(turn-level information-gain advantages) for any objective above, and ``sao``
holds the DIS actor loss of the experimental value-head recipe.
"""
