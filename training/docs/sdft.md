# Self-distillation fine-tuning (SDFT)

SDFT is maintained in the [Fireworks Tinker-Cookbook repository](https://github.com/fw-ai-external/tinker-cookbook).
Follow that repository's README and [SDFT recipe instructions](https://github.com/fw-ai-external/tinker-cookbook/blob/fireworks/tinker_cookbook/recipes/sdft/README.md)
for installation, configuration, supported models and execution. Use its own Python environment
so its dependencies are managed separately from this cookbook.

This cookbook does not include an SDFT loop, vendor copy or SDFT dependency extra.
Its [distillation recipe](../recipes/distillation_loop.py) implements sampled-token
on-policy distillation (OPD); it is a different objective from top-K SDFT.
