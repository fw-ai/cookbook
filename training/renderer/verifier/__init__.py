"""Renderer verifier — empirical probe + (forthcoming) L1/L2 layers.

The Phase 0 deliverable here is the empirical probe (``probe.py``):

    python -m training.renderer.verifier render \\
        --renderer glm5 \\
        --tokenizer-model zai-org/GLM-5.1 \\
        --model accounts/fireworks/models/glm-5p1 \\
        --input messages.json \\
        --output probe.json

The probe runs the renderer locally, calls a deployed Fireworks endpoint
to capture what the model actually emits, and writes an audit-table JSON
artifact that pairs the renderer's per-token claim (chunk source, weight)
against the empirical provenance (hard_append vs native_generated). The
artifact is the input human reviewers use to author the YAML spec that
later L1/L2 layers will assert against.

See ``training/renderer/verifier/README.md`` for the JSON schema and authoring
workflow.
"""
