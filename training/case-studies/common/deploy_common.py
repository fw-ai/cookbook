"""Pick a validated deployment shape for a model before creating a deployment.

Deployments are created from a deployment shape; passing only an accelerator is
rejected. ``deployment_shape_for`` asks the API which validated shapes this
account can deploy for the model and returns one, so a notebook fails here, with
the options listed, rather than after other paid work has started.
"""

from __future__ import annotations

from typing import Optional

PUBLIC_SHAPE_PREFIX = "accounts/fireworks/deploymentShapes/"


def deployment_shape_for(
    client,
    model: str,
    *,
    accelerator_type: Optional[str] = None,
    enable_addons: bool = False,
) -> str:
    """Return a deployment shape name for ``model``.

    ``model`` may be a base model, a fine-tuned LoRA model or a live-merge model;
    the server resolves it to its base. ``accelerator_type`` is a preference, not a
    requirement: if no validated shape uses it, another validated shape is used and
    the choice is printed. Fireworks-published shapes are preferred over
    account-specific ones. ``enable_addons=True`` only matches shapes that can serve
    LoRA adapters (multi-LoRA).
    """
    match = client.deployment_shape_versions.match_for_model(model, enable_addons=enable_addons)
    shapes = [v.snapshot for v in (match.deployment_shape_versions or []) if getattr(v, "snapshot", None)]
    if not shapes:
        kind = "multi-LoRA (enable_addons) " if enable_addons else ""
        raise RuntimeError(
            f"No validated {kind}deployment shape is available for {model} on this account. "
            "Pick another base model, or ask Fireworks support to enable a shape for it."
        )

    def rank(shape) -> tuple[int, int]:
        public = (shape.name or "").startswith(PUBLIC_SHAPE_PREFIX)
        same_accelerator = accelerator_type is not None and getattr(shape, "accelerator_type", None) == accelerator_type
        return (0 if same_accelerator else 1, 0 if public else 1)

    chosen = min(shapes, key=rank)
    if accelerator_type is not None and getattr(chosen, "accelerator_type", None) != accelerator_type:
        print(
            f"No validated {accelerator_type} shape for {model}; using {chosen.name} "
            f"({getattr(chosen, 'accelerator_type', '?')})."
        )
    return chosen.name
