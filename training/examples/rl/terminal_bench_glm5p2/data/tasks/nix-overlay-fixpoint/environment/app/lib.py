"""
Core library for NixOS-style configuration evaluation.

Provides recursive attribute set merging and overlay application --
the fundamental operations behind the NixOS module system and the
nixpkgs overlay mechanism.
"""
import copy


def recursive_update(base, override):
    """Recursively merge *override* into *base*, producing a new dict.

    Merge semantics (mirrors ``lib.recursiveUpdate`` in nixpkgs):

    * Both values are dicts  -> merge recursively.
    * Both values are lists  -> concatenate (``base ++ override``).
    * Otherwise              -> *override* replaces *base*.

    Neither input dict is mutated.
    """
    result = dict(base)
    for key, value in override.items():
        if key in result:
            if isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = recursive_update(result[key], value)
            else:
                result[key] = value
        else:
            result[key] = value
    return result


def apply_overlays(config, overlays):
    """Apply a chain of overlays to a merged module configuration.

    Each overlay is a callable ``(self_ref, super_ref) -> modifications``:

    * **self_ref** -- the running result, updated after each overlay.
    * **super_ref** -- the original pre-overlay configuration (immutable).

    Overlay modifications are deep-merged into the running result so
    that nested keys added by earlier overlays or modules are preserved.
    """
    result = copy.deepcopy(config)
    for overlay_fn in overlays:
        modifications = overlay_fn(result, config)
        result = {**result, **modifications}
    return result
