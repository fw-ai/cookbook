"""Compatibility imports for checks now owned by ``training.dataset_utils.validation``."""

from training.dataset_utils.validation import ImageRule, image_issue, register_image_rule

__all__ = ["ImageRule", "image_issue", "register_image_rule"]
