"""Training renderer for Nemotron 3.5 Super VL image and text.

The checkpoint uses Ultra's ChatML/reasoning/tool grammar and C-RADIO's
``<img>`` image spans. Training receives the expanded image span; a sampling
request sends one image placeholder and serving adds the delimiters.

``nemotron35_vl`` keeps thinking. ``nemotron35_vl_disable_thinking`` masks
the thinking channel.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

from fireworks.training import sdk
from training.renderer import RendererError, register_renderer
from training.renderer._nemotron3_split import (
    Nemotron3UltraDisableThinkingSplitRenderer,
    Nemotron3UltraSplitRenderer,
)
from training.renderer.plugins import register_renderer_image_capability


class Nemotron35ImageTokenCounter:
    def __init__(self, config):
        self.patch_size = int(config.get("patch_size", 16))
        self.factor = round(1 / float(config.get("downsample_ratio", 0.5)))
        self.minimum = int(config.get("min_num_patches", 1024))
        self.maximum = int(config.get("max_num_patches", 13312))
        self.max_model_len = int(config.get("max_model_len", 16384))

    @classmethod
    def from_pretrained(cls, model_name, **kwargs):
        path = Path(model_name) / "preprocessor_config.json"
        if not path.is_file():
            from huggingface_hub import hf_hub_download

            path = Path(hf_hub_download(model_name, "preprocessor_config.json"))
        config = json.loads(path.read_text())
        return cls(config.get("image_processor", config))

    def get_resize_config(self, media):
        width, height = media["image"].size
        available = max((self.max_model_len - 4) * self.factor**2, self.minimum)
        if self.maximum > 0:
            available = min(available, self.maximum)
        available = max(available, self.minimum)
        h, w = (
            round(height / self.patch_size + 0.5),
            round(width / self.patch_size + 0.5),
        )
        scale = min(math.sqrt(available / (h * w)), 1.0)
        h, w = math.floor(scale * h), math.floor(scale * w)
        if h == 0 or w == 0:
            # Match serving's bounded tile for extreme aspect ratios.
            long_side = (available // self.factor) // self.factor * self.factor
            if h == 0:
                w = min(long_side, max(self.factor, w - w % self.factor))
                h = self.factor
            else:
                h = min(long_side, max(self.factor, h - h % self.factor))
                w = self.factor
            return {"num_tokens": h * w // self.factor**2}
        if available > self.minimum and h * w < self.minimum:
            scale = math.sqrt(self.minimum / (h * w))
            h, w = math.ceil(scale * h), math.ceil(scale * w)
        if h % self.factor:
            inc = self.factor - h % self.factor
            h = h + inc if (h + inc) * w <= available else max(self.factor, h - h % self.factor)
        if w % self.factor:
            inc = self.factor - w % self.factor
            w = w + inc if h * (w + inc) <= available else max(self.factor, w - w % self.factor)
        return {"num_tokens": h * w // self.factor**2}


class _Nemotron35Images:
    def render_message(self, message, ctx):
        rendered = super().render_message(message, ctx)
        if rendered.header and rendered.output and isinstance(rendered.output[0], sdk.types.EncodedTextChunk):
            header = list(rendered.header.tokens)
            first = rendered.output[0]
            text = self.tokenizer.decode(header + list(first.tokens), skip_special_tokens=False)
            combined = self.tokenizer.encode(text, add_special_tokens=False)
            # A leading newline can merge with the header's newline. Preserve
            # the exact template tokenization and train a boundary-spanning
            # token with the message body, as the other BPE renderers do.
            split = 0
            while split < min(len(header), len(combined)) and header[split] == combined[split]:
                split += 1
            rendered = replace(
                rendered,
                header=sdk.types.EncodedTextChunk(tokens=combined[:split]),
                output=[
                    sdk.types.EncodedTextChunk(tokens=combined[split:]),
                    *rendered.output[1:],
                ],
            )
        return rendered

    @property
    def image_placeholder_token_id(self):
        return self.tokenizer.encode("<image>", add_special_tokens=False)[0]

    @property
    def sampler_image_placeholder_affixes(self):
        # These are present in the expanded training input. The sampler adds
        # them itself when expanding each request's single <image> token.
        return (
            self.tokenizer.encode("<img>", add_special_tokens=False),
            self.tokenizer.encode("</img>", add_special_tokens=False),
        )

    def _preprocess_message_parts(self, message, *, strip_thinking=False):
        content = message["content"]
        if isinstance(content, str):
            return [{"type": "text", "text": content}]
        images = [part for part in content if part["type"] == "image"]
        text = "".join(part["text"] for part in content if part["type"] == "text")
        explicit = bool(images) and "<image>" in text
        if explicit and text.count("<image>") != len(images):
            raise RendererError("Nemotron 3.5 image placeholders must match the number of images")
        result = []
        image_index = 0

        def append_image(part, *, numbered):
            nonlocal image_index
            image_index += 1
            prefix = f"<image {image_index}>" if numbered else ""
            result.extend(
                [
                    {"type": "text", "text": prefix + "<img>"},
                    part,
                    {"type": "text", "text": "</img>"},
                ]
            )

        for part in content:
            if part["type"] == "image":
                if not explicit:
                    append_image(part, numbered=len(images) > 1)
            elif part["type"] == "text":
                pieces = part["text"].split("<image>") if explicit else [part["text"]]
                for index, piece in enumerate(pieces):
                    if index:
                        append_image(images[image_index], numbered=False)
                    result.append({"type": "text", "text": piece})
            elif part["type"] == "thinking":
                if not strip_thinking:
                    result.append(
                        {
                            "type": "text",
                            "text": self._format_thinking_text(part["thinking"]),
                        }
                    )
            else:
                raise RendererError(f"Nemotron 3.5 does not support content type {part['type']!r}")
        return result


class Nemotron35Renderer(_Nemotron35Images, Nemotron3UltraSplitRenderer):
    pass


class Nemotron35DisableThinkingRenderer(_Nemotron35Images, Nemotron3UltraDisableThinkingSplitRenderer):
    pass


RENDERER_NAMES = (
    "nemotron35_vl",
    "nemotron35_vl_disable_thinking",
)


register_renderer("nemotron35_vl", lambda tok, ip=None: Nemotron35Renderer(tok, image_processor=ip))
register_renderer(
    "nemotron35_vl_disable_thinking",
    lambda tok, ip=None: Nemotron35DisableThinkingRenderer(tok, image_processor=ip),
)
for _name in RENDERER_NAMES:
    register_renderer_image_capability(
        _name,
        Nemotron35ImageTokenCounter.from_pretrained,
        supports_images=True,
        supports_tool_images=True,
    )
