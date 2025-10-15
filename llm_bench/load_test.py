import abc
import argparse
import csv
from dataclasses import dataclass
from functools import partial
import os
import random
import sys
import traceback
from typing import Optional
from locust import HttpUser, task, events, constant_pacing
import copy
import json
import time
import orjson
import base64
import io
import itertools
from PIL import Image
import transformers
import re
import gevent
from locust.util.timespan import parse_timespan as _locust_parse_timespan

try:
    import locust_plugins
except ImportError:
    print("locust-plugins is not installed, Grafana won't work")


def add_custom_metric(name, value, length_value=0):
    events.request.fire(
        request_type="METRIC",
        name=name,
        response_time=value,
        response_length=length_value,
        exception=None,
        context=None,
    )


PROMPT_CHAT_IMAGE_PLACEHOLDER = "<image>"


class LimericsDataset:
    _PROMPT = "\n\nTranslate the limericks above to Spanish, then re-write limericks using different styles. Do it 10 times."

    def __init__(
        self,
        path: str,
        tokenizer_path: str,
        chat: bool,
        num_tokens: int,
        common_tokens: int,
    ):
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)
        self._num_tokens = num_tokens

        self._all_limericks = []
        with open(path, "r") as f:
            text = f.read()
            lims = text.split("\n\n")
            for i, lim in enumerate(lims):
                num_tokens = len(self._tokenizer.encode(lim))
                self._all_limericks.append((lim, num_tokens))

        self._prefix = ""
        self._suffix = self._PROMPT
        self._prefix_suffix_tokens = len(self._tokenizer.encode(self._PROMPT))
        while self._prefix_suffix_tokens < common_tokens:
            lim, num_tokens = self._all_limericks[
                random.randint(0, len(self._all_limericks) - 1)
            ]
            self._prefix += lim + "\n\n"
            self._prefix_suffix_tokens += num_tokens

        if chat:
            empty_tempalate_tokens = self._tokenizer.apply_chat_template(
                [{"role": "user", "content": ""}],
                tokenize=True,
                add_generation_prompt=True,
            )
            self._prefix_suffix_tokens += len(empty_tempalate_tokens)

    def __next__(self):
        prompt_tokens = self._prefix_suffix_tokens
        prompt = self._prefix
        while prompt_tokens < self._num_tokens:
            lim, num_tokens = self._all_limericks[
                random.randint(0, len(self._all_limericks) - 1)
            ]

            prompt += lim + "\n\n"
            prompt_tokens += num_tokens
        prompt += self._suffix

        return prompt, prompt_tokens

    def __iter__(self):
        return self


class JsonlDataset:
    def __init__(self, path: str):
        self.path = path

    def __iter__(self):
        return itertools.cycle(self._read_data())

    def _read_data(self):
        with open(self.path, "r") as f:
            for line in f:
                yield json.loads(line), 0


class DatasetHolder:
    _instance = None

    @classmethod
    def _create_dataset(cls, options: argparse.Namespace):
        if options.dataset.startswith("@"):
            return JsonlDataset(options.dataset[1:])
        elif options.dataset == "limerics":
            assert (
                options.tokenizer is not None
            ), "--tokenizer is required for limerics dataset"
            return LimericsDataset(
                path=os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "limericks.txt"
                ),
                tokenizer_path=options.tokenizer,
                chat=options.chat,
                num_tokens=options.prompt_tokens,
                common_tokens=options.prompt_cache_max_len,
            )
        else:
            raise ValueError(f"Unknown dataset: {options.dataset}")

    @classmethod
    def get_instance(cls, options: argparse.Namespace):
        if cls._instance is None:
            cls._instance = cls._create_dataset(options)
        return cls._instance


class FixedQPSPacer:
    _instance = None

    def __init__(self, qps, distribution):
        self.qps = qps
        self.distribution = distribution

        # It's kind of thread safe thanks to GIL as the only state is `t` - good enough for a loadtest
        def gen():
            t = time.time()
            mean_wait = 1 / self.qps
            while True:
                if self.distribution == "exponential":
                    wait = random.expovariate(1 / mean_wait)
                elif self.distribution == "uniform":
                    wait = random.uniform(0, 2 * mean_wait)
                elif self.distribution == "constant":
                    wait = mean_wait
                else:
                    print("Unknown distribution {self.distribution}")
                    os._exit(1)
                t += wait
                yield t

        self.iterator = gen()

    @classmethod
    def instance(cls, qps, distribution):
        if cls._instance is None:
            cls._instance = cls(qps, distribution)
        else:
            assert cls._instance.qps == qps
            assert cls._instance.distribution == distribution
        return cls._instance

    def wait_time_till_next(self):
        t = next(self.iterator)
        now = time.time()
        if now > t:
            print(
                f"WARNING: not enough locust users to keep up with the desired QPS. Either the number of locust users is too low or the server is overloaded. Delay: {now-t:.3f}s"
            )
            return 0
        return t - now


class LengthSampler:
    def __init__(self, distribution: str, mean: int, cap: Optional[int], alpha: float):
        self.distribution = distribution
        self.mean = mean
        self.cap = cap
        self.alpha = alpha

        if self.distribution == "exponential":
            self.sample_func = lambda: int(random.expovariate(1 / self.mean))
        elif self.distribution == "uniform":
            mx = self.mean + int(self.alpha * self.mean)
            if self.cap is not None:
                mx = min(mx, self.cap)
            self.sample_func = lambda: random.randint(
                max(1, self.mean - int(self.alpha * self.mean)), mx
            )
        elif self.distribution == "constant":
            self.sample_func = lambda: self.mean
        elif self.distribution == "normal":
            self.sample_func = lambda: int(
                random.gauss(self.mean, self.mean * self.alpha)
            )
        else:
            raise ValueError(f"Unknown distribution {self.distribution}")

    def sample(self) -> int:
        for _ in range(1000):
            sample = self.sample_func()
            if sample <= 0:
                continue
            if self.cap is not None and sample > self.cap:
                continue
            return sample
        else:
            raise ValueError(
                "Can't sample a value after 1000 attempts, check distribution parameters"
            )

    def __str__(self):
        r = int(self.mean * self.alpha)
        if self.distribution == "constant":
            s = str(self.mean)
        elif self.distribution == "uniform":
            s = f"uniform({self.mean} +/- {r})"
        elif self.distribution == "normal":
            s = f"normal({self.mean}, {r})"
        elif self.distribution == "exponential":
            s = f"exponential({self.mean})"
        else:
            assert False
        if self.cap is not None:
            s += f" capped at {self.cap}"
        return s


class InitTracker:
    users = None
    first_request_done = 0
    logging_params = None
    environment = None
    tokenizer = None
    deferred_run_time_seconds = None
    stop_scheduled = False
    stats_reset_done = False

    @classmethod
    def notify_init(cls, environment, logging_params):
        if cls.environment is None:
            cls.environment = environment
        if cls.logging_params is None:
            cls.logging_params = logging_params
        else:
            assert (
                cls.logging_params == logging_params
            ), f"Inconsistent settings between workers: {cls.logging_params} != {logging_params}"

    @classmethod
    def notify_first_request(cls):
        cls.first_request_done += 1

    @classmethod
    def notify_spawning_complete(cls, user_count):
        cls.users = user_count
        # Start steady-state measurement exactly when all users have spawned
        if not cls.stats_reset_done:
            cls.reset_stats()
            cls.stats_reset_done = True
        # If -t/--run-time was provided, schedule test stop relative to spawn complete
        if (
            cls.deferred_run_time_seconds is not None
            and not cls.stop_scheduled
            and cls.environment is not None
            and cls.environment.runner is not None
        ):
            delay = float(cls.deferred_run_time_seconds)
            print(f"Scheduling stop {delay}s after spawning complete (deferred -t)")
            gevent.spawn_later(delay, cls.environment.runner.quit)
            cls.stop_scheduled = True

    @classmethod
    def reset_stats(cls):
        assert cls.environment.runner, "only local mode is supported"
        print("Resetting stats after traffic reach a steady state")
        cls.environment.events.reset_stats.fire()
        cls.environment.runner.stats.reset_all()

    @classmethod
    def load_tokenizer(cls, dir):
        if not dir:
            return None
        if cls.tokenizer:
            return cls.tokenizer
        import transformers

        cls.tokenizer = transformers.AutoTokenizer.from_pretrained(dir)
        cls.tokenizer.add_bos_token = False
        cls.tokenizer.add_eos_token = False
        return cls.tokenizer


events.spawning_complete.add_listener(InitTracker.notify_spawning_complete)


def _parse_run_time_to_seconds(run_time_value):
    """Parse Locust -t/--run-time value into seconds (float). Supports both
    already-parsed numeric values and human strings like '30s', '5m', '1h30m'.
    """
    if not run_time_value:
        return None
    # If Locust already parsed it to a number (seconds), just use it
    if isinstance(run_time_value, (int, float)):
        return float(run_time_value)
    # Try Locust's own parser first
    if _locust_parse_timespan is not None:
        try:
            return float(_locust_parse_timespan(run_time_value))
        except Exception:
            pass
    # Fallback simple parser for strings like '1h30m15s'
    s = str(run_time_value).strip().lower()
    total = 0.0
    for value, unit in re.findall(r"(\d+)\s*([smhd])", s):
        n = float(value)
        if unit == "s":
            total += n
        elif unit == "m":
            total += n * 60
        elif unit == "h":
            total += n * 3600
        elif unit == "d":
            total += n * 86400
    if total == 0.0:
        raise ValueError(f"Unable to parse run time value: {run_time_value}")
    return total


@events.init.add_listener
def _defer_run_time_to_after_spawn(environment, **_kwargs):
    """Capture -t/--run-time and defer it to start counting after spawn completes.

    We store the desired duration, null out the original option to prevent
    Locust from scheduling an early stop, and then schedule our own stop in
    InitTracker.notify_spawning_complete.
    """
    try:
        run_time_value = getattr(environment.parsed_options, "run_time", None)
    except Exception:
        run_time_value = None
    seconds = _parse_run_time_to_seconds(run_time_value) if run_time_value else None
    if seconds:
        # Disable Locust's default run_time handling by clearing it
        try:
            environment.parsed_options.run_time = None
        except Exception:
            pass
        InitTracker.deferred_run_time_seconds = seconds
        InitTracker.environment = environment
        print(
            f"Deferring -t/--run-time to start after spawning complete: {seconds}s"
        )


@dataclass
class ChunkMetadata:
    text: str
    logprob_tokens: Optional[int]
    usage_tokens: Optional[int]
    prompt_usage_tokens: Optional[int]


class BaseProvider(abc.ABC):
    DEFAULT_MODEL_NAME = None

    def __init__(self, model, parsed_options):
        self.model = model
        self.parsed_options = parsed_options

    @abc.abstractmethod
    def get_url(self): ...

    @abc.abstractmethod
    def format_payload(self, prompt, max_tokens, images): ...

    @abc.abstractmethod
    def parse_output_json(self, json): ...


class OpenAIProvider(BaseProvider):
    def get_url(self):
        if self.parsed_options.embeddings:
            return "/v1/embeddings"
        elif self.parsed_options.chat:
            return "/v1/chat/completions"
        else:
            return "/v1/completions"

    def format_payload(self, prompt, max_tokens, images):
        if self.parsed_options.embeddings:
            data = {
                "model": self.model,
                "input": prompt,
            }
            return data

        data = {
            "model": self.model,
            "max_tokens": max_tokens,
            "stream": self.parsed_options.stream,
            "temperature": self.parsed_options.temperature,
            "n": self.parsed_options.n,
        }
        if self.parsed_options.top_k is not None:
            data["top_k"] = self.parsed_options.top_k
        if self.parsed_options.logprobs is not None:
            data["logprobs"] = self.parsed_options.logprobs
        if isinstance(prompt, str):
            if self.parsed_options.chat:
                if images is None:
                    data["messages"] = [{"role": "user", "content": prompt}]
                else:
                    image_urls = []
                    for image in images:
                        image_urls.append(
                            {"type": "image_url", "image_url": {"url": image}}
                        )
                    data["messages"] = [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": prompt}, *image_urls],
                        }
                    ]
            else:
                data["prompt"] = prompt
                if images is not None:
                    data["images"] = images
        else:
            assert isinstance(prompt, dict), "prompt must be a dict"
            for k, v in prompt.items():
                data[k] = v

        return data

    def parse_output_json(self, data):
        if self.parsed_options.embeddings:
            return ChunkMetadata(
                text="",
                logprob_tokens=None,
                usage_tokens=None,
                prompt_usage_tokens=None,
            )
        usage = data.get("usage", None)

        assert len(data["choices"]) == 1, f"Too many choices {len(data['choices'])}"
        choice = data["choices"][0]
        if self.parsed_options.chat:
            if self.parsed_options.stream:
                block = choice["delta"]
            else:
                block = choice["message"]
            text = (block.get("reasoning", "") or "") + (block.get("reasoning_content", "") or "") + (block.get("content", "") or "")
        else:
            text = choice["text"]

        logprobs = choice.get("logprobs", None)
        if logprobs and "tokens" in logprobs:
            logprob_tokens = len(logprobs["tokens"])
        else:
            logprob_tokens = None

        return ChunkMetadata(
            text=text,
            logprob_tokens=logprob_tokens,
            usage_tokens=usage["completion_tokens"] if usage else None,
            prompt_usage_tokens=usage.get("prompt_tokens", None) if usage else None,
        )


class FireworksProvider(OpenAIProvider):
    def format_payload(self, prompt, max_tokens, images):
        data = super().format_payload(prompt, max_tokens, images)
        if not self.parsed_options.embeddings:
            data["min_tokens"] = max_tokens
        data["prompt_cache_max_len"] = self.parsed_options.prompt_cache_max_len
        return data


class VllmProvider(OpenAIProvider):
    def format_payload(self, prompt, max_tokens, images):
        data = super().format_payload(prompt, max_tokens, images)
        data["ignore_eos"] = True
        return data


class TogetherProvider(OpenAIProvider):
    def get_url(self):
        assert not self.parsed_options.chat, "Chat is not supported"
        return "/"

    def format_payload(self, prompt, max_tokens, images):
        data = super().format_payload(prompt, max_tokens, images)
        data["ignore_eos"] = True
        data["stream_tokens"] = data.pop("stream")
        return data

    def parse_output_json(self, data):
        if not self.parsed_options.stream:
            data = data["output"]
        return super().parse_output_json(data)


class TgiProvider(BaseProvider):
    DEFAULT_MODEL_NAME = "<unused>"

    def get_url(self):
        assert self.parsed_options.n == 1, "n > 1 is not supported"
        assert not self.parsed_options.chat, "Chat is not supported"
        stream_suffix = "_stream" if self.parsed_options.stream else ""
        return f"/generate{stream_suffix}"

    def format_payload(self, prompt, max_tokens, images):
        assert isinstance(prompt, str), "prompt must be a string"
        assert images is None, "images are not supported"
        data = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": self.parsed_options.temperature,
                "top_n_tokens": self.parsed_options.logprobs,
                "details": self.parsed_options.logprobs is not None,
            },
        }
        return data

    def parse_output_json(self, data):
        if "token" in data:
            # streaming chunk
            return ChunkMetadata(
                text=data["token"]["text"],
                logprob_tokens=1,
                usage_tokens=None,
                prompt_usage_tokens=None,
            )
        else:
            # non-streaming response
            return ChunkMetadata(
                text=data["generated_text"],
                logprob_tokens=(
                    len(data["details"]["tokens"]) if "details" in data else None
                ),
                usage_tokens=(
                    data["details"]["generated_tokens"] if "details" in data else None
                ),
                prompt_usage_tokens=None,
            )


PROVIDER_CLASS_MAP = {
    "fireworks": FireworksProvider,
    "vllm": VllmProvider,
    "sglang": VllmProvider,
    "openai": OpenAIProvider,
    "together": TogetherProvider,
    "tgi": TgiProvider,
}


def _load_curl_like_data(text):
    """
    Either use the passed string or load from a file if the string is `@filename`
    """
    if text.startswith("@"):
        try:
            if text.endswith(".jsonl"):
                with open(text[1:], "r") as f:
                    return [json.loads(line) for line in f]
            else:
                with open(text[1:], "r") as f:
                    return f.read()
        except Exception as e:
            raise ValueError(f"Failed to read file {text[1:]}") from e
    else:
        return text


class LLMUser(HttpUser):
    # no wait time, so every user creates a continuous load, sending requests as quickly as possible

    def on_start(self):
        try:
            self._on_start()
        except Exception as e:
            print(f"Failed to initialize: {repr(e)}")
            print(traceback.format_exc())
            sys.exit(1)

    def _guess_provider(self):
        self.model = self.environment.parsed_options.model
        self.provider = self.environment.parsed_options.provider
        # guess based on URL
        if self.provider is None:
            if "fireworks.ai" in self.host:
                self.provider = "fireworks"
            elif "together" in self.host:
                self.provider = "together"
            elif "openai" in self.host:
                self.provider = "openai"

        if (
            self.model is None
            and self.provider is not None
            and PROVIDER_CLASS_MAP[self.provider].DEFAULT_MODEL_NAME is not None
        ):
            self.model = PROVIDER_CLASS_MAP[self.provider].DEFAULT_MODEL_NAME

        if self.model and self.provider:
            return

        # vllm doesn't support /model/<name> endpoint, so iterate over all models
        try:
            resp = self.client.get("/v1/models")
            resp.raise_for_status()
            resp = resp.json()
        except Exception as e:
            raise ValueError(
                "Argument --model or --provider was not specified and /v1/models failed"
            ) from e

        models = resp["data"]
        assert len(models) > 0, "No models found in /v1/models"
        owned_by = None
        # pick the first model
        for m in models:
            if self.model is None or m["id"] == self.model:
                self.model = m["id"]
                owned_by = m["owned_by"]
                break
        if self.provider is None:
            if not owned_by:
                raise ValueError(
                    f"Model {self.model} not found in /v1/models. Specify --provider explicitly"
                )
            if owned_by in PROVIDER_CLASS_MAP:
                self.provider = owned_by
            else:
                raise ValueError(
                    f"Can't detect provider, specify it explicitly with --provider, owned_by={owned_by}"
                )

    def _on_start(self):
        self.client.headers["Content-Type"] = "application/json"
        if self.environment.parsed_options.api_key:
            self.client.headers["Authorization"] = (
                "Bearer " + self.environment.parsed_options.api_key
            )
        if self.environment.parsed_options.header:
            for header in self.environment.parsed_options.header:
                key, val = header.split(":", 1)
                self.client.headers[key] = val
        self._guess_provider()
        print(f" Provider {self.provider} using model {self.model} ".center(80, "*"))
        self.provider_formatter = PROVIDER_CLASS_MAP[self.provider](
            self.model, self.environment.parsed_options
        )

        self.stream = self.environment.parsed_options.stream

        image_resolutions = (
            self.environment.parsed_options.prompt_images_with_resolutions
        )
        self.prompt_images = None
        if image_resolutions:
            if not self.environment.parsed_options.chat:
                # Using regular /completions endpoint, each model has it's own image placeholder
                # e.g., <|image|> for Phi, <|image_pad|> for Qwen, <image> for Llava
                # So using /completions endpoint requires a bit more work to support this
                raise AssertionError(
                    "--prompt-images-with-resolutions is only supported with --chat mode."
                )
            self.prompt_images = [
                self._create_base64_image(width, height)
                for width, height in image_resolutions
            ]

        self.max_tokens_sampler = LengthSampler(
            distribution=self.environment.parsed_options.max_tokens_distribution,
            mean=self.environment.parsed_options.max_tokens,
            cap=self.environment.parsed_options.max_tokens_cap,
            alpha=self.environment.parsed_options.max_tokens_range,
        )
        self.temperature = self.environment.parsed_options.temperature

        logging_params = {
            # TODO: add some server info with git version
            "provider": self.provider,
            "model": self.model,
            "prompt_tokens": self.environment.parsed_options.prompt_tokens,  # might be overwritten based on metric
            "generation_tokens": str(self.max_tokens_sampler),
            "stream": self.stream,
            "temperature": self.temperature,
            "logprobs": self.environment.parsed_options.logprobs,
        }

        if self.environment.parsed_options.top_k is not None:
            logging_params["top_k"] = self.environment.parsed_options.top_k

        InitTracker.notify_init(self.environment, logging_params)

        if self.environment.parsed_options.qps is not None:
            if self.environment.parsed_options.burst:
                raise ValueError("Burst and QPS modes are mutually exclusive")
            pacer = FixedQPSPacer.instance(
                self.environment.parsed_options.qps,
                self.environment.parsed_options.qps_distribution,
            )
            # it will be called by Locust after each task
            self.wait_time = pacer.wait_time_till_next
            self.wait()
        elif self.environment.parsed_options.burst:
            self.wait_time = partial(
                constant_pacing(self.environment.parsed_options.burst), self
            )
        else:
            # introduce initial delay to avoid all users hitting the service at the same time
            time.sleep(random.random())

        self.first_done = False

        dataset = DatasetHolder.get_instance(self.environment.parsed_options)
        self.dataset = iter(dataset)

    def _create_base64_image(self, width, height):
        """Create a random RGB image with the given dimensions and return as base64 data URI."""
        img = Image.new("RGB", (width, height))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{img_str}"

    def _get_input(self):
        prompt, prompt_tokens = next(self.dataset)

        if self.prompt_images:
            images = self.prompt_images
            prompt_images_positioning = (
                self.environment.parsed_options.prompt_images_positioning
            )
            prompt = self.insert_image_placeholders(
                prompt, len(images), prompt_images_positioning
            )
        else:
            images = None

        return prompt, prompt_tokens, images

    def insert_image_placeholders(self, prompt, num_images, prompt_images_positioning):
        if num_images <= 0:
            return prompt

        prompt_length = len(prompt)
        if prompt_length == 0:
            return PROMPT_CHAT_IMAGE_PLACEHOLDER * num_images

        if prompt_images_positioning == "space-evenly":
            """
            Insert <image> placeholders evenly throughout the prompt.
            E.g., for 3 images, a prompt "abcdefgh" is changed to "ab<image>cd<image>ef<image>gh"

            Images are spaced out evenly based on on character length.
            This may result in a few extra tokens if the image tags are placed in the middle of tokens.
            But shouldn't affect results meaningfully.
            """
            # we need num_images + 1 segments to place between <image> tags
            segment_length = prompt_length / (num_images + 1)
            result = ""
            for i in range(num_images):
                # Move a sliding window of segment_length across the prompt
                # Truncating to ensure all segments are non-overlapping
                # If segment_end is truncated, that character will be included in the next segment
                segment_start = int(i * segment_length)
                segment_end = int((i + 1) * segment_length)
                result += (
                    prompt[segment_start:segment_end] + PROMPT_CHAT_IMAGE_PLACEHOLDER
                )

            # Final segment
            result += prompt[int(num_images * segment_length) :]

            return result
        elif prompt_images_positioning == "end":
            return prompt + PROMPT_CHAT_IMAGE_PLACEHOLDER * num_images
        else:
            raise ValueError(
                f"Invalid prompt images positioning: {prompt_images_positioning}"
            )

    @task
    def generate_text(self):
        max_tokens = self.max_tokens_sampler.sample()
        prompt, prompt_usage_tokens, images = self._get_input()
        data = self.provider_formatter.format_payload(prompt, max_tokens, images)
        t_start = time.perf_counter()

        with self.client.post(
            self.provider_formatter.get_url(),
            data=json.dumps(data),
            stream=True,
            catch_response=True,
        ) as response:
            combined_text = ""
            done_empty_chunk = False
            done = False
            total_usage_tokens = None
            total_logprob_tokens = None
            try:
                response.raise_for_status()
            except Exception as e:
                raise RuntimeError(f"Error in response: {response.text}") from e
            t_first_token = None
            for chunk in response.iter_lines(delimiter=b"\n\n"):
                if len(chunk) == 0:
                    continue  # come providers send empty lines between data chunks
                if done:
                    if chunk != b"data: [DONE]":
                        print(f"WARNING: Received more chunks after [DONE]: {chunk}")
                try:
                    now = time.perf_counter()
                    if self.provider_formatter.parsed_options.embeddings:
                        t_first_token = now
                        break
                    if self.stream:
                        assert chunk.startswith(
                            b"data:"
                        ), f"Unexpected chunk not starting with 'data': {chunk}"
                        chunk = chunk[len(b"data:") :]
                        if chunk.strip() == b"[DONE]":
                            done = True
                            continue
                    if done_empty_chunk:
                        print(f"WARNING: Received more chunks after the trailing last chunk: {chunk}")
                    data = orjson.loads(chunk)
                    if not data.get("choices"):
                        done_empty_chunk = True
                        continue
                    out = self.provider_formatter.parse_output_json(data)
                    if out.usage_tokens:
                        total_usage_tokens = out.usage_tokens
                    if out.prompt_usage_tokens:
                        prompt_usage_tokens = out.prompt_usage_tokens
                    combined_text += out.text

                    # some providers (SGLang) send an empty chunk first skewing the TTFT
                    if combined_text and t_first_token is None:
                        t_first_token = now

                    if out.logprob_tokens:
                        total_logprob_tokens = (
                            total_logprob_tokens or 0
                        ) + out.logprob_tokens
                except Exception as e:
                    print(f"Failed to parse response: {chunk} with error {repr(e)}")
                    response.failure(e)
                    return
            assert t_first_token is not None, "empty response received"
            if (
                (total_logprob_tokens is not None)
                and (total_usage_tokens is not None)
                and total_logprob_tokens != total_usage_tokens
            ):
                print(
                    f"WARNING: usage_tokens {total_usage_tokens} != logprob_tokens {total_logprob_tokens}"
                )
            if total_logprob_tokens is not None:
                num_tokens = total_logprob_tokens
            else:
                num_tokens = total_usage_tokens

            num_tokens = num_tokens or 0
            num_chars = len(combined_text)
            now = time.perf_counter()
            dur_total = now - t_start
            dur_generation = now - t_first_token
            dur_first_token = t_first_token - t_start
            print(
                f"Response received: total {dur_total*1000:.2f} ms, first token {dur_first_token*1000:.2f} ms, {num_chars} chars, {num_tokens} tokens"
            )
            if self.environment.parsed_options.show_response:
                print("---")
                print(combined_text)
                print("---")
            if num_chars:
                add_custom_metric(
                    "latency_per_char", dur_generation / num_chars * 1000, num_chars
                )
            if self.stream:
                add_custom_metric("time_to_first_token", dur_first_token * 1000)
            add_custom_metric("total_latency", dur_total * 1000)
            if num_tokens:
                if num_tokens != max_tokens:
                    print(
                        f"WARNING: wrong number of tokens: {num_tokens}, expected {max_tokens}"
                    )
                add_custom_metric("num_tokens", num_tokens)
                add_custom_metric(
                    "latency_per_token", dur_generation / num_tokens * 1000, num_tokens
                )
                add_custom_metric(
                    "overall_latency_per_token",
                    dur_total / num_tokens * 1000,
                    num_tokens,
                )
            prompt_tokens = prompt_usage_tokens or self.prompt_tokenizer_tokens
            if prompt_tokens:
                add_custom_metric("prompt_tokens", prompt_tokens)

            if not self.first_done:
                self.first_done = True
                InitTracker.notify_first_request()


def parse_resolution(res_str):
    """Parse a resolution string like '3084x1080' into a tuple of integers (width, height)."""
    try:
        width, height = map(int, res_str.split("x"))
        return (width, height)
    except (ValueError, AttributeError):
        raise argparse.ArgumentTypeError(
            f"Invalid resolution format: {res_str}. Expected format: WIDTHxHEIGHT (e.g. 1024x1024)"
        )


@events.init_command_line_parser.add_listener
def init_parser(parser):
    parser.add_argument(
        "--provider",
        choices=list(PROVIDER_CLASS_MAP.keys()),
        type=str,
        help="Which flavor of API to use. If not specified, we'll try to guess based on the URL and /v1/models output",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        env_var="DATASET",
        type=str,
        help="Either 'limerics' or a path to a JSONL file",
        default="limerics",
    )
    parser.add_argument(
        "-m",
        "--model",
        env_var="MODEL",
        type=str,
        help="The model to use for generating text. If not specified we will pick the first model from the service as returned by /v1/models",
    )
    parser.add_argument(
        "--tokenizer",
        env_var="TOKENIZER",
        type=str,
        help="Specify HF tokenizer to use for validating the output of the model. It's optional, we're going to rely on 'usage' or 'logprobs' field to get token count information",
    )
    parser.add_argument(
        "--chat",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use /v1/chat/completions API",
    )
    parser.add_argument(
        "--embeddings",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use /v1/embeddings API",
    )
    parser.add_argument(
        "-p",
        "--prompt-tokens",
        env_var="PROMPT_TOKENS",
        type=int,
        default=512,
        help="Length of the prompt in tokens. Default 512",
    )
    parser.add_argument(
        "--prompt-images-with-resolutions",
        type=parse_resolution,
        nargs="+",
        default=[],
        help="Images to add to the prompt for vision models, defined by their resolutions in format WIDTHxHEIGHT. "
        'For example, "--prompt-images-with-resolutions 3084x1080 1024x1024" will insert 2 images '
        "(3084 width x 1080 height and 1024 width x 1024 height) into the prompt. "
        "Images will be spaced out evenly across the prompt."
        "Only supported with --chat mode.",
    )
    parser.add_argument(
        "--prompt-images-positioning",
        type=str,
        choices=["space-evenly", "end"],
        default="space-evenly",
        help="How to position the images in the prompt. "
        "space-evenly: images are spaced out evenly across the prompt. E.g., 3 images in 'abcdefgh' is 'ab<image>cd<image>ef<image>gh'"
        "end: images are added to the end of the prompt. E.g., 3 images in 'abcdefgh' is 'abcdefgh<image><image><image>'"
        "Only relevant with --prompt-images-with-resolutions.",
    )
    parser.add_argument(
        "-o",
        "--max-tokens",
        env_var="MAX_TOKENS",
        type=int,
        default=64,
        help="Max number of tokens to generate. If --max-tokens-distribution is non-constant this is going to be the mean. Defaults to 64",
    )
    parser.add_argument(
        "--max-tokens-cap",
        env_var="MAX_TOKENS_CAP",
        type=int,
        help="If --max-tokens-distribution is non-constant, this truncates the distribition at the specified limit",
    )
    parser.add_argument(
        "--max-tokens-distribution",
        env_var="MAX_TOKENS_DISTRIBUTION",
        type=str,
        choices=["constant", "uniform", "exponential", "normal"],
        default="constant",
        help="How to sample `max-tokens` on each request",
    )
    parser.add_argument(
        "--max-tokens-range",
        env_var="MAX_TOKENS_RANGE",
        type=float,
        default=0.3,
        help="Specifies the width of the distribution. Specified value `alpha` is relative to `max-tokens`. For uniform distribution we'd sample from [max_tokens - max_tokens * alpha, max_tokens + max_tokens * alpha]. For normal distribution we'd sample from `N(max_tokens, max_tokens * alpha)`. Defaults to 0.3",
    )
    parser.add_argument(
        "--top-k",
        env_var="TOP_K",
        type=int,
        default=None,
        help="Specifies the top-k sampling parameter.",
    )
    parser.add_argument(
        "--stream",
        dest="stream",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the streaming API",
    )
    parser.add_argument(
        "-k",
        "--api-key",
        env_var="API_KEY",
        help="Auth for the API",
    )
    parser.add_argument(
        "--temperature",
        env_var="TEMPERATURE",
        type=float,
        default=1.0,
        help="Temperature parameter for the API",
    )
    parser.add_argument(
        "--logprobs",
        type=int,
        default=None,
        help="Whether to ask for logprobs, it makes things slower for some providers but is necessary for token count in streaming (unless it's Fireworks API that returns usage in streaming mode)",
    )
    parser.add_argument(
        "--summary-file",
        type=str,
        help="Append the line with the summary to the specified CSV file. Useful for generating a spreadsheet with perf sweep results. If the file doesn't exist, writes out the header first",
    )
    parser.add_argument(
        "--qps",
        type=float,
        default=None,
        help="Enabled 'fixed QPS' mode where requests are issues at the specified rate regardless of how long the processing takes. In this case --users and --spawn-rate need to be set to a sufficiently high value (e.g. 100)",
    )
    parser.add_argument(
        "--qps-distribution",
        type=str,
        choices=["constant", "uniform", "exponential"],
        default="constant",
        help="Must be used with --qps. Specifies how to space out requests: equally ('constant') or by sampling wait times from a distribution ('uniform' or 'exponential'). Expected QPS is going to match --qps",
    )
    parser.add_argument(
        "--burst",
        type=float,
        default=None,
        help="Makes requests to arrive in bursts every specified number of seconds. Note that burst duration has to be longer than maximum time of the response. Size of the burst is controlled by --users. The spawn rate -r is best set to a high value",
    )
    parser.add_argument(
        "--show-response",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print the result of each generation",
    )
    parser.add_argument(
        "-pcml",
        "--prompt-cache-max-len",
        env_var="PROMPT_CACHE_MAX_LEN",
        type=int,
        default=0,
        help="Maximum length of the prompt cache to use. Defaults to 0 (no caching).",
    )
    parser.add_argument(
        "--header",
        action="append",
        default=[],
        help="Arbitrary headers to add to the inference request. Can be used multiple times. For example, --header header1:value1 --header header2:value2",
    )
    parser.add_argument(
        "-n",
        "--n",
        default=1,
        type=int,
        help="How many sequences to generate (makes sense to use with non-zero temperature).",
    )


@events.quitting.add_listener
def _(environment, **kw):
    total_latency = environment.stats.entries[("total_latency", "METRIC")]
    if environment.stats.total.num_failures > 0 or total_latency.num_requests == 0:
        print("Test failed due to failed requests")
        environment.process_exit_code = 1
        return

    entries = copy.copy(InitTracker.logging_params)
    if environment.parsed_options.qps is not None:
        entries["concurrency"] = (
            f"QPS {environment.parsed_options.qps} {environment.parsed_options.qps_distribution}"
        )
    else:
        entries["concurrency"] = InitTracker.users
    for metric_name in [
        "time_to_first_token",
        "latency_per_token",
        "overall_latency_per_token",
        "num_tokens",
        "total_latency",
        "prompt_tokens",  # might overwrite the static value based on server side tokenization
    ]:
        entries[metric_name] = environment.stats.entries[
            (metric_name, "METRIC")
        ].avg_response_time
    if not environment.parsed_options.stream:
        # if there's no streaming these metrics are meaningless
        entries["time_to_first_token"] = ""
        entries["latency_per_token"] = ""
    entries["num_requests"] = total_latency.num_requests
    entries["qps"] = total_latency.total_rps
    percentile_to_report = [50, 90, 95, 99, 99.9]
    percentile_metrics = ["time_to_first_token", "total_latency"]
    for percentile_metric in percentile_metrics:
        metrics = environment.stats.entries[percentile_metric, "METRIC"]
        for percentile in percentile_to_report:
            name = f"P{percentile}_{percentile_metric}"
            entries[name] = metrics.get_response_time_percentile(percentile / 100)

    pretty_name = lambda s: " ".join([w.capitalize() for w in s.split("_")])
    entries = {pretty_name(k): v for k, v in entries.items()}

    # print in the final event handler to make sure our output is the last one
    @events.quit.add_listener
    def exit_printer(**kw):
        max_width = max(len(k) for k in entries.keys())
        print(" Summary ".center(80, "="))
        for k, v in entries.items():
            print(f"{k:<{max_width}}: {v}")
        print("=" * 80)

    if environment.parsed_options.summary_file:
        with open(environment.parsed_options.summary_file, "a") as f:
            writer = csv.DictWriter(f, fieldnames=entries.keys())
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(entries)
