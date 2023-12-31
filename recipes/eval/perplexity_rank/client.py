# Copyright (c) Fireworks AI, Inc. and affiliates.
#
# All Rights Reserved.

import abc
from importlib import import_module
import math

import fireworks.client
import openai
import torch
from omegaconf import DictConfig
from recipes.common.env import env
from recipes.common.peft import load_inference_model
from recipes.common.tokenizer import load_tokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class Client(abc.ABC):
    """
    Client that calls the model.
    """

    def __init__(self, config: DictConfig) -> None:
        """
        Args:
            config: client config.
        """
        self._config = config

    @staticmethod
    def create(config: DictConfig) -> "Client":
        """
        Creates a transform object from the provided config.

        Args:
            config: the config with transform parameters.

        Returns:
            transform object.
        """
        clazz = config.model.client_class
        module_name, class_name = clazz.rsplit(".", 1)
        module = import_module(module_name)
        cls = getattr(module, class_name)
        return cls(config)

    @abc.abstractmethod
    def perplexity(self, prompt: str, completion: str) -> float:
        """
        Computes perplexity on a completion given the prompt.

        Args:
            prompt: the prompt to use as a condition,
            completion: the continuation to compute the perplexity on.

        Returns:
            perplexity value.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def completion(self, prompt: str) -> str:
        """
        Generates a completion for a given prompt.

        Args:
            prompt: the prompt to pass on to the model.

        Returns:
            completion of the prompt.
        """
        raise NotImplementedError


class LocalClient(Client):
    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)
        self._tokenizer = load_tokenizer(config.model, add_eos_token=False)
        self._model = load_inference_model(config)
        self._device = env().device

    def perplexity(self, prompt: str, completion: str) -> float:
        num_document_tokens = len(
            self._tokenizer(prompt, return_tensors="pt")["input_ids"][0]
        )
        text = prompt + completion
        input_ids = self._tokenizer(text, return_tensors="pt")["input_ids"].to(
            self._device
        )
        num_target_tokens = len(input_ids[0]) - num_document_tokens
        target_ids = input_ids.clone()
        target_ids[:, :-num_target_tokens] = -100
        with torch.no_grad():
            outputs = self._model(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss

        return math.exp(neg_log_likelihood)

    def completion(self, prompt: str) -> str:
        inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        outputs = self._model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=512,
        )
        outputs = outputs.squeeze(0)
        try:
            index = outputs.tolist().index(self._tokenizer.eos_token_id)
            outputs = outputs[:index]
        except ValueError:
            ...
        outputs = outputs.unsqueeze(0)
        decoded = self._tokenizer.batch_decode(outputs, skip_special_tokens=False)[0]
        start = decoded.find(prompt) + len(prompt)
        return decoded[start:]


class FireworksClient(Client):
    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)
        self._model = config.model.name
        api_key = config.model.get("api_key")
        if api_key:
            fireworks.client.api_key = api_key

    def perplexity(self, prompt: str, completion: str) -> float:
        text = prompt + completion
        # TODO(pawel): replace with fireworks.client.Completion.create
        # after the client fix lands.
        response = openai.Completion.create(
            model=self._model,
            prompt=text,
            max_tokens=0,
            logprobs=1,
            echo=True,
        )
        logprobs = response.choices[0].logprobs
        completion_start = logprobs.text_offset.index(len(prompt))
        completion_logprobs = logprobs.token_logprobs[completion_start:]
        return math.exp(-sum(completion_logprobs) / len(completion_logprobs))

    def completion(self, prompt: str) -> str:
        completion = fireworks.client.Completion.create(
            model=self._model,
            prompt=prompt,
        )
        return completion.choices[0].text


class SentenceTransformerClient(Client):
    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)
        device = env().device
        self._model = SentenceTransformer(config.model.name, device=device)

    def perplexity(self, prompt: str, completion: str) -> float:
        prompt_embedding = self._model.encode(prompt, show_progress_bar=False)
        completion_embedding = self._model.encode(completion, show_progress_bar=False)
        similarity = cosine_similarity([prompt_embedding], [completion_embedding])[0][0]
        similarity = min(similarity, 1.0)
        return 1.0 - similarity

    def completion(self, prompt: str) -> str:
        raise NotImplementedError("completion is not supported by embedding models")
