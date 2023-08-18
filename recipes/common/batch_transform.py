# Copyright (c) Fireworks AI, Inc. and affiliates.
#
# All Rights Reserved.

import abc
from typing import Any, Dict

from omegaconf import DictConfig
from transformers import AutoTokenizer


class BatchTransform(abc.ABC):
    """
    Transform processes batches of data.
    """

    def __init__(self, config: DictConfig, tokenizer: AutoTokenizer) -> None:
        """
        Args:
            config: dataset config,
            tokenizer: the tokenizer to use for things like EOS token.
        """
        self._config = config
        self._tokenizer = tokenizer
        self.prompt_column = None
        self.completion_column = None

    @abc.abstractmethod
    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transforms a batch of data.

        Args:
            batch: the input data to transform arranged as columns
                of lists of values.

        Returns:
            transformed batch.
        """
        raise NotImplementedError()


class StringTemplate(BatchTransform):
    """
    Transform that constructs the data by filling a string template
    with parameters from input data rows.
    """

    def __init__(self, config: DictConfig, tokenizer: AutoTokenizer) -> None:
        super().__init__(config, tokenizer)
        self.prompt_column = "_prompt"
        self.completion_column = "_completion"

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        n_rows = len(next(iter(batch.values())))
        prompts = []
        completions = []
        for i in range(n_rows):
            row = {}
            for key, values in batch.items():
                row[key] = values[i]
            prompts.append(self._config.prompt_template.format(**row))
            completions.append(
                self._config.completion_template.format(**row)
                + self._tokenizer.eos_token
            )
        result = {self.prompt_column: prompts, self.completion_column: completions}
        return result


class Lambda(BatchTransform):
    """
    Transform that applies a lambda to each row of input data.
    """

    _LAMBDA_CACHE = {}

    def __init__(self, config: DictConfig, tokenizer: AutoTokenizer) -> None:
        super().__init__(config, tokenizer)
        self.prompt_column = "_prompt"
        self.completion_column = "_completion"

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        n_rows = len(next(iter(batch.values())))
        prompts = []
        completions = []
        prompt_lambda = Lambda._LAMBDA_CACHE.get(self._config.prompt_lambda)
        completion_lambda = Lambda._LAMBDA_CACHE.get(self._config.completion_lambda)
        if prompt_lambda is None:
            prompt_lambda = eval(self._config.prompt_lambda)
            Lambda._LAMBDA_CACHE[self._config.prompt_lambda] = prompt_lambda
            completion_lambda = eval(self._config.completion_lambda)
            Lambda._LAMBDA_CACHE[self._config.completion_lambda] = completion_lambda

        for i in range(n_rows):
            row = {}
            for key, values in batch.items():
                row[key] = values[i]
            prompts.append(prompt_lambda(row))
            completions.append(completion_lambda(row) + self._tokenizer.eos_token)
        result = {self.prompt_column: prompts, self.completion_column: completions}
        return result
