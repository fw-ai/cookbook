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
        self.output_column = None

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
        self.output_column = "output"

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        n_rows = len(next(iter(batch.values())))
        rows = []
        for i in range(n_rows):
            row = {}
            for key, values in batch.items():
                row[key] = values[i]
            rows.append(
                self._config.prompt_template.format(**row) +
                self._tokenizer.eos_token)
        result = {self.output_column: rows}
        return result


class Lambda(BatchTransform):
    """
    Transform that applies a lambda to each row of input data.
    """
    _LAMBDA_CACHE = {}

    def __init__(self, config: DictConfig, tokenizer: AutoTokenizer) -> None:
        super().__init__(config, tokenizer)
        self.output_column = "output"

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        n_rows = len(next(iter(batch.values())))
        rows = []
        prompt_lambda = Lambda._LAMBDA_CACHE.get(self._config.prompt_lambda)
        if prompt_lambda is None:
            prompt_lambda = eval(self._config.prompt_lambda)
            Lambda._LAMBDA_CACHE[self._config.prompt_lambda] = prompt_lambda

        for i in range(n_rows):
            row = {}
            for key, values in batch.items():
                row[key] = values[i]
            try:
                rows.append(prompt_lambda(row) + self._tokenizer.eos_token)
            except KeyError:
                print(f"DEBUG: row {row}")
        result = {self.output_column: rows}
        return result
