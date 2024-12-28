# Copyright (c) Fireworks AI, Inc. and affiliates.
#
# All Rights Reserved.

import abc
from collections import defaultdict
from typing import Any, Dict, List, Optional

import hydra
from omegaconf import DictConfig
from recipes.eval.common.format import compile_jinja_template
from transformers import AutoTokenizer


class BatchTransform(abc.ABC):
    """
    Transform processes batches of data.
    """

    def __init__(self, config: DictConfig, tokenizer: Optional[AutoTokenizer]) -> None:
        """
        Args:
            config: dataset config,
            tokenizer: the tokenizer that will be processing the data.
        """
        self._config = config
        self._tokenizer = tokenizer
        self.template_columns = {}

    @abc.abstractmethod
    def _call(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transforms a batch of data.

        Args:
            batch: the input data to transform arranged as columns
                of lists of values.

        Returns:
            transformed batch.
        """

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transforms a batch of data. Wrapper around _call handling common parameters.

        Args:
            batch: the input data to transform arranged as columns
                of lists of values.

        Returns:
            transformed batch.
        """
        result = self._call(batch)

        # Handle pass-through columns
        passthrough_columns = self._config.get("passthrough_columns", [])
        for column in passthrough_columns:
            if column in batch:
                if column not in result:
                    result[column] = batch[column]
            else:
                raise ValueError(
                    f"Pass-through column '{column}' not found in the input batch."
                )

        return result


class Jinja(BatchTransform):
    """
    Transform that applies a function calling template to each row of the data.
    """

    # templates are not copyable so they cannot be compiled inside the constructor
    _TEMPLATE_CACHE = {}

    def __init__(self, config: DictConfig, tokenizer: Optional[AutoTokenizer]) -> None:
        """
        See parent.
        """
        super().__init__(config, tokenizer)
        self._output_image_column = "_images"
        self._input_image_column = config.get("image_column_name", None)
        for column in config.templates:
            self.template_columns[column] = f"_{column}"

    def _call(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        See parent.
        """
        if not batch:
            return {}

        # make sure that each column has the same number of rows
        n_rows = len(next(iter(batch.values())))
        for key in batch:
            cur_n_rows = len(batch[key])
            if cur_n_rows != n_rows:
                raise ValueError(
                    f"column '{next(iter(batch.keys()))}' has {n_rows} rows while column '{key}' has {cur_n_rows} rows"
                )

        rendered_templates = defaultdict(list)
        mode = self._config.get("mode", "train")
        for i in range(n_rows):
            row = {}
            for key, values in batch.items():
                row[key] = values[i]
            for column, template in self._config.templates.items():
                compiled_template = Jinja._TEMPLATE_CACHE.get(template)
                if compiled_template is None:
                    compiled_template = compile_jinja_template(template)
                    Jinja._TEMPLATE_CACHE[template] = compiled_template
                rendered_column = self.template_columns[column]
                rendered_templates[rendered_column].append(
                    compiled_template.render(
                        mode=mode, **row, **self._tokenizer.special_tokens_map
                    )
                )
            if self._input_image_column is not None:
                rendered_templates[self._output_image_column].append(
                    row[self._input_image_column]
                )
        return dict(rendered_templates)


class Sequential(BatchTransform):
    """
    Composite transform that executes sub-transforms sequentially.
    """

    def __init__(self, config: DictConfig, tokenizer: Optional[AutoTokenizer]) -> None:
        """
        See parent.
        """
        super().__init__(config, tokenizer)
        self._children = [
            hydra.utils.get_class(cfg.get("class"))(cfg, tokenizer)
            for cfg in config.children
        ]
        self.template_columns = self._config.get("template_columns", {})

    def _call(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        See parent.
        """
        for child in self._children:
            batch = child(batch)
        return batch
