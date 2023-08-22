# Copyright (c) Fireworks AI, Inc. and affiliates.
#
# All Rights Reserved.

import abc
from importlib import import_module
import random
import re
import sys
from typing import Any, Dict, Union
import warnings

from omegaconf import DictConfig
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict


class DatasetTransform(abc.ABC):
    """
    Transform processes a dataset.
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
    def __call__(
        self, dataset: Union[Dataset, DatasetDict]
    ) -> Union[Dataset, DatasetDict]:
        """
        Transforms a dataset.

        Args:
            dataset: the input dataset to transform.

        Returns:
            transformed data.
        """
        raise NotImplementedError()

    @staticmethod
    def create(config: DictConfig, tokenizer: AutoTokenizer) -> "DatasetTransform":
        """
        Creates a transform object from the provided config.

        Args:
            config: the config with transform parameters,
            tokenizer: the tokenizer that can be used to customize the
                transform.

        Returns:
            transform object.
        """
        clazz = config.get("class")
        module_name, class_name = clazz.rsplit(".", 1)
        module = import_module(module_name)
        cls = getattr(module, class_name)
        return cls(config, tokenizer)


class NaturalQuestions(DatasetTransform):
    """
    Transform for a natural questions style data.
    The input data is formatted as:

    <sys_prompt>\n\nDocument [1] <doc_1>\n ... Document [n] <doc_n>\n\n
    Question: <question>\nAnswer: <answer>\nLong Answer: <long_answer>\n
    Gold Document ID: <doc_id>
    """

    @staticmethod
    def _remove_document_prefix(document: str) -> str:
        """
        Strips out the document prefix.

        Args:
            document: the document content.

        Returns:
            document content without prefix.
        """
        separator = "] "
        return document[document.find(separator) + len(separator) :]

    def _expand_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parses data components out of a text string.

        Args:
            row: the row of data.

        Returns:
            extracted relevant data components.
        """
        queries = []
        answers = []
        documents = []
        scores = []
        pattern = r"Gold Document ID: (\d+)"
        text = row["text"]

        _instruction, documents_str, questions_str = text.split("\n\n")
        candidates = documents_str.split("\n")
        query, answer, _long_answer, ideal = questions_str.split("\n")
        answer = self._remove_answer_prefix(answer)

        # query
        prefix = "Question: "
        if not query.startswith(prefix):
            raise ValueError(f"Expected question to start with {prefix}, found {query}")
        query = query[len(prefix) :]

        # ideal
        match = re.fullmatch(pattern, ideal)
        if not match:
            raise ValueError(
                f"Ideal should match the following pattern: {pattern}, found {ideal}"
            )
        id = int(match.group(1))
        ideal_candidate = candidates[id - 1]  # document ids start from 1
        if not ideal_candidate.startswith(f"Document [{id}]"):
            prefix = ideal_candidate[: ideal_candidate.find("]") + 1]
            raise ValueError(
                f"Mismatch between expected document id {id} and document prefix {prefix}"
            )
        documents.append(self._remove_document_prefix(ideal_candidate))
        queries.append(query)
        answers.append(answer)
        scores.append(9)

        # negatives
        num_candidates = len(candidates)
        num_negatives = min(self._config.negatives, num_candidates)
        negative_indices = random.sample(range(num_candidates), num_negatives)
        documents.extend(
            [self._remove_document_prefix(candidates[i]) for i in negative_indices]
        )
        queries.extend([query] * num_negatives)
        answers.extend([answer] * num_negatives)
        scores.extend([1] * num_negatives)

        if [len(queries), len(answers), len(documents)] != [len(scores)] * 3:
            raise ValueError(
                f"queries ({len(queries)}) answers ({len(answers)}), documents ({len(documents)}), "
                f"and scores ({len(scores)}) should all have the same length"
            )

        return {
            "query": queries,
            "answer": answers,
            "document": documents,
            "score": scores,
        }

    @staticmethod
    def _flatten_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Flattens nested lists.

        Args:
            batch: data with possibly nested lists as values.

        Returns:
            data with flattened values.
        """
        result = {}
        for key in batch:
            result[key] = [item for sublist in batch[key] for item in sublist]
        return result

    @staticmethod
    def _remove_answer_prefix(answer: str) -> str:
        """
        Strips answer prefix from a string.

        Args:
            answer: the string to process.

        Returns:
            input string without the answer prefix.
        """
        prefix = "Answer: "
        if not answer.startswith(prefix):
            raise ValueError(f"answer should start with '{prefix}', found {answer}")
        return answer[len(prefix) :]

    def __call__(
        self, dataset: Union[Dataset, DatasetDict]
    ) -> Union[Dataset, DatasetDict]:
        """
        See the parent class.
        """
        columns_to_remove = dataset.column_names
        dataset = dataset.map(
            self._expand_row, remove_columns=columns_to_remove, num_proc=16
        )
        dataset = dataset.map(self._flatten_batch, batched=True)

        max_samples = self._config.get("max_samples")
        if max_samples is not None and max_samples > 0:
            max_samples = min(len(dataset), max_samples)
            dataset = dataset.select(range(max_samples))
            print(f"truncated dataset to size {len(dataset)}")

        return dataset


class QueryDocument(DatasetTransform):
    """
    Transform processing data in query-document format.

    The input data stores one query-document pair per row.
    """

    def __call__(
        self, dataset: Union[Dataset, DatasetDict]
    ) -> Union[Dataset, DatasetDict]:
        dataset = dataset.sort(self._config.query_column)
        queries = []
        documents = []
        scores = []
        docs_per_query = self._config.get("docs_per_query")
        i = 0
        num_queries = 0
        skipped_rows = 0
        skipped_queries = 0
        while i < len(dataset):
            query = dataset[i][self._config.query_column]

            # process rows for a single query
            score_ids = []
            while i < len(dataset) and dataset[i][self._config.query_column] == query:
                score_ids.append((int(dataset[i][self._config.score_column]), i))
                i += 1
            sorted_score_ids = sorted(score_ids, reverse=True)
            if docs_per_query is not None:
                if len(sorted_score_ids) < docs_per_query:
                    skipped_rows += len(sorted_score_ids)
                    skipped_queries += 1
                    continue
                sorted_score_ids = sorted_score_ids[:docs_per_query]
            num_queries += 1
            for score, id in sorted_score_ids:
                queries.append(dataset[id][self._config.query_column])
                documents.append(dataset[id][self._config.document_column])
                scores.append(score)
        if skipped_rows > 0:
            warnings.warn(
                f"{skipped_rows} / {len(dataset)} rows and {skipped_queries} / {skipped_queries + num_queries} "
                f"queries were excluded due to the query size filter of {docs_per_query}"
            )
        print(f"extracted {num_queries} queries")
        return Dataset.from_dict(
            {
                "query": queries,
                "answer": [""] * len(queries),
                "document": documents,
                "score": scores,
            }
        )


class MsMarcoRank(DatasetTransform):
    """
    Transform processing data in MsMarco format.

    Each input data row is a <query, [positives], [negatives]> tuple.
    """

    def __call__(
        self, dataset: Union[Dataset, DatasetDict]
    ) -> Union[Dataset, DatasetDict]:
        queries = []
        documents = []
        scores = []
        max_samples = self._config.get("max_samples", sys.maxsize)
        if self._config.get("max_samples") is not None:
            print(f"limiting to max samples {max_samples}")
        docs_per_query = self._config.get("docs_per_query")
        if docs_per_query is not None:
            print(f"limiting to max docs per query {docs_per_query}")
        num_queries = 0
        skipped_queries = 0
        for row in dataset:
            query, positive, negative = row["query"], row["positive"], row["negative"]
            if (
                docs_per_query is not None
                and len(positive) + len(negative) < docs_per_query
            ):
                skipped_queries += 1
                continue
            limit = docs_per_query if docs_per_query else len(positive) + len(negative)
            for doc in positive[:limit]:
                queries.append(query)
                documents.append(doc)
                scores.append(9)
            for doc in negative[: limit - len(positive)]:
                queries.append(query)
                documents.append(doc)
                scores.append(1)
            num_queries += 1
            if num_queries == max_samples:
                break

        if skipped_queries > 0:
            warnings.warn(
                f"{skipped_queries} / {skipped_queries + num_queries} queries "
                f"were excluded due to the query size filter of {docs_per_query}"
            )
        print(f"extracted {len(queries)} queries")
        return Dataset.from_dict(
            {
                "query": queries,
                "answer": [""] * len(queries),
                "document": documents,
                "score": scores,
            }
        )
