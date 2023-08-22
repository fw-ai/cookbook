# Copyright (c) Fireworks AI, Inc. and affiliates.
#
# All Rights Reserved.

import random
import re
from typing import Any, Dict

from recipes.common.batch_transform import BatchTransform


class NaturalQuestions(BatchTransform):
    """
    Transform that constructs the data by filling a string template
    with parameters from input data rows.
    """

    @staticmethod
    def _remove_document_prefix(document: str) -> str:
        separator = "] "
        return document[document.find(separator) + len(separator) :]

    @staticmethod
    def _remove_answer_prefix(answer: str) -> str:
        prefix = "Answer: "
        if not answer.startswith(prefix):
            raise ValueError(f"answer should start with '{prefix}', found {answer}")
        return answer[len(prefix) :]

    def __call__(self, row: Dict[str, Any]) -> Dict[str, Any]:
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
            "answers": answers,
            "document": documents,
            "score": scores,
        }


class QueryDocument(BatchTransform):
    """
    Transform that constructs the data by filling a string template
    with parameters from input data rows.
    """

    def __call__(self, row: Dict[str, Any]) -> Dict[str, Any]:
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
            "answers": answers,
            "document": documents,
            "score": scores,
        }
