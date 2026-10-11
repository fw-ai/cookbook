"""Multi-hop QA search environment for IGPO training.

Provides a self-contained search environment backed by a paragraph pool
(e.g. from HotpotQA).  The model interacts via two tools:

- ``search(query)``: TF-IDF retrieval over the paragraph pool.
- ``submit_answer(answer)``: Terminates the episode and returns F1 reward.

No external APIs are needed — retrieval runs locally over the paragraphs
bundled with each question.
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

TOOL_NAME_SEARCH = "search"
TOOL_NAME_SUBMIT = "submit_answer"

MULTIHOP_QA_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": TOOL_NAME_SEARCH,
            "description": (
                "Search for information about a topic. Returns the most "
                "relevant paragraphs from the knowledge base."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query.",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": TOOL_NAME_SUBMIT,
            "description": (
                "Submit your final answer to the question. Use this once you "
                "are confident in your answer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "Your final answer.",
                    },
                },
                "required": ["answer"],
                "additionalProperties": False,
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Text normalisation / F1 helpers (SQuAD-style)
# ---------------------------------------------------------------------------

def _normalize_answer(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())


def _token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = _normalize_answer(prediction).split()
    gt_tokens = _normalize_answer(ground_truth).split()
    if not gt_tokens:
        return 1.0 if not pred_tokens else 0.0
    if not pred_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# TF-IDF retrieval (stdlib only, no sklearn)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def _build_tfidf_index(
    paragraphs: Sequence[Dict[str, Any]],
) -> Tuple[List[str], List[List[str]], List[Dict[str, float]]]:
    """Build a simple TF-IDF index over paragraphs.

    Returns (titles, doc_tokens_list, tfidf_vectors).
    """
    titles: List[str] = []
    doc_tokens_list: List[List[str]] = []

    for para in paragraphs:
        title = para.get("title", "")
        sentences = para.get("sentences", [])
        text = title + " " + " ".join(sentences)
        tokens = _tokenize(text)
        titles.append(title)
        doc_tokens_list.append(tokens)

    N = len(doc_tokens_list)
    df: Dict[str, int] = {}
    for tokens in doc_tokens_list:
        for term in set(tokens):
            df[term] = df.get(term, 0) + 1

    tfidf_vectors: List[Dict[str, float]] = []
    for tokens in doc_tokens_list:
        tf: Dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        vec: Dict[str, float] = {}
        for t, count in tf.items():
            idf = math.log((N + 1) / (df.get(t, 0) + 1)) + 1
            vec[t] = count * idf
        tfidf_vectors.append(vec)

    return titles, doc_tokens_list, tfidf_vectors


def _tfidf_search(
    query: str,
    titles: List[str],
    paragraphs: Sequence[Dict[str, Any]],
    tfidf_vectors: List[Dict[str, float]],
    top_k: int = 2,
) -> List[Dict[str, Any]]:
    query_tokens = _tokenize(query)
    scores: List[Tuple[float, int]] = []
    for idx, vec in enumerate(tfidf_vectors):
        score = sum(vec.get(t, 0.0) for t in query_tokens)
        scores.append((score, idx))
    scores.sort(key=lambda x: -x[0])

    results: List[Dict[str, Any]] = []
    for score, idx in scores[:top_k]:
        if score <= 0:
            break
        para = paragraphs[idx]
        results.append({
            "title": titles[idx],
            "content": " ".join(para.get("sentences", [])),
            "relevance_score": round(score, 3),
        })
    return results


# ---------------------------------------------------------------------------
# Step result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SearchQAStepResult:
    observation: str
    reward: float
    terminated: bool
    truncated: bool
    tool_name: str
    step_index: int

    def as_tool_result(self) -> Dict[str, Any]:
        return {
            "observation": self.observation,
            "reward": self.reward,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "tool_name": self.tool_name,
            "step_index": self.step_index,
        }


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class SearchQAEnv:
    """Multi-hop QA environment with search and answer submission tools."""

    def __init__(
        self,
        paragraphs: Sequence[Dict[str, Any]],
        ground_truth: str,
        max_steps: int = 8,
        search_top_k: int = 2,
    ):
        self._paragraphs = list(paragraphs)
        self._ground_truth = ground_truth
        self._max_steps = max_steps
        self._search_top_k = search_top_k

        self._titles, self._doc_tokens, self._tfidf_vecs = _build_tfidf_index(
            self._paragraphs
        )

        self._step_count = 0
        self._terminated = False
        self._truncated = False
        self._submitted_answer: str | None = None

    def reset(self) -> Dict[str, Any]:
        self._step_count = 0
        self._terminated = False
        self._truncated = False
        self._submitted_answer = None
        return {"observation": "Ready. Use the search tool to find information, then submit your answer."}

    def step(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        tool_name = str(tool_name).strip().lower()

        if self._terminated or self._truncated:
            result = SearchQAStepResult(
                observation="Episode already ended.",
                reward=0.0,
                terminated=self._terminated,
                truncated=self._truncated,
                tool_name=tool_name,
                step_index=self._step_count,
            )
            return result.as_tool_result()

        self._step_count += 1

        if tool_name == TOOL_NAME_SEARCH:
            return self._handle_search(arguments)
        elif tool_name == TOOL_NAME_SUBMIT:
            return self._handle_submit(arguments)
        else:
            self._terminated = True
            result = SearchQAStepResult(
                observation=f"Unknown tool '{tool_name}'. Expected 'search' or 'submit_answer'.",
                reward=0.0,
                terminated=True,
                truncated=False,
                tool_name=tool_name,
                step_index=self._step_count,
            )
            return result.as_tool_result()

    def _handle_search(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        query = str(arguments.get("query", ""))
        results = _tfidf_search(
            query, self._titles, self._paragraphs,
            self._tfidf_vecs, top_k=self._search_top_k,
        )

        if results:
            parts = []
            for r in results:
                parts.append(f"**{r['title']}**: {r['content']}")
            observation = "\n\n".join(parts)
        else:
            observation = "No relevant results found. Try a different query."

        truncated = self._step_count >= self._max_steps and not self._terminated
        if truncated:
            self._truncated = True
            observation += "\n\n[Max steps reached. Submit your answer now.]"

        result = SearchQAStepResult(
            observation=observation,
            reward=0.0,
            terminated=False,
            truncated=truncated,
            tool_name=TOOL_NAME_SEARCH,
            step_index=self._step_count,
        )
        return result.as_tool_result()

    def _handle_submit(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        answer = str(arguments.get("answer", ""))
        self._submitted_answer = answer
        self._terminated = True

        f1 = _token_f1(answer, self._ground_truth)

        observation = f"Answer submitted: {answer!r}. F1 score: {f1:.3f}"
        result = SearchQAStepResult(
            observation=observation,
            reward=f1,
            terminated=True,
            truncated=False,
            tool_name=TOOL_NAME_SUBMIT,
            step_index=self._step_count,
        )
        return result.as_tool_result()


def build_search_qa_env(
    context: Dict[str, Any],
    ground_truth: str,
    max_steps: int = 8,
    search_top_k: int = 2,
) -> SearchQAEnv:
    """Create a SearchQAEnv from a HotpotQA-style context dict."""
    titles = context.get("titles", [])
    paragraphs_raw = context.get("paragraphs", [])

    paragraphs: List[Dict[str, Any]] = []
    for i, sentences in enumerate(paragraphs_raw):
        title = titles[i] if i < len(titles) else f"Document {i}"
        paragraphs.append({"title": title, "sentences": sentences})

    return SearchQAEnv(
        paragraphs=paragraphs,
        ground_truth=ground_truth,
        max_steps=max_steps,
        search_top_k=search_top_k,
    )
