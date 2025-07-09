import random
import re
import json
import os
from reward_kit import reward_function
from reward_kit.models import EvaluateResult, MetricResult  # See documentation for core data types
from fireworks import LLM



user_template = """
    You are a helpful assistant. Your task is to judge which AI-generated completion best fulfills the user's intent.

    Instructions:
    1. Carefully read the conversation below between the user and the assistant.
    2. Examine each proposed completion. Each completion should appropriately address the user's question in one or more of these domains:
      - Software Engineering: correctness, clarity, and practicality of code or technical explanation.
      - Mathematics: accuracy of problem solving, logical reasoning, and correctness of final answers.
      - Creative Writing: originality, coherence, style, and how well it matches the user's creative prompt.
    3. Assess each completion on how well it satisfies the user's request in the relevant domain(s).
    4. Consider the instructions provided in the system message, and whether the completion aligns with them.
    5. The solution should be correct, clear, well-structured, and concise if possible, and does **NOT** contain any text unrelated to the problem-solving.
    6. Choose the completion id that most accurately and completely satisfies the user's intent.
    7. Output your reasoning first, enclosed in `<reasoning>` tags, clearly explaining why the chosen completion is best or why none are suitable.
    8. Ensure the content of `<top_completion>` is fully supported by your reasoning.

    Examples:
    * If completion X seems the most plausible and relevant: `<reasoning>completion_X is the most accurate choice because it follows system prompt in the following way ...</reasoning>\n<top_completion>[X]</top_completion>`
    * If *no* completion is suitable: `<reasoning>No completion satisfies the instructions in the system prompt because ...</reasoning>\n<top_completion>[]</top_completion>`
    
    <conversation>
    {conversation}
    </conversation>

    <completion_1>
    {completion_1}
    </completion_1>

    <completion_2>
    {completion_2}
    </completion_2>
"""

llm = LLM(
    model="qwen3-235b-a22b",
    deployment_type="serverless",
    api_key=os.getenv("<YOUR_API_KEY>"),
    accelerator_type="NVIDIA_H200_141GB",
    draft_model="accounts/fireworks/models/qwen3-1p7b",
    draft_token_count=2,
    accelerator_count=8,
    min_replica_count=1,
) 
llm.apply()


@reward_function(mode="batch")
def evaluate(rollouts_messages=[], ground_truth=None, **kwargs) -> list[EvaluateResult]:
    conversation = rollouts_messages[0][:-1]
    group = [messages[-1]["content"] for messages in rollouts_messages]
    group_size = len(group)

    shuffled_group, orig_to_new = group_shuffle(group)

    user_message = user_template.format(
        conversation=conversation, completion_1=shuffled_group[0], completion_2=shuffled_group[1]
    )

    messages = [{"role": "user", "content": user_message}]


    response = llm.chat.completions.create(messages=messages)
    completion = response.choices[0].message.content

    top = _extract_top(completion)
    scores = []

    for i in range(group_size):
        if i % group_size + 1 in top:
            scores.append(1)
        else:
            scores.append(0)

    reshuffled_scores = group_unshuffle(scores, orig_to_new)

    output = []

    for score in reshuffled_scores:
        output.append(
            EvaluateResult(
                score=score,  # Required: The final evaluation score
                is_score_valid=True,  # Optional: Whether the score is valid, true by default
                reason="model judge reward",  # Optional: The reason for the score
                metrics={  # Optional: A dict of sub metrics for debugging purposes
                    "model_judge_reward": MetricResult(is_score_valid=True, score=score, reason="model judge reward"),
                },
            )
        )
    print(len(output), len(rollouts_messages))
    return output


def group_shuffle(group: list) -> tuple[list, list[tuple[int, int]]]:
    """
    Shuffles data within  group of consecutive group_size elements.
    Returns the shuffled data and shuffle information for reshuffling.
    """

    result = group.copy()
    orig_to_new = []
    group_size = len(group)

    # Get indices for current group
    group_indices = list(range(group_size))

    # Store original indices
    original_indices = group_indices.copy()

    # Shuffle indices
    random.shuffle(group_indices)

    # Apply shuffle to result
    for orig, new in zip(original_indices, group_indices):
        result[new] = group[orig]

    # Store mapping of original to new indices
    orig_to_new = [(orig, new) for orig, new in zip(original_indices, group_indices)]

    return result, orig_to_new


def group_unshuffle(group: list, orig_to_new: list[tuple[int, int]]) -> list:
    """
    Unshuffles group using the shuffle information.
    """

    result = group.copy()

    for orig_idx, new_idx in orig_to_new:
        result[orig_idx] = group[new_idx]

    return result


def _extract_top(completion: str) -> list[int]:
    completion = completion.strip()
    match = re.search(r"<top_completion>(.*?)</top_completion>", completion, re.DOTALL)
    if not match:
        return []
    top_str = match.group(1).strip()
    if top_str == "":
        return []
    try:
        top = json.loads(top_str)
    except json.JSONDecodeError:
        return []
    if isinstance(top, int):
        top_ids = [top]
    elif isinstance(top, list):
        try:
            top_ids = set(int(x) for x in top)
        except (ValueError, TypeError):
            return []
    else:
        return []
    for id in top_ids:
        if id < 1 or id > 2:
            return []
        return list(top_ids)

    return []
