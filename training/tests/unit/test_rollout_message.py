import pytest

from training.utils.rl.rollout import (
    MessageTrajectoryAssembler,
    MessageValidationError,
    TITOTokenizer,
)


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, tools=None, **kwargs):
        rendered = "".join(f"<{msg['role']}>{msg.get('content') or ''}</{msg['role']}>\n" for msg in messages)
        if add_generation_prompt:
            rendered += "<assistant>"
        if tokenize:
            return self.encode(rendered, add_special_tokens=False)
        return rendered

    def encode(self, text, add_special_tokens=False):
        return [ord(ch) for ch in text]


def _add_turn(assembler, messages, index):
    prompt = assembler.prepare_next_input(messages)
    assistant = {"role": "assistant", "content": f"a{index}"}
    assembler.add_assistant_response(
        request_messages=messages,
        assistant_message=assistant,
        prompt_token_ids=prompt,
        completion_token_ids=[1000 + index],
        completion_logprobs=[-0.1],
    )
    checkpoint_messages = [*messages, assistant]
    return checkpoint_messages, [
        *checkpoint_messages,
        {"role": "tool", "content": f"t{index}", "tool_call_id": f"call_{index}"},
    ]


def test_message_in_preserves_generated_assistant_tokens_across_tool_turn():
    assembler = MessageTrajectoryAssembler(TITOTokenizer(FakeTokenizer()))
    first_messages = [{"role": "user", "content": "hi"}]

    first_prompt = assembler.prepare_next_input(first_messages)
    generated_tokens = [9001, 9002]
    assembler.add_assistant_response(
        request_messages=first_messages,
        assistant_message={"role": "assistant", "content": "retokenized text would differ"},
        prompt_token_ids=first_prompt,
        completion_token_ids=generated_tokens,
        completion_logprobs=[-0.1, -0.2],
    )

    second_messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "retokenized text would differ"},
        {"role": "tool", "content": "42", "tool_call_id": "call_1"},
    ]
    second_prompt = assembler.prepare_next_input(second_messages)

    assert second_prompt[: len(first_prompt) + len(generated_tokens)] == first_prompt + generated_tokens


def test_message_in_user_and_system_appends_are_allowed_by_default():
    assembler = MessageTrajectoryAssembler(TITOTokenizer(FakeTokenizer()))
    first_messages = [{"role": "user", "content": "hi"}]
    first_prompt = assembler.prepare_next_input(first_messages)
    assembler.add_assistant_response(
        request_messages=first_messages,
        assistant_message={"role": "assistant", "content": "hello"},
        prompt_token_ids=first_prompt,
        completion_token_ids=[101],
        completion_logprobs=[-0.1],
    )

    next_prompt = assembler.prepare_next_input(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "system", "content": "retry carefully"},
            {"role": "user", "content": "again"},
        ]
    )

    assert next_prompt[: len(first_prompt) + 1] == first_prompt + [101]


def test_message_in_rejects_modified_prior_message():
    assembler = MessageTrajectoryAssembler(TITOTokenizer(FakeTokenizer()))
    first_messages = [{"role": "user", "content": "hi"}]
    first_prompt = assembler.prepare_next_input(first_messages)
    assembler.add_assistant_response(
        request_messages=first_messages,
        assistant_message={"role": "assistant", "content": "hello"},
        prompt_token_ids=first_prompt,
        completion_token_ids=[101],
        completion_logprobs=[-0.1],
    )

    with pytest.raises(MessageValidationError):
        assembler.prepare_next_input([{"role": "user", "content": "changed"}])


def test_message_in_one_step_rollback_to_assistant_checkpoint():
    assembler = MessageTrajectoryAssembler(TITOTokenizer(FakeTokenizer()))

    m1 = [{"role": "user", "content": "hi"}]
    p1 = assembler.prepare_next_input(m1)
    assembler.add_assistant_response(
        request_messages=m1,
        assistant_message={"role": "assistant", "content": "call tool"},
        prompt_token_ids=p1,
        completion_token_ids=[101],
        completion_logprobs=[-0.1],
    )
    m2 = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "call tool"},
        {"role": "tool", "content": "first", "tool_call_id": "call_1"},
    ]
    p2 = assembler.prepare_next_input(m2)
    assembler.add_assistant_response(
        request_messages=m2,
        assistant_message={"role": "assistant", "content": "final"},
        prompt_token_ids=p2,
        completion_token_ids=[202],
        completion_logprobs=[-0.2],
    )

    retry_prompt = assembler.prepare_next_input(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "call tool"},
            {"role": "tool", "content": "retry", "tool_call_id": "call_1"},
        ]
    )

    assert assembler.token_ids == p1 + [101]
    assert retry_prompt[: len(p1) + 1] == p1 + [101]


def test_message_in_multi_step_rollback_raises():
    assembler = MessageTrajectoryAssembler(TITOTokenizer(FakeTokenizer()), max_assistant_rollback_steps=1)

    m1 = [{"role": "user", "content": "hi"}]
    p1 = assembler.prepare_next_input(m1)
    assembler.add_assistant_response(
        request_messages=m1,
        assistant_message={"role": "assistant", "content": "a1"},
        prompt_token_ids=p1,
        completion_token_ids=[101],
        completion_logprobs=[-0.1],
    )
    m2 = m1 + [{"role": "assistant", "content": "a1"}, {"role": "tool", "content": "t1"}]
    p2 = assembler.prepare_next_input(m2)
    assembler.add_assistant_response(
        request_messages=m2,
        assistant_message={"role": "assistant", "content": "a2"},
        prompt_token_ids=p2,
        completion_token_ids=[202],
        completion_logprobs=[-0.2],
    )
    m3 = m2 + [{"role": "assistant", "content": "a2"}, {"role": "tool", "content": "t2"}]
    p3 = assembler.prepare_next_input(m3)
    assembler.add_assistant_response(
        request_messages=m3,
        assistant_message={"role": "assistant", "content": "a3"},
        prompt_token_ids=p3,
        completion_token_ids=[303],
        completion_logprobs=[-0.3],
    )

    with pytest.raises(MessageValidationError, match="exceeds"):
        assembler.prepare_next_input(m1 + [{"role": "assistant", "content": "a1"}, {"role": "tool", "content": "retry"}])


def test_message_in_retains_only_reachable_checkpoints():
    assembler = MessageTrajectoryAssembler(
        TITOTokenizer(FakeTokenizer()), max_assistant_rollback_steps=2
    )
    messages = [{"role": "user", "content": "hi"}]
    checkpoint_messages = []

    for index in range(10):
        checkpoint, messages = _add_turn(assembler, messages, index)
        checkpoint_messages.append(checkpoint)

    assert len(assembler.checkpoints) == 3
    assert [checkpoint.messages[-1]["content"] for checkpoint in assembler.checkpoints] == [
        "a7",
        "a8",
        "a9",
    ]

    retry_prompt = assembler.prepare_next_input(
        [
            *checkpoint_messages[7],
            {"role": "tool", "content": "retry", "tool_call_id": "call_7"},
        ]
    )

    assert len(assembler.checkpoints) == 1
    assert assembler.messages == checkpoint_messages[7]
    assert retry_prompt[: len(assembler.token_ids)] == assembler.token_ids


def test_message_in_rejects_rollback_older_than_retained_window():
    assembler = MessageTrajectoryAssembler(
        TITOTokenizer(FakeTokenizer()), max_assistant_rollback_steps=2
    )
    messages = [{"role": "user", "content": "hi"}]
    checkpoint_messages = []

    for index in range(5):
        checkpoint, messages = _add_turn(assembler, messages, index)
        checkpoint_messages.append(checkpoint)

    with pytest.raises(MessageValidationError, match="discard_count=3 exceeds"):
        assembler.prepare_next_input(
            [
                *checkpoint_messages[1],
                {"role": "tool", "content": "retry", "tool_call_id": "call_1"},
            ]
        )


def test_message_in_rejects_negative_rollback_depth():
    with pytest.raises(ValueError, match="max_assistant_rollback_steps must be >= 0"):
        MessageTrajectoryAssembler(
            TITOTokenizer(FakeTokenizer()), max_assistant_rollback_steps=-1
        )
