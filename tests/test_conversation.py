from typing import cast

import pytest

from omni_ai_controller.client import ChatResult, ModelServerClient
from omni_ai_controller.conversation import Conversation


class FakeClient:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.received: list[dict[str, str]] = []
        self.thinking = False

    def chat(
        self, messages: list[dict[str, str]], *, enable_thinking: bool
    ) -> ChatResult:
        self.received = list(messages)
        self.thinking = enable_thinking
        if self.fail:
            raise RuntimeError("request failed")
        return ChatResult("回答", "推理", {"id": "test"})


def test_conversation_keeps_multi_turn_history() -> None:
    conversation = Conversation()
    conversation.reset("系统提示", enable_thinking=False)
    client = FakeClient()

    result = conversation.send(cast(ModelServerClient, client), "问题")

    assert result.content == "回答"
    assert client.thinking is False
    assert client.received == [
        {"role": "system", "content": "系统提示"},
        {"role": "user", "content": "问题"},
    ]
    assert conversation.messages[-1] == {"role": "assistant", "content": "回答"}


def test_conversation_rolls_back_failed_message() -> None:
    conversation = Conversation()
    client = FakeClient(fail=True)

    with pytest.raises(RuntimeError, match="request failed"):
        conversation.send(cast(ModelServerClient, client), "问题")

    assert conversation.messages == []
    assert conversation.last_result is None
