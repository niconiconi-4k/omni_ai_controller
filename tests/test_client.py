import json
from pathlib import Path
from unittest.mock import patch

import pytest

from omni_ai_controller.client import ModelServerClient, ServerRequestError
from omni_ai_controller.config import ServerConfig


class FakeResponse:
    def __init__(self, data: dict[str, object]) -> None:
        self.data = json.dumps(data).encode("utf-8")

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self) -> bytes:
        return self.data


def config() -> ServerConfig:
    return ServerConfig(
        model_dir=Path("/tmp/model"),
        base_url="http://127.0.0.1:8000",
        api_key="api-key",
        control_token="control-token",
        model_name="test-model",
    )


def test_chat_parses_content_and_reasoning() -> None:
    response = {
        "choices": [
            {"message": {"content": "最终答案", "reasoning_content": "推理过程"}}
        ]
    }
    with patch("omni_ai_controller.client.urlopen", return_value=FakeResponse(response)):
        result = ModelServerClient(config()).chat(
            [{"role": "user", "content": "你好"}], enable_thinking=True
        )

    assert result.content == "最终答案"
    assert result.reasoning_content == "推理过程"


def test_chat_rejects_invalid_response() -> None:
    with patch(
        "omni_ai_controller.client.urlopen", return_value=FakeResponse({"choices": []})
    ):
        with pytest.raises(ServerRequestError, match="choices"):
            ModelServerClient(config()).chat(
                [{"role": "user", "content": "你好"}], enable_thinking=False
            )
