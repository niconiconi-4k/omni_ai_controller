import json
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pytest

from omni_ai_controller.client import ModelServerClient, ServerRequestCancelled, ServerRequestError, ServerRequestTimeout
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


def test_chat_applies_optional_output_limit() -> None:
    response = {"choices": [{"message": {"content": "完成"}}]}
    with patch(
        "omni_ai_controller.client.urlopen", return_value=FakeResponse(response)
    ) as request:
        ModelServerClient(config()).chat(
            [{"role": "user", "content": "你好"}],
            enable_thinking=False,
            max_tokens=4096,
        )

    payload = json.loads(request.call_args.args[0].data.decode("utf-8"))
    assert payload["max_tokens"] == 4096


def test_chat_json_sends_strict_schema() -> None:
    response = {"id": "local-1", "choices": [{"message": {"content": "{}"}}]}
    with patch(
        "omni_ai_controller.client.urlopen", return_value=FakeResponse(response)
    ) as request:
        result = ModelServerClient(config()).chat_json(
            [{"role": "user", "content": "分类"}],
            schema_name="test_schema",
            schema={"type": "object", "properties": {}},
        )

    payload = json.loads(request.call_args.args[0].data.decode("utf-8"))
    assert payload["temperature"] == 0
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["response_format"]["json_schema"]["strict"] is True
    assert payload["response_format"]["json_schema"]["name"] == "test_schema"
    assert result.raw["id"] == "local-1"


def test_streaming_json_keeps_content_finish_reason_and_token_usage():
    events = [
        {"id": "stream-1", "choices": [{"delta": {"content": '{"summary":"'}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": '完成"}'}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"total_tokens": 123}},
    ]
    wire = b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events) + b"data: [DONE]\n\n"
    response = BytesIO(wire)
    with patch("omni_ai_controller.client.urlopen", return_value=response) as request:
        result = ModelServerClient(config()).chat_json([], schema_name="test", schema={}, stream=True, timeout=42)
    assert result.content == '{"summary":"完成"}'
    assert result.raw["choices"][0]["finish_reason"] == "stop"
    assert result.raw["usage"]["total_tokens"] == 123
    assert request.call_args.kwargs["timeout"] == 42
    assert json.loads(request.call_args.args[0].data)["stream_options"] == {"include_usage": True}
    assert response.closed


def test_streaming_json_timeout_closes_connection_and_rejects_partial_output():
    response = BytesIO(b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n')
    with patch("omni_ai_controller.client.urlopen", return_value=response), patch(
        "omni_ai_controller.client.monotonic", side_effect=[0, 1, 11],
    ):
        with pytest.raises(ServerRequestTimeout):
            ModelServerClient(config()).chat_json([], schema_name="test", schema={}, stream=True, timeout=10)
    assert response.closed


def test_streaming_json_requires_done_and_a_finish_reason():
    response = BytesIO(b'data: {"choices":[{"delta":{"content":"{}"},"finish_reason":"stop"}]}\n\n')
    with patch("omni_ai_controller.client.urlopen", return_value=response):
        with pytest.raises(ServerRequestError, match="未完整结束"):
            ModelServerClient(config()).chat_json([], schema_name="test", schema={}, stream=True)
    assert response.closed


def test_parent_cancellation_closes_model_stream_without_applying_partial_json():
    response = BytesIO(b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n')
    flags = iter([False, True])
    with patch("omni_ai_controller.client.urlopen", return_value=response):
        with pytest.raises(ServerRequestCancelled):
            ModelServerClient(config()).chat_json([], schema_name="test", schema={}, stream=True, cancelled=lambda: next(flags))
    assert response.closed
