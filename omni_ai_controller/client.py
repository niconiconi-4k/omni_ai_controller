from __future__ import annotations

import json
from dataclasses import dataclass
from time import monotonic
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request

from .config import ServerConfig
from .model_transport import model_response, urlopen
from .model_queue import run_model_call
from .request_errors import ServerRequestError, ServerRequestTimeout, ServerRequestCancelled


@dataclass(frozen=True)
class ChatResult:
    content: str
    reasoning_content: str
    raw: dict[str, Any]


class ModelServerClient:
    def __init__(self, config: ServerConfig) -> None:
        self.config = config

    async def async_chat(self, messages: list[dict[str, str]], **kwargs: Any) -> ChatResult:
        return await run_model_call(self.chat, messages, **kwargs)

    async def async_chat_json(self, messages: list[dict[str, str]], **kwargs: Any) -> ChatResult:
        return await run_model_call(self.chat_json, messages, **kwargs)

    def live(self) -> bool:
        try:
            self._request("GET", "/health/live", timeout=3)
            return True
        except ServerRequestError:
            return False

    def status(self) -> dict[str, Any]:
        return self._request("GET", "/status", timeout=10)

    def start_model(self) -> dict[str, Any]:
        self.config.require_credentials()
        return self._request(
            "POST",
            "/control/start",
            headers={"X-Control-Token": self.config.control_token},
            timeout=30,
        )

    def stop_model(self) -> dict[str, Any]:
        self.config.require_credentials()
        return self._request(
            "POST",
            "/control/stop",
            headers={"X-Control-Token": self.config.control_token},
            timeout=45,
        )

    def restart_model(self) -> dict[str, Any]:
        self.config.require_credentials()
        return self._request(
            "POST",
            "/control/restart",
            headers={"X-Control-Token": self.config.control_token},
            timeout=60,
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        enable_thinking: bool,
        max_tokens: int | None = None,
    ) -> ChatResult:
        self.config.require_credentials()
        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        data = self._request(
            "POST",
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.config.api_key}"},
            payload=payload,
            timeout=3600,
        )
        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ServerRequestError("模型响应缺少 choices[0].message") from exc

        content = message.get("content") or ""
        reasoning = message.get("reasoning_content") or ""
        return ChatResult(str(content), str(reasoning), data)

    def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        schema_name: str,
        schema: dict[str, Any],
        max_tokens: int = 768,
        timeout: float = 3600,
        stream: bool = False,
        cancelled: Callable[[], bool] | None = None,
    ) -> ChatResult:
        self.config.require_credentials()
        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "stream": stream,
            "temperature": 0,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": schema,
                },
            },
        }
        if stream:
            payload["stream_options"] = {"include_usage": True}
        data = (self._request_stream if stream else self._request)(
            "POST",
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.config.api_key}"},
            payload=payload,
            timeout=timeout,
            cancelled=cancelled,
        )
        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ServerRequestError("模型响应缺少 choices[0].message") from exc
        content = message.get("content") or ""
        reasoning = message.get("reasoning_content") or ""
        return ChatResult(str(content), str(reasoning), data)

    def _request_stream(
        self, method: str, path: str, *, headers: dict[str, str],
        payload: dict[str, Any], timeout: float,
        cancelled: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        """Keep the upstream response cancellable, with a total wall-clock budget.

        Streaming headers reach the gateway immediately. Closing this response
        also closes its upstream vLLM stream instead of abandoning a non-stream
        request that is still waiting for its first HTTP response.
        """
        request = Request(
            f"{self.config.base_url}{path}", method=method,
            headers={"Accept": "text/event-stream", "Content-Type": "application/json", **headers},
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        )
        deadline = monotonic() + timeout
        content: list[str] = []
        reasoning: list[str] = []
        raw: dict[str, Any] = {}
        finish_reason = None
        completed = False
        try:
            with model_response("local", request, timeout=timeout, opener=urlopen, cancelled=cancelled) as response:
                while True:
                    if cancelled is not None and cancelled():
                        raise ServerRequestCancelled("上层审核已结束等待；关闭当前模型流，不再提交后续分片")
                    remaining = deadline - monotonic()
                    if remaining <= 0:
                        raise TimeoutError("智能审计单步计算达到时间预算")
                    # urllib timeouts are per read: constrain each socket read
                    # to the remaining total wall-clock budget as well.
                    socket = getattr(getattr(getattr(response, "fp", None), "raw", None), "_sock", None)
                    if socket is not None:
                        socket.settimeout(remaining)
                    line = response.readline()
                    if not line:
                        break
                    if not line.startswith(b"data:"):
                        continue
                    event = line[5:].strip()
                    if event == b"[DONE]":
                        completed = True
                        break
                    chunk = json.loads(event)
                    if chunk.get("error"):
                        raise ServerRequestError("模型流式响应报告错误")
                    if chunk.get("id"):
                        raw["id"] = chunk["id"]
                    if isinstance(chunk.get("usage"), dict):
                        raw["usage"] = chunk["usage"]
                    for choice in chunk.get("choices") or []:
                        delta = choice.get("delta") or {}
                        content.append(str(delta.get("content") or ""))
                        reasoning.append(str(delta.get("reasoning_content") or ""))
                        finish_reason = choice.get("finish_reason") or finish_reason
        except HTTPError as exc:
            raise ServerRequestError(f"HTTP {exc.code}: 模型流式请求失败") from exc
        except TimeoutError as exc:
            raise ServerRequestTimeout("智能审计单步模型请求超时") from exc
        except URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise ServerRequestTimeout("智能审计单步模型请求超时") from exc
            raise ServerRequestError("智能审计模型连接中断") from exc
        except (ValueError, AttributeError, UnicodeDecodeError) as exc:
            raise ServerRequestError("模型服务返回无效的流式 JSON") from exc
        if not completed or finish_reason is None:
            raise ServerRequestError("模型流式响应未完整结束，不能应用部分输出")
        raw["choices"] = [{"finish_reason": finish_reason, "message": {
            "content": "".join(content), "reasoning_content": "".join(reasoning),
        }}]
        return raw

    def _request(
        self,
        method: str,
        path: str,
        *,
        headers: dict[str, str] | None = None,
        payload: dict[str, Any] | None = None,
        timeout: float,
        cancelled: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        request_headers = {"Accept": "application/json", **(headers or {})}
        body = None
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            request_headers["Content-Type"] = "application/json"

        request = Request(
            f"{self.config.base_url}{path}",
            data=body,
            headers=request_headers,
            method=method,
        )
        try:
            # Health/control requests must not wait behind inference.
            response_context = (
                model_response("local", request, timeout=timeout, opener=urlopen, cancelled=cancelled)
                if path == "/v1/chat/completions" else urlopen(request, timeout=timeout)
            )
            with response_context as response:
                raw = response.read()
        except HTTPError as exc:
            exc.close()
            raise ServerRequestError(f"HTTP {exc.code}: 模型服务请求失败") from None
        except URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise ServerRequestTimeout("请求模型服务超时") from exc
            raise ServerRequestError("无法连接模型服务") from None
        except TimeoutError as exc:
            raise ServerRequestTimeout("请求模型服务超时") from exc

        if not raw:
            return {}
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ServerRequestError("模型服务返回了无效的 JSON") from exc
        if not isinstance(parsed, dict):
            raise ServerRequestError("模型服务返回的 JSON 不是对象")
        return parsed
