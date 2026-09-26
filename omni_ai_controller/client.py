from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import ServerConfig


class ServerRequestError(RuntimeError):
    """Raised when the model server cannot satisfy a request."""


@dataclass(frozen=True)
class ChatResult:
    content: str
    reasoning_content: str
    raw: dict[str, Any]


class ModelServerClient:
    def __init__(self, config: ServerConfig) -> None:
        self.config = config

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
    ) -> ChatResult:
        self.config.require_credentials()
        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        }
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
    ) -> ChatResult:
        self.config.require_credentials()
        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "stream": False,
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

    def _request(
        self,
        method: str,
        path: str,
        *,
        headers: dict[str, str] | None = None,
        payload: dict[str, Any] | None = None,
        timeout: float,
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
            with urlopen(request, timeout=timeout) as response:
                raw = response.read()
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ServerRequestError(f"HTTP {exc.code}: {detail or exc.reason}") from exc
        except URLError as exc:
            raise ServerRequestError(f"无法连接模型服务：{exc.reason}") from exc
        except TimeoutError as exc:
            raise ServerRequestError("请求模型服务超时") from exc

        if not raw:
            return {}
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ServerRequestError("模型服务返回了无效的 JSON") from exc
        if not isinstance(parsed, dict):
            raise ServerRequestError("模型服务返回的 JSON 不是对象")
        return parsed
