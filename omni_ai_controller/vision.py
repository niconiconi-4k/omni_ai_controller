from __future__ import annotations

import base64
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

SUPPORTED_VISION_MODELS: dict[str, dict[str, object]] = {
    "gpt-4o": {
        "label": "GPT-4o",
        "description": "成熟的多模态识图模型，128K 上下文",
        "default": True,
    },
    "gpt-4.1": {
        "label": "GPT-4.1",
        "description": "指令遵循更强，约 1M 上下文",
        "default": False,
    },
    "gpt-6-sol": {
        "label": "GPT-6 Sol",
        "description": "适合复杂凭证与银行流水识别，约 1M 上下文",
        "default": False,
    },
}
DEFAULT_VISION_MODEL = "gpt-4o"
OPENAI_CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"
MAX_VISION_IMAGE_BYTES = 20 * 1024 * 1024


class VisionSettingsError(RuntimeError):
    """Raised when protected OpenAI vision settings cannot be read or changed."""


class VisionRequestError(RuntimeError):
    def __init__(self, message: str, *, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


class VisionSettingsStore:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        if self.path.is_symlink():
            raise VisionSettingsError("OpenAI 识图配置不能是符号链接")
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise VisionSettingsError("无法读取 OpenAI 识图配置") from exc
        if not isinstance(payload, dict):
            raise VisionSettingsError("OpenAI 识图配置格式无效")
        return payload

    def status(self) -> dict[str, Any]:
        payload = self._read()
        model = str(payload.get("model") or DEFAULT_VISION_MODEL)
        if model not in SUPPORTED_VISION_MODELS:
            model = DEFAULT_VISION_MODEL
        return {
            "provider": "openai",
            "model": model,
            "configured": bool(payload.get("api_key")),
            "updated_at": payload.get("updated_at"),
            "models": [
                {"id": model_id, **metadata}
                for model_id, metadata in SUPPORTED_VISION_MODELS.items()
            ],
        }

    def save(self, *, model: str, api_key: str | None = None) -> dict[str, Any]:
        if model not in SUPPORTED_VISION_MODELS:
            raise VisionSettingsError("不支持该 OpenAI 识图模型")
        current = self._read()
        normalized_key = api_key.strip() if api_key is not None else ""
        if api_key is not None:
            if len(normalized_key) < 20 or len(normalized_key) > 512:
                raise VisionSettingsError("OpenAI API 密钥长度无效")
            if any(character.isspace() for character in normalized_key):
                raise VisionSettingsError("OpenAI API 密钥不能包含空白字符")
        else:
            normalized_key = str(current.get("api_key") or "")
        if not normalized_key:
            raise VisionSettingsError("首次配置必须填写 OpenAI API 密钥")

        payload = {
            "provider": "openai",
            "model": model,
            "api_key": normalized_key,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            if self.path.parent.is_symlink():
                raise VisionSettingsError("OpenAI 识图配置目录不能是符号链接")
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=".openai-vision-",
                suffix=".json",
                dir=self.path.parent,
                text=True,
            )
            try:
                os.fchmod(descriptor, 0o600)
                with os.fdopen(descriptor, "w", encoding="utf-8") as target:
                    json.dump(payload, target, ensure_ascii=False)
                    target.write("\n")
                    target.flush()
                    os.fsync(target.fileno())
                os.replace(temporary_name, self.path)
                os.chmod(self.path, 0o600)
            except Exception:
                try:
                    os.unlink(temporary_name)
                except OSError:
                    pass
                raise
        except VisionSettingsError:
            raise
        except OSError as exc:
            raise VisionSettingsError("无法保存 OpenAI 识图配置") from exc
        return self.status()

    def remove_key(self) -> dict[str, Any]:
        try:
            if self.path.is_symlink():
                raise VisionSettingsError("OpenAI 识图配置不能是符号链接")
            self.path.unlink(missing_ok=True)
        except OSError as exc:
            raise VisionSettingsError("无法删除 OpenAI API 密钥") from exc
        return self.status()

    def credentials(self) -> tuple[str, str]:
        payload = self._read()
        api_key = str(payload.get("api_key") or "")
        model = str(payload.get("model") or DEFAULT_VISION_MODEL)
        if not api_key:
            raise VisionRequestError("OpenAI 识图模型尚未配置 API 密钥", status_code=503)
        if model not in SUPPORTED_VISION_MODELS:
            raise VisionRequestError("OpenAI 识图模型配置无效", status_code=503)
        return model, api_key


class OpenAIVisionClient:
    def __init__(self, settings: VisionSettingsStore) -> None:
        self.settings = settings

    def recognize(
        self,
        image: bytes,
        *,
        filename: str,
        content_type: str,
    ) -> dict[str, Any]:
        if content_type not in {"image/jpeg", "image/png", "image/webp"}:
            raise VisionRequestError("OpenAI 识图仅支持 JPEG、PNG 和 WebP", status_code=415)
        if not image or len(image) > MAX_VISION_IMAGE_BYTES:
            raise VisionRequestError("图片必须介于 1 字节和 20 MiB 之间", status_code=413)
        model, api_key = self.settings.credentials()
        encoded = base64.b64encode(image).decode("ascii")
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Analyze this financial receipt or invoice. Extract all visible text and "
                                "identify the final amount actually paid or payable. Currency may be SEK, "
                                "EUR, USD, CNY or another ISO currency. Do not invent unreadable values. "
                                "Return only data matching the supplied JSON schema."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{content_type};base64,{encoded}",
                                "detail": "high",
                            },
                        },
                    ],
                }
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "receipt_vision_result",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "status": {
                                "type": "string",
                                "enum": ["accepted", "needs_manual_confirmation", "needs_reupload"],
                            },
                            "reasons": {"type": "array", "items": {"type": "string"}},
                            "text": {"type": "string"},
                            "payment_candidates": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "properties": {
                                        "text": {"type": "string"},
                                        "amount": {"type": "string"},
                                        "currency": {"type": ["string", "null"]},
                                        "keyword": {"type": ["string", "null"]},
                                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                                    },
                                    "required": ["text", "amount", "currency", "keyword", "confidence"],
                                },
                            },
                        },
                        "required": ["status", "reasons", "text", "payment_candidates"],
                    },
                },
            },
        }
        if model == "gpt-6-sol":
            payload["reasoning_effort"] = "none"
            payload["temperature"] = 0
            payload["max_completion_tokens"] = 4096
        else:
            payload["temperature"] = 0
            payload["max_tokens"] = 4096
        request = Request(
            OPENAI_CHAT_COMPLETIONS_URL,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urlopen(request, timeout=180) as response:
                raw = response.read()
        except HTTPError as exc:
            messages = {
                400: "OpenAI 拒绝了图片或请求参数",
                401: "OpenAI API 密钥无效或没有模型访问权限",
                403: "OpenAI API 账户无权调用该模型",
                404: "所选 OpenAI 模型当前不可用",
                413: "图片超过 OpenAI 接口限制",
                429: "OpenAI API 已达到速率或额度限制",
            }
            raise VisionRequestError(
                messages.get(exc.code, "OpenAI 识图服务返回错误"),
                status_code=exc.code if exc.code in messages else 502,
            ) from exc
        except (URLError, TimeoutError) as exc:
            raise VisionRequestError("无法连接 OpenAI 识图服务", status_code=504) from exc

        try:
            response_payload = json.loads(raw.decode("utf-8"))
            content = response_payload["choices"][0]["message"]["content"]
            if isinstance(content, list):
                content = "".join(
                    str(item.get("text") or "") for item in content if isinstance(item, dict)
                )
            result = json.loads(str(content))
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise VisionRequestError("OpenAI 返回了无效的识图结果") from exc
        if not isinstance(result, dict):
            raise VisionRequestError("OpenAI 返回了无效的识图结果")

        payment_candidates: list[dict[str, Any]] = []
        for index, candidate in enumerate(result.get("payment_candidates") or []):
            if not isinstance(candidate, dict) or not candidate.get("amount"):
                continue
            currency = str(candidate.get("currency") or "").strip().casefold()
            keyword = str(candidate.get("keyword") or "total").strip().casefold()
            confidence = max(0.0, min(float(candidate.get("confidence") or 0), 1.0))
            payment_candidates.append(
                {
                    "line_index": index,
                    "text": str(candidate.get("text") or ""),
                    "amounts": [str(candidate["amount"])],
                    "keywords": [keyword] if keyword else [],
                    "currencies": [currency] if currency else [],
                    "confidence": confidence,
                }
            )
        status = str(result.get("status") or "needs_manual_confirmation")
        if status not in {"accepted", "needs_manual_confirmation", "needs_reupload"}:
            status = "needs_manual_confirmation"
        response_model = str(response_payload.get("model") or model)
        return {
            "request_id": str(response_payload.get("id") or "openai-vision"),
            "status": status,
            "reasons": [str(value) for value in (result.get("reasons") or [])],
            "model": {"provider": "openai", "vision": response_model},
            "image": {
                "filename": filename,
                "content_type": content_type,
                "size_bytes": len(image),
            },
            "receipts": [
                {
                    "index": 1,
                    "text": str(result.get("text") or ""),
                    "lines": [],
                    "payment_candidates": payment_candidates,
                }
            ],
            "processing_ms": round((time.perf_counter() - started) * 1000, 2),
            "usage": response_payload.get("usage") or {},
        }
