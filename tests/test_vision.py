import json
import stat
from pathlib import Path
from unittest.mock import patch

import pytest

from omni_ai_controller.vision import (
    OpenAIVisionClient,
    VisionRequestError,
    VisionSettingsStore,
)


class FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


def test_settings_store_never_returns_key_and_uses_private_mode(tmp_path: Path) -> None:
    path = tmp_path / "openai-vision.json"
    store = VisionSettingsStore(path)

    status = store.save(model="gpt-4o", api_key="sk-test-012345678901234567890")

    assert status["configured"] is True
    assert status["model"] == "gpt-4o"
    assert "api_key" not in status
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert "sk-test" in path.read_text(encoding="utf-8")

    removed = store.remove_key()
    assert removed["configured"] is False
    assert not path.exists()


def test_settings_store_can_change_model_without_reentering_key(tmp_path: Path) -> None:
    store = VisionSettingsStore(tmp_path / "vision.json")
    store.save(model="gpt-4o", api_key="sk-test-012345678901234567890")

    status = store.save(model="gpt-4.1")

    assert status["model"] == "gpt-4.1"
    model, key = store.credentials()
    assert model == "gpt-4.1"
    assert key.startswith("sk-test-")


def test_openai_vision_normalizes_receipt_result(tmp_path: Path) -> None:
    store = VisionSettingsStore(tmp_path / "vision.json")
    store.save(model="gpt-4o", api_key="sk-test-012345678901234567890")
    response = {
        "id": "request-1",
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "status": "accepted",
                            "reasons": [],
                            "text": "TOTAL 123,45 SEK",
                            "payment_candidates": [
                                {
                                    "text": "TOTAL 123,45 SEK",
                                    "amount": "123,45",
                                    "currency": "SEK",
                                    "keyword": "total",
                                    "confidence": 0.98,
                                }
                            ],
                        }
                    )
                }
            }
        ],
        "usage": {"total_tokens": 100},
    }

    with patch("omni_ai_controller.vision.urlopen", return_value=FakeResponse(response)) as request:
        result = OpenAIVisionClient(store).recognize(
            b"image-bytes",
            filename="receipt.png",
            content_type="image/png",
        )

    assert result["request_id"] == "request-1"
    assert result["model"] == {"provider": "openai", "vision": "gpt-4o"}
    assert result["receipts"][0]["payment_candidates"][0]["amounts"] == ["123,45"]
    sent_request = request.call_args.args[0]
    assert sent_request.full_url == "https://api.openai.com/v1/chat/completions"
    assert sent_request.get_header("Authorization").startswith("Bearer sk-test-")


def test_openai_vision_requires_configuration(tmp_path: Path) -> None:
    client = OpenAIVisionClient(VisionSettingsStore(tmp_path / "missing.json"))

    with pytest.raises(VisionRequestError, match="尚未配置"):
        client.recognize(b"image", filename="receipt.png", content_type="image/png")
