import json
from types import SimpleNamespace

import pytest

from omni_ai_controller.voucher_classifier import (
    VoucherClassificationError,
    classify_voucher,
)


class FakeClient:
    def __init__(self, content: str) -> None:
        self.content = content
        self.config = SimpleNamespace(model_name="qwen3.6-27b-instruct")
        self.messages = None
        self.schema = None

    def chat_json(self, messages, *, schema_name, schema, max_tokens=768):  # type: ignore[no-untyped-def]
        self.messages = messages
        self.schema = schema
        return SimpleNamespace(
            content=self.content,
            raw={"id": "qwen-1", "usage": {"total_tokens": 42}},
        )


def test_classify_voucher_returns_allowlisted_result() -> None:
    client = FakeClient(json.dumps({
        "document_type": "expense_voucher",
        "is_certain": True,
        "confidence": 0.92,
        "reason": "供应商发票",
        "evidence": ["Invoice"],
    }))

    result = classify_voucher(
        client,  # type: ignore[arg-type]
        text="Invoice 100 SEK",
        financial_facts={"amount_decimal": "100.00"},
    )

    assert result["classification"]["document_type"] == "expense_voucher"
    assert result["model"] == "qwen3.6-27b-instruct"
    assert result["usage"]["total_tokens"] == 42
    assert "未经信任" in client.messages[0]["content"]
    assert "bank_voucher" not in client.schema["properties"]["document_type"]["enum"]


def test_classify_voucher_rejects_invalid_json() -> None:
    with pytest.raises(VoucherClassificationError, match="格式无效"):
        classify_voucher(
            FakeClient("not-json"),  # type: ignore[arg-type]
            text="Invoice",
            financial_facts={},
        )
