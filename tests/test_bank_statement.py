from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from omni_ai_controller.bank_statement import OpenAIBankStatementClient
from omni_ai_controller.vision import VisionSettingsStore


class FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


def test_statement_client_forces_gpt6_and_strict_transaction_schema(tmp_path: Path) -> None:
    store = VisionSettingsStore(tmp_path / "vision.json")
    store.save(model="gpt-4o", api_key="sk-test-012345678901234567890")
    response = {
        "id": "statement-request-1",
        "model": "gpt-6-sol",
        "choices": [{"message": {"content": json.dumps({
            "status": "accepted",
            "warnings": [],
            "statement": {
                "institution_name": "Example Bank",
                "account_holder": "Example AB",
                "account_number": "****1234",
                "iban": None,
                "bic": None,
                "currency": "SEK",
                "period_start": "2026-09-01",
                "period_end": "2026-09-30",
            },
            "transactions": [{
                "row_index": 1,
                "booking_date": "2026-09-02",
                "value_date": "2026-09-02",
                "transaction_time_text": None,
                "amount": "-100.00",
                "currency": "SEK",
                "direction": "debit",
                "description": "CARD PURCHASE",
                "counterparty": "Shop AB",
                "reference": "REF-1",
                "verification_code": "CTRL-1",
                "transaction_type": "card",
                "source_page": 1,
                "source_row": None,
                "confidence": 0.98,
            }],
            "coverage": {"transaction_count": 1, "possibly_truncated": False, "reason": ""},
        })}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    }

    with patch("omni_ai_controller.bank_statement.urlopen", return_value=FakeResponse(response)) as request:
        result = OpenAIBankStatementClient(store).recognize(
            pages=[(b"page-one", "page-001.jpg", "image/jpeg", 1)],
            document_text="=== PDF PAGE 1/1 ===\nTransaction table",
            source_kind="pdf_hybrid",
            filename="statement.pdf",
        )

    sent_payload = json.loads(request.call_args.args[0].data.decode("utf-8"))
    assert sent_payload["model"] == "gpt-6-sol"
    assert sent_payload["reasoning_effort"] == "none"
    assert sent_payload["max_completion_tokens"] == 32768
    assert sent_payload["response_format"]["json_schema"]["strict"] is True
    transaction_schema = sent_payload["response_format"]["json_schema"]["schema"]["properties"]["transactions"]["items"]
    assert transaction_schema["additionalProperties"] is False
    assert "verification_code" in transaction_schema["required"]
    assert "balance_after" not in transaction_schema["properties"]
    statement_schema = sent_payload["response_format"]["json_schema"]["schema"]["properties"]["statement"]
    assert "opening_balance" not in statement_schema["properties"]
    assert "closing_balance" not in statement_schema["properties"]
    prompt = sent_payload["messages"][0]["content"][0]["text"]
    assert "Ignore navigation" in prompt
    assert "Debit/outgoing amounts must be negative" in prompt
    assert result["transactions"][0]["reference"] == "REF-1"
    assert result["source"] == {"filename": "statement.pdf", "kind": "pdf_hybrid", "page_count": 1}
