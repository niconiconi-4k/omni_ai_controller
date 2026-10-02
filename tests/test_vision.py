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


def test_unconfigured_vision_defaults_to_gpt6_without_overwriting_explicit_model(tmp_path):
    store = VisionSettingsStore(tmp_path / "vision.json")
    status = store.status()
    assert status["model"] == "gpt-6.1-sol"
    assert [item["id"] for item in status["models"] if item["default"]] == ["gpt-6.1-sol"]
    store.save(model="gpt-4.1", api_key="sk-test-012345678901234567890")
    assert store.status()["model"] == "gpt-4.1"


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


def test_settings_store_supports_gpt_6_sol(tmp_path: Path) -> None:
    store = VisionSettingsStore(tmp_path / "vision.json")
    store.save(model="gpt-4o", api_key="sk-test-012345678901234567890")

    status = store.save(model="gpt-6-sol")

    assert status["model"] == "gpt-6-sol"
    assert any(model["id"] == "gpt-6-sol" for model in status["models"])
    model, _ = store.credentials()
    assert model == "gpt-6-sol"


def test_openai_vision_normalizes_receipt_result(tmp_path: Path) -> None:
    store = VisionSettingsStore(tmp_path / "vision.json")
    store.save(model="gpt-4o", api_key="sk-test-012345678901234567890")
    response = {
        "id": "request-1",
        "model": "gpt-4o-2024-11-20",
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
                            "financial_facts": {
                                "amount_text": "123,45 SEK",
                                "amount_decimal": "123.45",
                                "currency": "SEK",
                                "reference_numbers": ["RF18 5390"],
                                "account_numbers": ["**** 4242"],
                                "transaction_time_text": "2026-09-26 10:00",
                                "transaction_time_iso": "2026-09-26T10:00:00+02:00",
                                "payer": {"name": "Buyer AB", "organization_number": "556000-0001"},
                                "payee": {"name": "Seller AB", "organization_number": "556000-0002"},
                            },
                            "classification": {
                                "document_type": "expense_voucher",
                                "is_certain": True,
                                "confidence": 0.97,
                                "reason": "供应商发票",
                                "evidence": ["Invoice", "Amount due"],
                            },
                        }
                    )
                }
            }
        ],
        "usage": {
            "prompt_tokens": 80,
            "completion_tokens": 20,
            "total_tokens": 100,
            "prompt_tokens_details": {"cached_tokens": 10},
        },
    }

    with patch("omni_ai_controller.vision.urlopen", return_value=FakeResponse(response)) as request:
        result = OpenAIVisionClient(store).recognize(
            b"image-bytes",
            filename="receipt.png",
            content_type="image/png",
            subject_company_name="Buyer AB",
            audit_period_start="2026-09-01",
            audit_period_end="2026-09-30",
        )

    assert result["request_id"] == "request-1"
    assert result["model"] == {"provider": "openai", "vision": "gpt-4o-2024-11-20"}
    assert result["usage"]["prompt_tokens"] == 80
    assert result["usage"]["completion_tokens"] == 20
    assert result["usage"]["prompt_tokens_details"]["cached_tokens"] == 10
    assert result["receipts"][0]["payment_candidates"][0]["amounts"] == ["123,45"]
    assert result["financial_facts"]["account_numbers"] == ["**** 4242"]
    assert result["classification"]["document_type"] == "expense_voucher"
    sent_request = request.call_args.args[0]
    sent_payload = json.loads(sent_request.data.decode("utf-8"))
    assert sent_request.full_url == "https://api.openai.com/v1/chat/completions"
    assert sent_request.get_header("Authorization").startswith("Bearer sk-test-")
    schema = sent_payload["response_format"]["json_schema"]["schema"]
    assert "financial_facts" in schema["properties"]
    assert "classification" in schema["properties"]
    assert "receipts" in schema["properties"]
    receipt_schema = schema["properties"]["receipts"]["items"]
    assert receipt_schema["properties"]["relevance"]["enum"] == ["unassessed"]
    facts_schema = receipt_schema["properties"]["financial_facts"]
    assert facts_schema["properties"]["amount_effect"]["enum"] == [
        "normal", "reversal", "unknown",
    ]
    assert "amount_effect" in facts_schema["required"]
    assert "bank_voucher" not in schema["properties"]["classification"]["properties"]["document_type"]["enum"]
    instruction = sent_payload["messages"][0]["content"][0]["text"]
    assert 'subject company name is "Buyer AB"' in instruction
    assert "determine transaction direction before document type" in instruction
    assert "An invoice is not inherently income" in instruction
    assert "Kreditfaktura/Kreditnota" in instruction
    assert "intrinsic accounting role" in instruction
    assert "ownership is unverified" in instruction
    assert "must be negative for a reversal" in instruction
    assert "Do not negate an ordinary document merely because it is classified as an expense" in instruction
    assert 'audit period is "2026-09-01" through "2026-09-30"' not in instruction
    assert "Do not decide audit-period" in instruction
    assert "Do not transcribe advertisements" in instruction
    assert "Split every distinct financial document" in instruction


def test_openai_vision_requires_configuration(tmp_path: Path) -> None:
    client = OpenAIVisionClient(VisionSettingsStore(tmp_path / "missing.json"))

    with pytest.raises(VisionRequestError, match="尚未配置"):
        client.recognize(b"image", filename="receipt.png", content_type="image/png")


def test_openai_vision_sends_ordered_pdf_pages_and_layout_text(tmp_path: Path) -> None:
    store = VisionSettingsStore(tmp_path / "vision.json")
    store.save(model="gpt-4o", api_key="sk-test-012345678901234567890")
    response = {
        "id": "request-pdf-1",
        "model": "gpt-6-sol",
        "choices": [{"message": {"content": json.dumps({
            "status": "accepted", "reasons": [], "text": "PAGE ONE\nPAGE TWO",
            "payment_candidates": [],
        })}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
    }

    with patch("omni_ai_controller.vision.urlopen", return_value=FakeResponse(response)) as request:
        result = OpenAIVisionClient(store).recognize_document(
            [
                (b"page-one", "page-001.jpg", "image/jpeg", 1),
                (b"page-two", "page-002.jpg", "image/jpeg", 2),
            ],
            document_text="=== PDF PAGE 1/2 ===\nInvoice\n\n=== PDF PAGE 2/2 ===\nPaid",
            model_override="gpt-6-sol",
            classify=False,
        )

    sent_payload = json.loads(request.call_args.args[0].data.decode("utf-8"))
    content = sent_payload["messages"][0]["content"]
    assert [item["type"] for item in content].count("image_url") == 2
    assert "PDF PAGE 1/2" in content[1]["text"]
    assert content[2]["text"].startswith("Rendered page 1/2")
    assert content[4]["text"].startswith("Rendered page 2/2")
    assert result["image"]["page_count"] == 2


def test_openai_vision_uses_verified_text_layer_without_images(tmp_path: Path) -> None:
    store = VisionSettingsStore(tmp_path / "vision.json")
    store.save(model="gpt-4o", api_key="sk-test-012345678901234567890")
    response = {
        "id": "request-text-pdf-1",
        "model": "gpt-6-sol",
        "choices": [{"message": {"content": json.dumps({
            "status": "accepted", "reasons": [], "primary_receipt_index": None,
            "text": "12 July payslips", "financial_facts": {}, "classification": {},
            "receipts": [{
                "page_start": 1, "page_end": 1, "relevance": "excluded",
                "relevance_reason": "July document outside September audit",
                "document_date_text": "2026-07-31", "document_date_iso": "2026-07-31",
                "due_date_text": None, "due_date_iso": None,
                    "text": "July payslip", "payment_candidates": [],
                    "financial_facts": {},
                    "classification": {"document_type": "payroll_voucher"},
            }],
        })}}],
        "usage": {"prompt_tokens": 500, "completion_tokens": 100, "total_tokens": 600},
    }

    with patch("omni_ai_controller.vision.urlopen", return_value=FakeResponse(response)) as request:
        result = OpenAIVisionClient(store).recognize_document(
            [],
            page_count_override=12,
            document_text="=== PDF PAGE 1/12 ===\nLönebesked 2026-07-31",
            model_override="gpt-6-sol",
            audit_period_start="2026-09-01",
            audit_period_end="2026-09-30",
        )

    sent_payload = json.loads(request.call_args.args[0].data.decode("utf-8"))
    content = sent_payload["messages"][0]["content"]
    assert not any(item["type"] == "image_url" for item in content)
    assert "already passed local quality checks" in content[0]["text"]
    assert "at most 500 characters" in content[0]["text"]
    assert "relevance=unassessed" in content[0]["text"]
    assert "deferred_to_audit" in content[0]["text"]
    assert "Nettolön" in content[0]["text"]
    assert result["status"] == "accepted"
    assert result["primary_receipt_index"] == 1
    assert result["receipts"][0]["relevance"] == "unassessed"
    assert result["receipts"][0]["relevance_reason"] == "deferred_to_audit"
    assert result["image"]["page_count"] == 12
    assert result["image"]["recognition_mode"] == "text_layer"
    assert result["receipts"][0]["financial_extraction_complete"] is False
    assert result["receipts"][0]["financial_extraction_issue"] == "payroll_final_amount_missing"


def test_openai_vision_defers_all_receipt_relevance_to_audit(tmp_path: Path) -> None:
    store = VisionSettingsStore(tmp_path / "vision.json")
    store.save(model="gpt-4o", api_key="sk-test-012345678901234567890")

    def facts(amount: str) -> dict[str, object]:
        return {
            "amount_text": f"{amount} SEK", "amount_decimal": amount, "currency": "SEK",
            "reference_numbers": [], "account_numbers": [],
            "transaction_time_text": None, "transaction_time_iso": None,
            "payer": {"name": "Buyer AB", "organization_number": None},
            "payee": {"name": "Seller AB", "organization_number": None},
        }

    classification = {
        "document_type": "expense_voucher", "is_certain": True, "confidence": 0.96,
        "reason": "supplier invoice", "evidence": ["Fakturamottagare Buyer AB"],
    }
    response = {
        "id": "request-multi-1", "model": "gpt-6-sol",
        "choices": [{"message": {"content": json.dumps({
            "status": "accepted", "reasons": [], "primary_receipt_index": 2,
            "text": "Two invoices", "financial_facts": facts("999.00"),
            "classification": classification,
            "receipts": [
                {
                    "page_start": 1, "page_end": 1, "relevance": "excluded",
                    "relevance_reason": "February invoice outside September audit",
                    "document_date_text": "2026-02-01", "document_date_iso": "2026-02-01",
                    "due_date_text": "2026-02-28", "due_date_iso": "2026-02-28",
                    "text": "Old invoice total 999 SEK", "payment_candidates": [],
                    "financial_facts": facts("999.00"), "classification": classification,
                },
                {
                    "page_start": 2, "page_end": 2, "relevance": "included",
                    "relevance_reason": "September invoice",
                    "document_date_text": "2026-09-03", "document_date_iso": "2026-09-03",
                    "due_date_text": "2026-09-30", "due_date_iso": "2026-09-30",
                    "text": "Current invoice total 123 SEK", "payment_candidates": [],
                    "financial_facts": facts("123.00"), "classification": classification,
                },
            ],
        })}}],
        "usage": {},
    }

    with patch("omni_ai_controller.vision.urlopen", return_value=FakeResponse(response)):
        result = OpenAIVisionClient(store).recognize_document(
            [(b"old", "page-001.jpg", "image/jpeg", 1),
             (b"current", "page-002.jpg", "image/jpeg", 2)],
            document_text="old invoice\ncurrent invoice",
            audit_period_start="2026-09-01",
            audit_period_end="2026-09-30",
        )

    assert result["status"] == "accepted"
    assert result["primary_receipt_index"] == 2
    assert result["receipts"][0]["relevance"] == "unassessed"
    assert result["receipts"][1]["relevance"] == "unassessed"
    assert all(
        receipt["relevance_reason"] == "deferred_to_audit"
        for receipt in result["receipts"]
    )
    assert result["financial_facts"]["amount_decimal"] == "123.00"


def test_gpt_6_sol_uses_compatible_chat_completion_parameters(tmp_path: Path) -> None:
    store = VisionSettingsStore(tmp_path / "vision.json")
    store.save(model="gpt-4o", api_key="sk-test-012345678901234567890")
    response = {
        "id": "request-sol-1",
        "model": "gpt-6-sol",
        "choices": [{"message": {"content": json.dumps({
            "status": "accepted", "reasons": [], "text": "TOTAL 10.00 SEK",
            "payment_candidates": [],
        })}}],
        "usage": {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
    }

    with patch("omni_ai_controller.vision.urlopen", return_value=FakeResponse(response)) as request:
        result = OpenAIVisionClient(store).recognize(
            b"image-bytes",
            filename="receipt.png",
            content_type="image/png",
            model_override="gpt-6-sol",
            classify=False,
        )

    sent_payload = json.loads(request.call_args.args[0].data.decode("utf-8"))
    assert sent_payload["model"] == "gpt-6-sol"
    assert sent_payload["reasoning_effort"] == "none"
    assert sent_payload["max_completion_tokens"] == 8192
    assert "max_tokens" not in sent_payload
    assert "Do not classify this document" in sent_payload["messages"][0]["content"][0]["text"]
    assert result["model"]["vision"] == "gpt-6-sol"
