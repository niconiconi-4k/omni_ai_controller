import json
from unittest.mock import patch

import pytest

from omni_ai_controller.vision import FINANCIAL_FACTS_SCHEMA, OpenAIVisionClient, VisionRequestError, VisionSettingsStore
from test_vision import FakeResponse


def client(tmp_path):
    store = VisionSettingsStore(tmp_path / "vision.json")
    store.save(model="gpt-6.1-sol", api_key="sk-test-not-a-real-secret-1234567890")
    return OpenAIVisionClient(store)


def response(receipts=None, reviews=None):
    return {"model": "gpt-6.1-sol", "id": "fake-vision",
            "choices": [{"message": {"content": json.dumps({
                "status": "accepted", "primary_receipt_index": 1, "receipts": receipts or [],
                "page_reviews": reviews or [{"page_number": 1, "disposition": "advertisement", "reason": "Ad"}],
                "financial_facts": {"amount_decimal": "999999"}, "classification": {},
            })}}]}


@pytest.mark.parametrize("hint", ["auto", "receipt", "invoice"])
def test_photograph_sol61_uses_vision_structured_output_and_supported_reasoning(tmp_path, hint):
    with patch("omni_ai_controller.vision.urlopen", return_value=FakeResponse(response())) as send:
        result = client(tmp_path).recognize(b"fake-image", filename="photo.jpg", content_type="image/jpeg", document_hint=hint)
    payload = json.loads(send.call_args.args[0].data)
    assert payload["model"] == "gpt-6.1-sol"
    assert payload["reasoning_effort"] == "low"
    assert payload["max_completion_tokens"] >= 8192
    assert "temperature" not in payload and "max_tokens" not in payload
    prompt = payload["messages"][0]["content"][0]["text"]
    for phrase in ("photograph, not a PDF", "Amount is the primary", "Refunds remain signed negative",
                   "taxable bases", "actual payment date", f"hint is {hint}"):
        assert phrase in prompt
    schema = payload["response_format"]["json_schema"]
    assert schema["strict"] is True
    assert "page_reviews" in schema["schema"]["required"]
    assert set(FINANCIAL_FACTS_SCHEMA["properties"]) == set(FINANCIAL_FACTS_SCHEMA["required"])
    assert "account_numbers" in FINANCIAL_FACTS_SCHEMA["properties"]["payer"]["required"]
    assert "taxes" in FINANCIAL_FACTS_SCHEMA["required"]
    assert result["receipts"] == []  # Do not manufacture an invoice from an advertisement.
    assert result["financial_facts"] == {}
    assert result["status"] == "needs_manual_confirmation"


@pytest.mark.parametrize("pages,text,count", [([], "hidden", 1),
    ([(b"x", "p.jpg", "image/jpeg", 1)], "untrusted text", 1),
    ([(b"x", "p.jpg", "image/jpeg", 1)] * 2, "", 2)])
def test_visual_pdf_rejects_text_layer_missing_or_duplicate_pages(tmp_path, pages, text, count):
    with patch("omni_ai_controller.vision.urlopen") as send:
        with pytest.raises(VisionRequestError):
            client(tmp_path).recognize_document(pages, document_text=text, page_count_override=count, source_kind="pdf_rendered")
        send.assert_not_called()


def test_visual_pdf_preserves_separate_vouchers_continuations_taxes_refund_and_page_reviews(tmp_path):
    facts = {"amount_decimal": "-125.00", "amount_effect": "reversal", "document_kind": "invoice",
             "document_date_iso": "2026-06-10", "due_date_iso": "2026-07-10",
             "payer": {"account_numbers": ["payer-test"]}, "payee": {"account_numbers": ["payee-test"]},
             "taxes": [{"label": "Moms", "rate_percent": "25", "amount_decimal": "-25.00"}]}
    receipts = [{"page_start": 1, "page_end": 3, "source_pages": [1, 3], "financial_facts": facts},
                {"page_start": 4, "page_end": 4, "source_pages": [4], "financial_facts": {"amount_decimal": "100"}}]
    reviews = [{"page_number": number, "disposition": disposition, "reason": "short"}
               for number, disposition in enumerate(["financial", "terms", "continuation", "financial"], 1)]
    pages = [(b"fake", f"p{number}.jpg", "image/jpeg", number) for number in range(1, 5)]
    with patch("omni_ai_controller.vision.urlopen", return_value=FakeResponse(response(receipts, reviews))) as send:
        result = client(tmp_path).recognize_document(pages, document_text=None, page_count_override=4, source_kind="pdf_rendered")
    content = json.loads(send.call_args.args[0].data)["messages"][0]["content"]
    assert sum(item["type"] == "image_url" for item in content) == 4
    assert "Inspect ALL pages once" in content[0]["text"]
    assert "not permission to attach unrelated" in content[0]["text"]
    assert result["receipts"][0]["financial_facts"] == facts
    assert result["receipts"][0]["source_pages"] == [1, 3]
    assert len(result["receipts"]) == 2
    assert all(item["relevance"] == "unassessed" for item in result["receipts"])
    assert result["page_reviews"] == reviews


def test_advertisement_cannot_create_a_financial_candidate(tmp_path):
    receipts = [{"page_start": 1, "source_pages": [1], "financial_facts": {"amount_decimal": "9.99"}}]
    with patch("omni_ai_controller.vision.urlopen", return_value=FakeResponse(response(receipts))):
        result = client(tmp_path).recognize(b"fake", filename="ad.jpg", content_type="image/jpeg")
    assert result["receipts"] == [] and result["financial_facts"] == {}