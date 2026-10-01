import json
from unittest.mock import patch

from omni_ai_controller.vision import FINANCIAL_FACTS_SCHEMA, OpenAIVisionClient, VisionSettingsStore
from test_vision import FakeResponse


def test_vision_preserves_invoice_date_roles_and_visible_payment_components(tmp_path):
    store = VisionSettingsStore(tmp_path / "vision.json")
    store.save(model="gpt-6-sol", api_key="sk-test-012345678901234567890")
    facts = {"amount_decimal": "20571.00", "document_kind": "sales_report",
             "transaction_time_role": "sales_activity", "transaction_time_iso": "2026-07-01T21:30:00",
             "document_date_iso": "2026-07-01", "due_date_iso": None,
             "amount_components": [{"role": "card", "amount_decimal": "20129.00", "label": "Kort", "currency": "SEK"}]}
    output = {"status": "accepted", "reasons": [], "primary_receipt_index": 1,
              "receipts": [{"text": "POS daily report", "page_start": 1, "page_end": 1,
                            "financial_facts": facts, "classification": {"document_type": "income_voucher"}}]}
    response = {"choices": [{"message": {"content": json.dumps(output)}}]}
    with patch("omni_ai_controller.vision.urlopen", return_value=FakeResponse(response)) as request:
        result = OpenAIVisionClient(store).recognize(b"fake-image", filename="report.jpg", content_type="image/jpeg")
    receipt = result["receipts"][0]
    assert receipt["document_date_iso"] == "2026-07-01"
    assert receipt["relevance"] == "unassessed"
    assert receipt["financial_facts"]["transaction_time_role"] == "sales_activity"
    assert receipt["financial_facts"]["amount_components"] == facts["amount_components"]
    sent = json.loads(request.call_args.args[0].data)
    prompt = sent["messages"][0]["content"][0]["text"]
    assert "neither is an actual transaction_time_iso" in prompt
    assert "Do not decide audit-period" in prompt
    assert set(FINANCIAL_FACTS_SCHEMA["properties"]) == set(FINANCIAL_FACTS_SCHEMA["required"])