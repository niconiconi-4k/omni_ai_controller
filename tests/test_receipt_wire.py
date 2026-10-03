from copy import deepcopy
import json
from unittest.mock import patch

import pytest

from omni_ai_controller.receipt_wire import DERIVED_RECEIPT_FIELDS, compact_receipt_schema
from omni_ai_controller.vision import FINANCIAL_FACTS_SCHEMA, RECEIPT_RESULT_SCHEMA, OpenAIVisionClient, VisionSettingsStore
from test_vision import FakeResponse


def receipt_fixture(pages, *, classify=True):
    classification = {
        "document_type": "expense_voucher" if classify else None,
        "is_certain": classify, "confidence": 0.96 if classify else 0,
        "reason": "Supplier credit invoice" if classify else "Local classification requested",
        "evidence": ["Supplier AB"] if classify else [],
    }
    facts = {
        "amount_text": "-125,00 EUR", "amount_decimal": "-125.00", "amount_effect": "reversal",
        "currency": "EUR", "reference_numbers": ["INV-test", "OCR-test"], "account_numbers": ["**** 4242"],
        "transaction_time_text": None, "transaction_time_iso": None, "transaction_time_role": "unknown",
        "document_kind": "invoice", "document_date_iso": "2026-06-10", "due_date_iso": "2026-07-10",
        "amount_components": [{"role": "fee", "amount_decimal": "-5.00", "label": "Fee", "currency": "EUR"}],
        "payer": {"name": "Buyer AB", "organization_number": "TEST-B", "account_numbers": ["payer-test"]},
        "payee": {"name": "Supplier AB", "organization_number": "TEST-S", "account_numbers": ["payee-test"]},
        "taxes": [{"label": "Moms", "rate_percent": "25", "amount_decimal": "-25.00",
                   "taxable_amount_decimal": "-100.00", "currency": "EUR"}],
    }
    return {"page_start": min(pages), "page_end": max(pages), "source_pages": pages,
            "relevance": "unassessed", "relevance_reason": "deferred_to_audit",
            "document_date_text": "Invoice date 10 June", "document_date_iso": facts["document_date_iso"],
            "due_date_text": "Pay by 10 July", "due_date_iso": facts["due_date_iso"],
            "text": "Supplier credit invoice. Final refund -125 EUR. Invoice June, deadline July.",
            "payment_candidates": [{"text": "Final refund", "amount": "-125.00", "currency": "EUR",
                                    "keyword": "total", "confidence": 0.98}],
            "financial_facts": facts, "classification": classification}


def model_results(page_count, groups, *, classify=True):
    receipts = [receipt_fixture(pages, classify=classify) for pages in groups]
    full = {"status": "accepted", "reasons": [], "primary_receipt_index": len(receipts),
            "text": "Package summary not consumed by the service",
            "receipts": receipts, "financial_facts": receipts[-1]["financial_facts"],
            "classification": receipts[-1]["classification"],
            "page_reviews": [{"page_number": number, "disposition": "financial", "reason": "Financial page"}
                             for number in range(1, page_count + 1)]}
    lean = deepcopy(full)
    for name in ("text", "financial_facts", "classification"):
        lean.pop(name)
    for receipt in lean["receipts"]:
        for name in DERIVED_RECEIPT_FIELDS + (() if classify else ("classification",)):
            receipt.pop(name)
    return full, lean


def fake_response(result):
    return FakeResponse({"id": "fake-lean", "model": "gpt-6.1-sol",
                         "choices": [{"message": {"content": json.dumps(result)}}],
                         "usage": {"prompt_tokens": 10, "completion_tokens": 20}})


@pytest.mark.parametrize("classify", [True, False])
@pytest.mark.parametrize("page_count,groups", [(1, [[1]]), (4, [[1, 3], [4]]), (12, [[n] for n in range(1, 13)])])
def test_compact_wire_restores_identical_public_result(tmp_path, page_count, groups, classify):
    store = VisionSettingsStore(tmp_path / "vision.json")
    store.save(model="gpt-6.1-sol", api_key="sk-test-12345678901234567890")
    full, lean = model_results(page_count, groups, classify=classify)
    original = deepcopy(lean)
    pages = [(b"image", f"page-{n}.jpg", "image/jpeg", n) for n in range(1, page_count + 1)]
    client = OpenAIVisionClient(store)
    with patch("omni_ai_controller.vision.time.perf_counter", return_value=100), \
         patch("omni_ai_controller.vision.urlopen", return_value=fake_response(full)):
        before = client.recognize_document(pages, document_text=None, source_kind="legacy", classify=classify)
    with patch("omni_ai_controller.vision.time.perf_counter", return_value=100), \
         patch("omni_ai_controller.vision.urlopen", return_value=fake_response(lean)) as send:
        after = client.recognize_document(pages, document_text=None, source_kind="pdf_rendered", classify=classify)
    assert after == before
    assert lean == original
    assert after["financial_facts"]["amount_decimal"] == "-125.00"
    assert after["financial_facts"]["taxes"] == full["financial_facts"]["taxes"]
    payload = json.loads(send.call_args.args[0].data)
    content = payload["messages"][0]["content"]
    assert len([item for item in content if item["type"] == "image_url"]) == page_count
    assert all(item["image_url"]["detail"] == "high" for item in content if item["type"] == "image_url")
    assert payload["reasoning_effort"] == "low" and payload["max_completion_tokens"] == 16384
    schema = payload["response_format"]["json_schema"]["schema"]
    assert not {"text", "financial_facts", "classification"} & schema["properties"].keys()
    assert not set(DERIVED_RECEIPT_FIELDS) & schema["properties"]["receipts"]["items"]["properties"].keys()
    assert ("classification" in schema["properties"]["receipts"]["items"]["properties"]) == classify
    assert len(json.dumps(lean)) < len(json.dumps(full))


@pytest.mark.parametrize("classify", [True, False])
def test_compact_schema_keeps_every_financial_field_and_does_not_mutate_legacy(classify):
    original = deepcopy(RECEIPT_RESULT_SCHEMA)
    schema = compact_receipt_schema(RECEIPT_RESULT_SCHEMA, classify=classify)
    facts = schema["properties"]["receipts"]["items"]["properties"]["financial_facts"]
    expanded = deepcopy(facts)
    for name in ("payer", "payee"):
        assert expanded["properties"][name] == {"$ref": "#/$defs/party"}
        expanded["properties"][name] = schema["$defs"]["party"]
    assert expanded == FINANCIAL_FACTS_SCHEMA
    assert original == RECEIPT_RESULT_SCHEMA
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])


def test_compact_advertisement_stays_empty_without_fabricating_financial_facts(tmp_path):
    store = VisionSettingsStore(tmp_path / "vision.json")
    store.save(model="gpt-6.1-sol", api_key="sk-test-12345678901234567890")
    lean = {"status": "needs_manual_confirmation", "reasons": ["No financial document"],
            "primary_receipt_index": None, "receipts": [],
            "page_reviews": [{"page_number": 1, "disposition": "advertisement", "reason": "Ad"}]}
    with patch("omni_ai_controller.vision.urlopen", return_value=fake_response(lean)):
        result = OpenAIVisionClient(store).recognize(b"image", filename="ad.jpg", content_type="image/jpeg")
    assert result["receipts"] == [] and result["primary_receipt_index"] is None
    assert result["financial_facts"] == result["classification"] == {}


@pytest.mark.parametrize("classify", [True, False])
def test_channel_count_schema_is_strict_required_nullable_in_both_wire_shapes(classify):
    compact = compact_receipt_schema(RECEIPT_RESULT_SCHEMA, classify=classify)

    def strict(node):
        if node.get("type") == "object":
            assert node["additionalProperties"] is False
            assert set(node["required"]) == set(node["properties"])
        for child in node.get("properties", {}).values():
            strict(child)
        if isinstance(node.get("items"), dict):
            strict(node["items"])
        for child in node.get("$defs", {}).values():
            strict(child)

    strict(RECEIPT_RESULT_SCHEMA)
    strict(compact)
    for facts in (FINANCIAL_FACTS_SCHEMA, compact["properties"]["receipts"]["items"]["properties"]["financial_facts"]):
        component = facts["properties"]["amount_components"]["items"]
        assert component["properties"]["transaction_count"]["type"] == ["integer", "null"]
        assert "transaction_count" in component["required"]
        assert {"card", "cash", "swish", "fee", "other", "bank_transfer", "mobile_payment", "wallet"} <= set(component["properties"]["role"]["enum"])


@pytest.mark.parametrize("source_kind", ["legacy", "image", "pdf_rendered", "text"])
def test_channel_counts_explicit_evidence_and_historical_absence_survive_all_wire_modes(tmp_path, source_kind):
    store = VisionSettingsStore(tmp_path / "vision.json")
    store.save(model="gpt-6.1-sol", api_key="sk-test-12345678901234567890")
    full, lean = model_results(1, [[1]])
    parts = [
        {"role": "swish", "label": "Swish(2)", "amount_decimal": "25", "currency": "EUR", "transaction_count": 2},
        {"role": "cash", "label": "Cash(0)", "amount_decimal": "0", "currency": "EUR", "transaction_count": 0},
        {"role": "card", "label": "Card", "amount_decimal": "70", "currency": "EUR"},
        {"role": "bank_transfer", "label": "Bank transfer", "amount_decimal": "5", "currency": "EUR", "transaction_count": None},
        {"role": "mobile_payment", "label": "Mobile(3)", "amount_decimal": "5", "currency": "EUR", "transaction_count": 3},
        {"role": "wallet", "label": "Wallet(1)", "amount_decimal": "20", "currency": "EUR", "transaction_count": 1},
    ] + [{"role": "other", "label": "Visible other channel " + "x" * 600, "amount_decimal": "1", "currency": "EUR", "transaction_count": None} for _ in range(40)]
    result_data = full if source_kind == "legacy" else lean
    result_data["receipts"][0]["financial_facts"]["amount_components"] = parts
    pages = [] if source_kind == "text" else [(b"image", "page.jpg", "image/jpeg", 1)]
    with patch("omni_ai_controller.vision.urlopen", return_value=fake_response(result_data)) as send:
        result = OpenAIVisionClient(store).recognize_document(pages, document_text="Visible source" if source_kind == "text" else None, source_kind=source_kind)
    assert result["financial_facts"]["amount_components"] == parts
    assert result["receipts"][0]["financial_facts"]["amount_components"] == parts
    assert "transaction_count" not in result["financial_facts"]["amount_components"][2]
    assert result["financial_facts"]["amount_decimal"] == "-125.00"  # No duplicated/new gross.
    payload = json.loads(send.call_args.args[0].data)
    prompt = payload["messages"][0]["content"][0]["text"]
    assert "Swish(2)" in prompt and "transaction_count=null" in prompt
    assert "never count the same gross amount twice" in prompt
    assert "Do not truncate amount_components" in prompt
    assert "bank account, phone number, document brand, company name" in prompt


@pytest.mark.parametrize("count,label", [(2, "Swish"), (3, "Swish(2)"), (True, "Swish(1)"),
    ("2", "Swish(2)"), (2.0, "Swish(2)"), (-2, "Swish(-2)")])
def test_unsupported_count_is_null_not_inferred_or_financial_field_drop(tmp_path, count, label):
    store = VisionSettingsStore(tmp_path / "vision.json")
    store.save(model="gpt-6.1-sol", api_key="sk-test-12345678901234567890")
    _, lean = model_results(1, [[1]])
    part = {"role": "swish", "label": label, "amount_decimal": "25", "currency": "EUR", "transaction_count": count}
    lean["receipts"][0]["financial_facts"]["amount_components"] = [part]
    with patch("omni_ai_controller.vision.urlopen", return_value=fake_response(lean)):
        result = OpenAIVisionClient(store).recognize(b"image", filename="r.jpg", content_type="image/jpeg")
    assert result["financial_facts"]["amount_components"] == [{**part, "transaction_count": None}]