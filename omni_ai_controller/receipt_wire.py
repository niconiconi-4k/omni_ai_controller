"""Compact model output only; the public receipt contract remains unchanged."""
from copy import deepcopy
from typing import Any

DERIVED_RECEIPT_FIELDS = (
    "page_start", "page_end", "relevance", "relevance_reason",
    "document_date_iso", "due_date_iso",
)


def compact_receipt_schema(receipt_schema: dict[str, Any], *, classify: bool) -> dict[str, Any]:
    receipt = deepcopy(receipt_schema)
    for name in DERIVED_RECEIPT_FIELDS + (() if classify else ("classification",)):
        receipt["properties"].pop(name)
        receipt["required"].remove(name)
    facts = receipt["properties"]["financial_facts"]
    party = facts["properties"]["payer"]
    facts["properties"]["payer"] = {"$ref": "#/$defs/party"}
    facts["properties"]["payee"] = {"$ref": "#/$defs/party"}
    properties = {
        "status": {"type": "string", "enum": ["accepted", "needs_manual_confirmation", "needs_reupload"]},
        "reasons": {"type": "array", "items": {"type": "string"}},
        "primary_receipt_index": {"type": ["integer", "null"], "minimum": 1},
        "receipts": {"type": "array", "minItems": 0, "maxItems": 36, "items": receipt},
    }
    # page_reviews is added by the caller using the existing page schema.
    return {"type": "object", "additionalProperties": False, "properties": properties,
            "required": list(properties), "$defs": {"party": party}}


def expand_receipt_result(result: dict[str, Any], *, page_count: int, classify: bool) -> dict[str, Any]:
    """Reconstruct structural duplicates, never infer financial values or classifications."""
    expanded = deepcopy(result)
    raw_receipts = expanded.get("receipts")
    if not isinstance(raw_receipts, list):
        return expanded  # Legacy response normalization still handles old formats.
    for receipt in raw_receipts:
        if not isinstance(receipt, dict):
            continue
        pages = receipt.get("source_pages")
        if isinstance(pages, list):
            valid_pages = [page for page in pages if isinstance(page, int) and not isinstance(page, bool)
                           and 1 <= page <= page_count]
            if valid_pages:
                receipt.setdefault("page_start", min(valid_pages))
                receipt.setdefault("page_end", max(valid_pages))
        facts = receipt.get("financial_facts")
        facts = facts if isinstance(facts, dict) else {}
        receipt.setdefault("relevance", "unassessed")
        receipt.setdefault("relevance_reason", "deferred_to_audit")
        receipt.setdefault("document_date_iso", facts.get("document_date_iso"))
        receipt.setdefault("due_date_iso", facts.get("due_date_iso"))
        if not classify:
            receipt.setdefault("classification", {
                "document_type": None, "is_certain": False, "confidence": 0,
                "reason": "Local classification requested", "evidence": [],
            })
    return expanded