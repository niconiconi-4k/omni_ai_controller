from copy import deepcopy
from decimal import localcontext

import pytest

from omni_ai_controller.audit_inventory import (
    MAX_NOTEBOOK_BYTES, base_facts, canonical_bytes, cashflow_facts,
    cashflow_sort_key, category, partition_cashflow,
)
from omni_ai_controller.audit_notebooks import AuditNotebooks


@pytest.mark.parametrize("amount,direction,lane,signed", [
    ("125.00", "debit", "out", "-125.00"),
    ("125.00", "credit", "in", "125.00"),
    ("-125.00", None, "out", "-125.00"),
    ("125.00", None, "in", "125.00"),
    ("-125.00", "debit", "out", "-125.00"),
    ("-125.00", "credit", "unknown", None),
    ("125.00", "neutral", "unknown", None),
    ("125.00", "nonsense", "unknown", None),
    ("NaN", "debit", "unknown", None),
    ("Infinity", "credit", "unknown", None),
    ("bad", "credit", "unknown", None),
    (True, "debit", "unknown", None),
    ("0", "credit", "unknown", None),
])
def test_bank_direction_precedes_positive_magnitude_and_contradictions_fail_safe(amount, direction, lane, signed):
    source = {"id": "t", "amount": amount, "direction": direction}
    before = deepcopy(source)
    expected = {"cashflow_lane": lane, "signed_cashflow_amount": signed}
    assert cashflow_facts(source, source_kind="transaction") == expected
    assert {key: base_facts(source)[key] for key in expected} == expected
    assert category(source) == {"in": "income", "out": "expense"}.get(lane, "unknown")
    assert source == before


@pytest.mark.parametrize("typ,normal,refund", [
    ("income_voucher", "in", "out"), ("expense_voucher", "out", "in"),
    ("payroll_voucher", "out", "in"), ("tax_voucher", "out", "in"),
    ("loan_interest_voucher", "out", "in"),
])
@pytest.mark.parametrize("reversed_source", [False, True])
def test_receipt_sign_is_type_plus_reversal_not_positive_income(typ, normal, refund, reversed_source):
    source = {"id": "r", "type": typ, "amount": "100.50", "signed_amount": "-100.50" if reversed_source else "100.50",
              "refund": reversed_source, "financial_facts": {"amount_effect": "reversal" if reversed_source else "normal"}}
    facts = base_facts(source)
    lane = refund if reversed_source else normal
    assert facts["cashflow_lane"] == lane
    assert facts["signed_cashflow_amount"] == ("100.50" if lane == "in" else "-100.50")
    assert facts["signed_amount"] == source["signed_amount"]
    assert facts["category"] == ("refund" if reversed_source else "payroll" if typ == "payroll_voucher" else "income" if typ == "income_voucher" else "expense")


def test_unknown_invoice_refund_and_stale_lane_never_manufacture_income():
    for source in ({"type": "invoice", "amount": "100"},
                   {"type": "refund", "amount": "-100"},
                   {"type": "income_voucher", "category": "unknown", "amount": "100"}):
        assert cashflow_facts(source)["cashflow_lane"] == "unknown"
    source = {"amount": "100", "direction": "debit", "cashflow_lane": "in", "signed_cashflow_amount": "100"}
    assert base_facts(source)["signed_cashflow_amount"] == "-100"
    assert cashflow_facts({**source, "signed_amount": "100"})["cashflow_lane"] == "unknown"
    assert cashflow_facts({"type": "income_voucher", "category": "refund", "amount": "100"})["signed_cashflow_amount"] == "-100"
    for role in ("unknown", "invoice_issue", "invoice_due"):
        facts = base_facts({"id": "date", "type": "expense_voucher", "amount": "100", "date_role": role, "date": "2026-10-01"})
        assert facts["event_date"] == ""


def test_lane_currency_actual_time_before_category_and_no_uuid_tiebreak():
    sources = [
        {"id": "z", "type": "expense_voucher", "amount": "100", "refund": True, "currency": "SEK", "event_date": "2026-10-01"},
        {"id": "a", "type": "income_voucher", "amount": "100", "currency": "SEK", "event_date": "2026-10-03"},
        {"id": "out", "type": "payroll_voucher", "amount": "100", "currency": "EUR", "event_date": "2026-09-01"},
        {"id": "unknown", "type": "invoice", "amount": "100", "currency": "EUR", "event_date": "2026-09-01"},
    ]
    before = deepcopy(sources)
    lanes = partition_cashflow(sources, source_kind="receipt")
    assert [r["id"] for r in lanes["in"]] == ["z", "a"]
    assert [r["id"] for r in lanes["out"]] == ["out"]
    assert [r["id"] for r in lanes["unknown"]] == ["unknown"]
    assert sources == before
    assert cashflow_sort_key(sources[0]) == cashflow_sort_key({**sources[0], "id": "arbitrary-other-uuid"})
    banks = [{"id": "negative", "amount": "-100", "event_date": "2026-10-01"},
             {"id": "positive", "amount": "100", "event_date": "2026-10-01"}]
    assert [r["id"] for r in partition_cashflow(banks, source_kind="transaction")["out"]] == ["negative"]


def test_cache_index_keeps_count_fields_complete_dates_unknown_and_eight_mib():
    components = [{"role": "swish", "label": "Swish(2)", "amount_decimal": "25", "transaction_count": 2},
                  {"role": "card", "label": "Card", "amount_decimal": "75"}]
    inventory = {"transactions": [{"id": "t", "amount": "100", "direction": "debit", "event_date": "2026-10-02"}],
                 "receipts": [{"id": "r", "type": "expense_voucher", "amount": "100", "date_role": "invoice_due",
                               "event_date": "2026-10-01", "financial_facts": {"amount_components": components}}]}
    original = deepcopy(inventory)
    books = AuditNotebooks("a", "r")
    books.organize_inventory(inventory)
    for book in books.snapshot()["notebooks"].values():
        assert book["limit_bytes"] == MAX_NOTEBOOK_BYTES == 8 * 1024 * 1024
        assert book["used_bytes"] == len(canonical_bytes(book)) <= MAX_NOTEBOOK_BYTES
        basics = [e["content"] for e in book["entries"] if e["kind"] == "source_basic"]
        assert all(b["cashflow_lane"] == "out" and b["signed_cashflow_amount"] == "-100" for b in basics)
        receipt = next(b for b in basics if books.codec.decode(b["id"], text=False) == "r")
        assert receipt["event_date"] == "" and receipt["financial_facts"]["amount_components"] == components
        index = next(e["content"] for e in book["entries"] if e["kind"] == "source_index")
        assert [books.codec.decode(i["id"], text=False) for i in index] == ["t", "r"]
        assert all(i["cashflow_lane"] == "out" and i["signed_cashflow_amount"] == "-100" for i in index)
    assert inventory == original


def test_high_precision_economic_sign_does_not_round_under_low_decimal_context():
    with localcontext() as context:
        context.prec = 3
        value = "12345678901234567890.123456789"
        assert cashflow_facts({"amount": value, "direction": "debit"})["signed_cashflow_amount"] == "-" + value