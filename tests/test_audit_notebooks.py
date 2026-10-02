import json

import pytest

from omni_ai_controller.audit_notebooks import (
    AuditNotebooks, MAX_NOTEBOOK_BYTES, MAX_TASK_ENTRIES, serialized,
)


def _task(**updates):
    return {"task_id": "find-1200", "operation": "search_amount", "objective": "查1200及重复候选",
            "transaction_ids": ["t"], "receipt_ids": ["r1", "r2", "r3"], "amount": 1200,
            "depends_on": [], **updates}


def _chunk():
    return {"transactions": [{"id": "t", "amount": -1200}], "receipts": [
        {"id": "r1", "amount": 1200, "currency": "SEK", "issue_date": "2026-10-01",
         "date": "2026-10-01", "date_role": "issue_date", "payment_date": "2026-10-05",
         "party": "A", "accounts": ["123"], "reference": "order-A", "taxes": [25]},
        {"id": "r2", "amount": 1200, "date": "2026-10-04", "date_role": "payment"},
        {"id": "r3", "amount": 999, "date": "2026-10-03"},
    ]}


@pytest.mark.parametrize("owner", ["audit_planner", "evidence_worker"])
def test_complete_utf8_notebook_has_hard_8mb_cap_with_eviction_and_overflow(owner):
    notebooks = AuditNotebooks("a", "run")
    source = {"raw": "中文€" * 650000, "confirmed_decisions": [{"transaction_id": "t"}]}
    original_length = len(source["raw"])
    assert notebooks.put(owner, "first", "derivative", source["raw"])
    assert notebooks.put(owner, "second", "derivative", source["raw"])
    notebook = notebooks.snapshot()["notebooks"][owner]
    assert notebook["evicted_entries"] == 1
    assert [entry["key"] for entry in notebook["entries"]] == ["second"]
    assert not notebooks.put(owner, "oversized", "derivative", "中文" * 1500000)
    notebook = notebooks.snapshot()["notebooks"][owner]
    document = serialized(notebook)
    assert notebook["overflow"] == 1
    assert notebook["used_bytes"] == len(document.encode("utf-8")) <= MAX_NOTEBOOK_BYTES
    assert json.loads(document) == notebook
    assert len(source["raw"]) == original_length
    assert source["confirmed_decisions"] == [{"transaction_id": "t"}]


def test_basic_cache_then_on_demand_details_and_actual_date_order():
    notebooks = AuditNotebooks("a", "run")
    chunk = _chunk()
    basic = notebooks.facts(chunk["receipts"][0])
    assert "party" not in basic and "accounts" not in basic and "taxes" not in basic
    assert basic["date_role"] == "issue_date" and basic["event_date"] == "2026-10-05"
    prepared = notebooks.prepare(chunk, [_task()])
    assert prepared["amount_searches"][0]["receipt_ids"] == ["r2", "r1"]
    assert prepared["amount_searches"][0]["duplicate_candidates"] == ["r2", "r1"]
    details = next(item for item in prepared["receipt_facts"] if item["id"] == "r1")
    assert details["party"] == "A" and details["accounts"] == ["123"]
    assert details["reference"] == "order-A" and details["taxes"] == [25]
    state = notebooks.snapshot()
    assert state["stats"]["exact_amount_hits"] == 2
    assert state["stats"]["duplicate_candidates"] == 1
    assert state["stats"]["cache_hits"] >= 1
    assert any(entry["kind"] == "search_results" for entry in state["notebooks"]["evidence_worker"]["entries"])


def test_cache_uses_unchanged_source_only_and_invalidates_all_projections():
    notebooks = AuditNotebooks("a", "run")
    receipt = _chunk()["receipts"][0]
    notebooks.facts(receipt)
    notebooks.facts(receipt, details=True)
    assert notebooks.facts(receipt)["amount"] == 1200
    assert notebooks.snapshot()["stats"]["cache_hits"] == 1
    changed = {**receipt, "amount": 900, "party": "B"}
    assert notebooks.facts(changed)["amount"] == 900
    assert not any(entry["key"] == "receipt:r1:details" for entry in notebooks.snapshot()["notebooks"]["evidence_worker"]["entries"])
    assert notebooks.facts(changed, details=True)["party"] == "B"
    assert notebooks.snapshot()["stats"]["invalidations"] == 1


def test_notebooks_bind_both_case_and_run_and_resume_real_cache():
    notebooks = AuditNotebooks("a", "run")
    notebooks.facts({"id": "r", "amount": 1200})
    state = notebooks.snapshot()
    resumed = AuditNotebooks("a", "run", state)
    resumed.facts({"id": "r", "amount": 1200})
    assert resumed.snapshot()["stats"]["cache_hits"] == 1
    for audit, run in (("other", "run"), ("a", "other")):
        reset = AuditNotebooks(audit, run, state).snapshot()
        assert all(not book["entries"] for book in reset["notebooks"].values())


def test_task_entries_are_bounded_deduplicated_and_transition_events_are_real():
    events = []
    notebooks = AuditNotebooks("a", "run", publish=events.append)
    notebooks.task("evidence_worker", _task(), "pending")
    notebooks.task("evidence_worker", _task(), "running")
    notebooks.task("evidence_worker", _task(), "completed")
    assert [event["task_lists"]["evidence_worker"][0]["status"] for event in events] == ["pending", "running", "completed"]
    assert len(notebooks.snapshot()["task_lists"]["evidence_worker"]) == 1
    notebooks.publish = None
    for index in range(MAX_TASK_ENTRIES + 3):
        notebooks.task("evidence_worker", _task(task_id=str(index)), "completed")
    state = notebooks.snapshot()
    assert len(state["task_lists"]["evidence_worker"]) == MAX_TASK_ENTRIES
    assert state["stats"]["task_evictions"] == 4
    active = AuditNotebooks("a", "active")
    for index in range(MAX_TASK_ENTRIES):
        active.task("audit_planner", _task(task_id=str(index)), "running")
    with pytest.raises(ValueError, match="有界预算"):
        active.task("audit_planner", _task(task_id="overflow"), "pending")
    assert active.snapshot()["stats"]["task_overflow"] == 1


def test_categories_and_actual_event_date_are_indexed_not_invoice_due_dates():
    notebooks = AuditNotebooks("a", "run")
    chunk = {"transactions": [], "receipts": [
        {"id": "i", "type": "income", "amount": 1200, "sale_date": "2026-09-29"},
        {"id": "e", "type": "expense", "amount": 1200, "date": "2026-10-01", "date_role": "issue_date", "due_date": "2026-11-01"},
        {"id": "p", "type": "payroll", "amount": 1200, "payment_date": "2026-10-02"},
        {"id": "r", "type": "refund", "amount": -1200, "payment_date": "2026-10-03"},
    ]}
    prepared = notebooks.prepare(chunk, [_task(receipt_ids=["i", "e", "p", "r"])])
    assert [item["category"] for item in prepared["receipt_facts"]] == ["expense", "income", "payroll", "refund"]
    assert prepared["receipt_facts"][0]["event_date"] == ""
    assert prepared["receipt_facts"][1]["event_date"] == "2026-09-29"


def test_existing_token_lean_financial_facts_preserve_roles_components_and_lazy_details():
    notebooks = AuditNotebooks("a", "run")
    receipt = {"id": "r1", "document_type": "income_voucher", "financial_facts": {
        "amount_decimal": "1200.00", "currency": "SEK", "amount_effect": "normal",
        "transaction_time_iso": "2026-09-29T17:15:00", "transaction_time_role": "sales_activity",
        "document_kind": "sales_report", "document_date_iso": "2026-10-01", "due_date_iso": None,
        "amount_components": [{"role": "card", "amount_decimal": "900.00"}, {"role": "cash", "amount_decimal": "300.00"}],
        "payer": {"name": "Buyer"}, "payee": {"name": "Store"},
        "reference_numbers": ["batch-9"], "account_numbers": ["account-9"], "taxes": [{"rate_percent": "25"}],
    }}
    basic = notebooks.facts(receipt)
    assert "payer" not in basic["financial_facts"] and "account_numbers" not in basic["financial_facts"]
    assert basic["event_date"] == "2026-09-29T17:15:00" and basic["category"] == "income"
    assert basic["financial_facts"]["amount_components"] == receipt["financial_facts"]["amount_components"]
    prepared = notebooks.prepare({"transactions": [{"id": "t", "amount_decimal": "1200"}], "receipts": [receipt]},
                                 [_task(operation="compare_details", receipt_ids=["r1"])])
    assert prepared["amount_searches"][0]["receipt_ids"] == ["r1"]
    detail = prepared["receipt_facts"][0]["financial_facts"]
    for key in ("payer", "payee", "reference_numbers", "account_numbers", "taxes"):
        assert detail[key] == receipt["financial_facts"][key]