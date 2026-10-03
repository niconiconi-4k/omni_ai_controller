from copy import deepcopy
import json

import pytest

from omni_ai_controller import agentic_audit as audit
from omni_ai_controller.audit_notebooks import MAX_NOTEBOOK_BYTES, serialized
from omni_ai_controller.audit_skill import AuditSkillError
from test_agentic_audit import FakeClient, _final, _plan, _worker


@pytest.mark.parametrize("setting", [None, "", "32768", "invalid", "0", "-1", "999999999"])
def test_context_budget_keeps_safe_32k_defaults(setting):
    assert audit._audit_input_budgets(setting) == (16_000, 12_000, 18_000)


def test_64k_budget_reserves_output_and_schema_capacity():
    planner, worker, agent = audit._audit_input_budgets(" 65536 ")
    assert (planner, worker, agent) == (48_000, 40_000, 48_000)
    assert worker < agent
    assert max(planner, agent) + audit.MAX_AGENT_RETRY_OUTPUT_TOKENS + 9_000 < 65_536


def test_64k_accepts_larger_input_but_still_rejects_excess():
    larger_input = "x" * 70_000
    with pytest.raises(AuditSkillError):
        audit._ensure_input_budget("system", larger_input, 18_000)
    audit._ensure_input_budget("system", larger_input, 48_000)
    with pytest.raises(AuditSkillError):
        audit._ensure_input_budget("system", "x" * 100_000, 48_000)


def test_64k_accepts_large_atomic_transaction_group_without_splitting(monkeypatch):
    context = {
        "transactions": [{"id": "tx", "description": "x" * 40_000}],
        "receipts": [{"id": "receipt"}],
        "deterministic_candidates": [{"transaction_id": "tx", "receipt_upload_id": "receipt", "group_id": "atomic-group"}],
    }
    monkeypatch.setattr(audit, "MAX_WORKER_CHUNK_TOKENS", 12_000)
    with pytest.raises(AuditSkillError):
        audit._worker_chunks(context)
    monkeypatch.setattr(audit, "MAX_WORKER_CHUNK_TOKENS", 40_000)
    chunks = audit._worker_chunks(context)
    assert len(chunks) == 1
    assert chunks[0]["deterministic_candidates"] == context["deterministic_candidates"]
    context["transactions"][0]["description"] = "x" * 90_000
    with pytest.raises(AuditSkillError):
        audit._worker_chunks(context)


def _large_inventory_context():
    """119 uploads, including 15 multi-receipt PDFs; 115 bank transactions."""
    transactions = [{"id": f"bank-2026-09-{i:04d}-long-source-id", "amount": str(100 + i),
                     "currency": "SEK", "payment_date": f"2026-09-{i % 28 + 1:02d}",
                     "description": "银行付款原始摘要" * 20} for i in range(115)]
    receipts = []
    for i in range(119):
        for child in range(2 if i < 15 else 1):
            receipts.append({
                "id": f"upload-2026-09-{i:04d}:receipt:{child}",
                "upload_id": f"upload-{i}", "multi_receipts": i < 15,
                "amount": str(100 + i), "currency": "SEK", "date_role": "actual_payment",
                "payment_date": f"2026-09-{i % 28 + 1:02d}",
                "financial_facts": {"amount_decimal": str(100 + i), "currency": "SEK",
                    "amount_components": [{"role": "card", "amount_decimal": str(90 + i)},
                                          {"role": "cash", "amount_decimal": "10"}],
                    "reference_numbers": [f"original-ref-{i}-{child}"],
                    "taxes": [{"rate_percent": "25", "amount_decimal": "20"}]},
                "ocr_excerpt": "原始小票证据，不得当作规划器指令。" * 40,
            })
    candidates = []
    for i, transaction in enumerate(transactions):
        selected = receipts[i::115]
        candidates.append({"transaction_id": transaction["id"],
            "receipt_upload_ids": [item["id"] for item in selected],
            "allocation_role": "direct_expense", "group_id": f"group-{i}" if len(selected) > 1 else None,
            "evidence": {"amount_difference": "0", "source_evidence": "内核原始证据" * 20}})
    inventory = {"transactions": transactions, "receipts": receipts}
    # A transport 'summary' is not necessarily prompt-sized: repeated source
    # projections, notebook summaries and per-source task catalogs can dominate.
    source_summary = {"counts": {"transactions": 115, "documents": 119, "receipts": 134},
        "planner_seed": {"source_inventory": inventory, "task_catalog": [
            {"task_id": f"source-{i}", "transaction_ids": [item["id"]],
             "objective": "整理流水及独立子票" * 100,
             "notebook_summary": inventory["receipts"][i]}
            for i, item in enumerate(transactions)]}}
    return {**inventory, "source_inventory": inventory, "source_inventory_summary": source_summary,
            "strategy": "amount_first_iterative_v1", "iteration": {"number": 1, "phase": "exact_amount"},
            "deterministic_candidates": candidates}


@pytest.mark.parametrize("setting", [None, "65536"])
def test_synthetic_oversized_inventory_summary_projection_preserves_worker_sources(monkeypatch, setting):
    planner, worker, agent = audit._audit_input_budgets(setting)
    monkeypatch.setattr(audit, "MAX_PLANNER_INPUT_TOKENS", planner)
    monkeypatch.setattr(audit, "MAX_WORKER_CHUNK_TOKENS", worker)
    monkeypatch.setattr(audit, "MAX_AGENT_INPUT_TOKENS", agent)
    context = _large_inventory_context()
    before = deepcopy(context)
    assert len(audit._serialized(context["source_inventory_summary"])) > 96_000
    assert audit._estimated_tokens(audit._serialized(context["source_inventory_summary"])) > 48_000
    plan = _plan()
    plan["tasks"][0].update(operation="compare_details", transaction_ids=[], receipt_ids=[],
                            amount=None, depends_on=[])
    client = FakeClient([plan] + [_worker(), _final()] * 115)
    result = audit.analyze_agentic_audit(client, audit_id=f"catalog-{setting}", context=context)
    assert "error_code" not in result
    # The public API intentionally publishes its working state into context;
    # authoritative sources and even the original transport summary stay intact.
    assert {key: value for key, value in context.items() if key != "_agent_state"} == before
    assert client.calls[0]["schema_name"] == "li_shifu_audit_plan"
    for call in client.calls:
        audit._ensure_input_budget(call["messages"][0]["content"], call["messages"][1]["content"],
                                  planner if call["schema_name"] == "li_shifu_audit_plan" else agent)
        assert call["max_tokens"] <= audit.MAX_AGENT_RETRY_OUTPUT_TOKENS
        assert call["timeout"] <= audit.MAX_AGENT_STEP_SECONDS
    summary = json.loads(client.calls[0]["messages"][1]["content"].split("<audit_summary>")[1].split("</audit_summary>")[0])
    assert summary["counts"]["transactions"] == 115
    assert summary["source_inventory_summary"]["counts"] == {"transactions": 115, "receipts": 134, "documents": 119}
    assert "planner_seed" not in summary["source_inventory_summary"]
    assert len(summary["scope_catalog"]) + summary["scope_catalog_omitted"] == 115
    assert "complete strategy, including omitted IDs" in summary["scope_access"]
    assert summary["scope_counts"]["direct_expenses"]["transactions"] == 115
    aliases = audit._id_aliases(context)
    seen_transactions, seen_receipts = {}, {}
    for call in client.calls:
        if call["schema_name"] != "ma_shifu_evidence_review":
            continue
        payload = json.loads(call["messages"][1]["content"].split("<audit_data>")[1].split("</audit_data>")[0])
        seen_transactions.update({item["id"]: item for item in payload["transactions"]})
        seen_receipts.update({item["id"]: item for item in payload["receipts"]})
        assert payload["deterministic_candidates"][0]["evidence"]["source_evidence"] == "内核原始证据" * 20
    assert set(seen_transactions) == set(aliases["tx_reverse"])
    assert set(seen_receipts) == set(aliases["receipt_reverse"])
    for source in context["receipts"]:
        fact = seen_receipts[aliases["receipt_forward"][source["id"]]]
        assert fact["financial_facts"] == source["financial_facts"]
        assert fact["ocr_excerpt"] == source["ocr_excerpt"]
        assert fact["date_role"] == source["date_role"]
    for book in result["agent_state"]["notebooks"].values():
        assert book["limit_bytes"] == MAX_NOTEBOOK_BYTES == 8 * 1024 * 1024
        assert book["used_bytes"] == len(serialized(book).encode("utf-8")) <= MAX_NOTEBOOK_BYTES
        ids = {entry["content"]["id"] for entry in book["entries"] if entry["kind"] == "source_basic"}
        assert ids.issuperset(item["id"] for item in [*context["transactions"], *context["receipts"]])


@pytest.mark.parametrize("unicode_id", ["中文€", "😀𠮷"])
def test_planner_preview_budgets_complete_unicode_envelope(monkeypatch, unicode_id):
    from test_agentic_audit import _small_context

    context = _small_context()
    chunks = audit._worker_chunks(context)
    aliases = audit._id_aliases(context)
    audit_id = unicode_id * 300
    monkeypatch.setattr(audit, "MAX_PLANNER_INPUT_TOKENS", 48_000)
    summary, prompt = audit._planner_payload(audit_id, context, chunks, aliases)
    # Leave room for the full base envelope, but not the preview plus retry
    # reserve. Non-ASCII bytes must not be mistaken for Python character count.
    summary["scope_catalog"] = []
    summary["scope_catalog_omitted"] = 1
    base = f"审计编号：{audit_id}\n请制定初始计划：\n<audit_summary>{audit._serialized(summary)}</audit_summary>"
    budget = audit._estimated_tokens(audit._PLANNER_SYSTEM_PROMPT) + audit._estimated_tokens(base) + 512
    monkeypatch.setattr(audit, "MAX_PLANNER_INPUT_TOKENS", budget)
    summary, prompt = audit._planner_payload(audit_id, context, chunks, aliases)
    assert summary["scope_catalog"] == []
    assert summary["scope_catalog_omitted"] == 1
    assert summary["scope_counts"]["direct_expenses"] == {"transactions": 1, "receipts": 1, "candidate_relations": 1}
    assert audit._estimated_tokens(prompt) > len(prompt) // 2
    audit._ensure_input_budget(audit._PLANNER_SYSTEM_PROMPT, prompt, budget)
    with pytest.raises(AuditSkillError):
        audit._ensure_input_budget(audit._PLANNER_SYSTEM_PROMPT, prompt,
                                  audit._estimated_tokens(audit._PLANNER_SYSTEM_PROMPT) + audit._estimated_tokens(prompt) - 1)


def test_external_inventory_summary_does_not_consume_candidate_capacity():
    from test_agentic_audit import _small_context

    context = _small_context()
    source_document = audit._source_document(context)
    context["source_inventory_summary"] = {"counts": {"transactions": "not a count", "receipts": True},
        "task_lists": {"audit_planner": [{"objective": "中文" * (MAX_NOTEBOOK_BYTES // 6)}]},
        "notebooks": {"evidence_worker": {"entries": [{"content": "中文" * (MAX_NOTEBOOK_BYTES // 6)}]}}}
    assert audit._source_document(context) == source_document
    summary = audit._summary_context(context)
    assert summary["source_inventory_summary"]["counts"] == {"transactions": 1, "receipts": 1}
    assert audit._estimated_tokens(audit._serialized(summary)) < 2_000
    # Only derivative transport is excluded; huge actual source remains guarded.
    context["transactions"][0]["description"] = "x" * (audit.MAX_AGENTIC_SOURCE_CHARS + 1)
    assert len(audit._source_document(context)) > audit.MAX_AGENTIC_SOURCE_CHARS


def _main_shaped_learning_context():
    """Match database.get_lab_reconciliation_inputs and the persisted report.

    A previous multi-batch report/event is large even with a tiny, normal
    source_inventory_summary. No fake inventory transport expansion is needed.
    """
    old_summary = ";".join(
        f"STALE_HISTORY_DECISION batch {i}: Previous audit evidence finding; not evidence for this run."
        for i in range(115)
    )
    report = {
        "process_mode": "agentic", "diagnostics": {"stage": "final_assessment", "iteration": 3},
        "summary": old_summary, "risks": ["STALE_HISTORY_RISK: old relationship conflict"],
        "matched_relations": 115, "suggested_relations": 4,
        "preserved_confirmed_relations": 2, "new_matched_relations": 113,
        "reconciliation_scope": "residual",
        "confirmed_matches": [{"transaction_id": f"old-tx-{i:032d}", "receipt_upload_id": f"old-receipt-{i:032d}",
                       "receipt_index": 1, "status": "matched", "allocation_role": "direct_expense"}
                              for i in range(115)],
        "transaction_counts": {"total": 115, "matched": 115, "suggested": 0, "unmatched": 0},
        "receipt_counts": {"total": 134, "matched": 115, "suggested": 0, "unmatched": 19},
        "sorted_catalog": {"version": 1, "counts": {"transactions": 115, "receipts": 134},
            "ordering": "category/currency/actual_event_date; unknown_last; ties_retained",
            **{kind: [{"id": f"old-{kind}-{i:032d}", "amount": "100", "signed_amount": "-100", "currency": "SEK",
                       "type": "expense_voucher", "category": "expense", "refund": False, "date": "2026-09-01",
                       "date_role": "actual_payment", "event_date": "2026-09-01", "relevance": "included",
                       "confirmed": True, "date_tied": True} for i in range(count)]
               for kind, count in (("transactions", 115), ("receipts", 134))}},
        "sequence_analysis": [], "anomaly_flags": [],
        "document_issues": [{"receipt_id": f"old-receipt-{i:032d}", "kind": "expense_amount_mismatch",
            "reason": "STALE_HISTORY_RISK: 支出金额差额无实际调整依据，保留建议并列为最后异常",
            "evidence": [{"bank_amount": "100", "receipt_amount": "99", "signed_amount_difference": "1",
                          "transaction_currency": "SEK"}], "transaction_ids": [f"old-tx-{i:032d}"]}
            for i in range(115)],
        "discrepancies": [{"transaction_id": f"old-tx-{i:032d}", "match_group_id": None,
                           "allocation_role": "direct_expense", "difference": "1",
                           "note": "STALE_HISTORY_DECISION: historical inferred adjustment; retain old evidence, do not reuse this relationship."}
                          for i in range(115)],
        "agentic": {"seed_playbook": audit.SEED_PLAYBOOK_ID, "step_count": 231,
                    "plan_assessment": "STALE_HISTORY_PLAN", "skill_draft_count": 1,
                    "iterations": [], "stop_reason": "completed"},
    }
    return {
        "recent_workflow_events": [{"action": "audit", "from_stage": "auditing", "to_stage": "review",
            "correction_round": 1, "details": {"matched_relations": 115, "suggested_relations": 4},
            "created_at": "2026-10-01T10:00:00Z"}],
        "previous_reconciliation": {"process_mode": "agentic", "strategy_version": "li_ma_agentic_v1",
            "report": report, "error_code": None, "completed_at": "2026-10-01T10:00:00Z"},
    }


@pytest.mark.parametrize("setting", [None, "65536"])
def test_main_shaped_147k_learning_history_is_not_current_role_evidence(monkeypatch, setting):
    from test_agentic_audit import _small_context

    planner, worker, agent = audit._audit_input_budgets(setting)
    monkeypatch.setattr(audit, "MAX_PLANNER_INPUT_TOKENS", planner)
    monkeypatch.setattr(audit, "MAX_WORKER_CHUNK_TOKENS", worker)
    monkeypatch.setattr(audit, "MAX_AGENT_INPUT_TOKENS", agent)
    context = _small_context()
    context.update({
        "learning_context": _main_shaped_learning_context(),
        "source_inventory": {"transactions": deepcopy(context["transactions"]), "receipts": deepcopy(context["receipts"])},
        "source_inventory_summary": {"counts": {"transactions": 1, "receipts": 1}, "purpose": "deterministic_cache_only"},
        "strategy": "amount_first_iterative_v1", "iteration": {"number": 1, "phase": "exact_amount"},
        "active_skills": [{"skill_key": f"approved-{owner}", "version": 2, "owner_agent": owner,
            "title": "已批准通用规则", "description": "可复用规则而非历史关系",
            "content": {"trigger_conditions": ["同金额消歧"], "guidance": ["依据本期日期证据"],
                        "previous_report": {"summary": "STALE_HISTORY_SKILL_REPORT"}}}
            for owner in ("audit_planner", "evidence_worker")],
    })
    context["deterministic_candidates"][0]["evidence"] = {"source_evidence": "CURRENT_KERNEL_EVIDENCE"}
    before = deepcopy(context)
    raw = audit._serialized(context["learning_context"])
    assert len(raw.encode("utf-8")) >= 147_467
    assert len(raw) > 96_000  # Reproduces the old 64k-mode character guard failure.
    assert len(audit._serialized(context["source_inventory_summary"]).encode("utf-8")) < 324
    client = FakeClient([_plan(), _worker(), _final()])
    result = audit.analyze_agentic_audit(client, audit_id=f"learning-{setting}", context=context)
    assert "error_code" not in result
    assert len(client.calls) == 3
    assert {key: value for key, value in context.items() if key != "_agent_state"} == before
    for call in client.calls:
        prompt = call["messages"][1]["content"]
        assert "STALE_HISTORY" not in prompt
        assert "old-tx-" not in prompt and "old-receipt-" not in prompt
        audit._ensure_input_budget(call["messages"][0]["content"], prompt,
                                  planner if call["schema_name"] == "li_shifu_audit_plan" else agent)
    summary = json.loads(client.calls[0]["messages"][1]["content"].split("<audit_summary>")[1].split("</audit_summary>")[0])
    history = summary["learning_context"]
    assert len(audit._serialized(history).encode("utf-8")) <= audit.MAX_LEARNING_PROMPT_BYTES
    assert history["previous_relation_counts"]["matched_relations"] == 115
    assert history["previous_risk_count"] == 1
    assert history["recent_workflow_event_count"] == 1
    assert history["approved_skill_ids"] == ["approved-audit_planner"]
    assert "not_current_evidence_or_decisions" in history["history_scope"]
    payload = json.loads(client.calls[1]["messages"][1]["content"].split("<audit_data>")[1].split("</audit_data>")[0])
    assert "learning_context" not in payload
    assert [skill["skill_key"] for skill in payload["active_skills"]] == ["approved-evidence_worker"]
    assert payload["active_skills"][0]["content"]["guidance"] == ["依据本期日期证据"]
    assert payload["deterministic_candidates"][0]["evidence"]["source_evidence"] == "CURRENT_KERNEL_EVIDENCE"
    final = json.loads(client.calls[2]["messages"][1]["content"].split("<agent_results>")[1].split("</agent_results>")[0])
    assert final["summary"]["learning_context"] == history
    assert final["kernel_candidates"][0]["evidence"]["source_evidence"] == "CURRENT_KERNEL_EVIDENCE"


@pytest.mark.parametrize("setting", [None, "65536"])
@pytest.mark.parametrize("text", ["中文😀" * 80, "\x00\n\"\\" * 80])
def test_all_roles_bound_approved_skill_rules_not_arbitrary_run_content(monkeypatch, setting, text):
    from test_agentic_audit import _small_context

    planner, worker, agent = audit._audit_input_budgets(setting)
    monkeypatch.setattr(audit, "MAX_PLANNER_INPUT_TOKENS", planner)
    monkeypatch.setattr(audit, "MAX_WORKER_CHUNK_TOKENS", worker)
    monkeypatch.setattr(audit, "MAX_AGENT_INPUT_TOKENS", agent)
    context = _small_context()
    context["active_skills"] = [{"skill_key": f"skill-{i}", "owner_agent": owner, "status": "active",
        "title": text, "description": text, "content": {
            "trigger_conditions": [text] * 12, "guidance": [text] * 12, "principles": [text] * 12,
            "deviation_policy": text, "strategy_order": audit.SEED_STRATEGY_ORDER,
            "steps": [{"decisions": [{"transaction_id": "STALE_SKILL_DECISION"}]}],
        }} for owner in ("audit_planner", "evidence_worker") for i in range(30)]
    context["active_skills"].insert(0, {"skill_key": "unapproved", "status": "draft", "owner_agent": "audit_planner"})
    before = deepcopy(context)
    client = FakeClient([_plan(), _worker(), _final()])
    result = audit.analyze_agentic_audit(client, audit_id=f"skill-budget-{setting}", context=context)
    assert "error_code" not in result
    assert {key: value for key, value in context.items() if key != "_agent_state"} == before
    for owner in ("audit_planner", "evidence_worker"):
        skills = audit._prompt_skills(context, owner)
        assert skills
        assert len(audit._serialized(skills).encode("utf-8")) <= audit.MAX_SKILL_PROMPT_BYTES
        assert all(len(audit._serialized(skill).encode("utf-8")) <= audit.MAX_SKILL_ITEM_BYTES for skill in skills)
        assert all(skill["owner_agent"] == owner for skill in skills)
    for call in client.calls:
        prompt = call["messages"][1]["content"]
        assert "STALE_SKILL_DECISION" not in prompt and "unapproved" not in prompt
        audit._ensure_input_budget(call["messages"][0]["content"], prompt,
                                  planner if call["schema_name"] == "li_shifu_audit_plan" else agent)


def test_external_learning_history_does_not_consume_current_source_capacity():
    from test_agentic_audit import _small_context

    context = _small_context()
    original = audit._source_document(context)
    context["learning_context"] = _main_shaped_learning_context()
    context["learning_context"]["previous_reconciliation"]["report"]["summary"] *= 500
    assert len(audit._serialized(context["learning_context"])) > audit.MAX_AGENTIC_SOURCE_CHARS
    assert audit._source_document(context) == original
    summary = audit._summary_context(context)
    assert len(audit._serialized(summary["learning_context"]).encode("utf-8")) <= audit.MAX_LEARNING_PROMPT_BYTES
    context["transactions"][0]["description"] = "x" * (audit.MAX_AGENTIC_SOURCE_CHARS + 1)
    assert len(audit._source_document(context)) > audit.MAX_AGENTIC_SOURCE_CHARS


def test_history_skill_id_preview_has_an_independent_total_budget():
    context = {"active_skills": [{"skill_key": f"approved-{i}-" + "k" * 110,
               "owner_agent": "audit_planner", "content": {}} for i in range(30)]}
    skills = audit._prompt_skills(context, "audit_planner")
    history = audit._learning_prompt_context(context, skills)
    assert len(audit._serialized(history).encode("utf-8")) <= audit.MAX_LEARNING_PROMPT_BYTES
    assert history["approved_skill_ids_omitted"] > 0
    assert len(history["approved_skill_ids"]) + history["approved_skill_ids_omitted"] == len(skills)
    assert all(key in [skill["skill_key"] for skill in skills] for key in history["approved_skill_ids"])