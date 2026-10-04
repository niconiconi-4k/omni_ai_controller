"""Controller-only regressions for cashflow lanes and atomic original-child groups."""
from copy import deepcopy
import json

import pytest

from omni_ai_controller import agentic_audit as agent
from omni_ai_controller.audit_inventory import cashflow_facts
from omni_ai_controller.audit_notebooks import AuditNotebooks
from omni_ai_controller.audit_skill import AuditSkillError
from test_agentic_audit import FakeClient, _final, _plan, _worker


def channel_context(count=4, group="channels", receipt="original:0", prefix="bank"):
    ids = [f"{prefix}-{i}" for i in range(count)]
    rows = [{"id": tx, "amount": 100, "currency": "SEK", "direction": "credit",
             "date": f"2026-10-0{1 + i % 3}", "category": "income"} for i, tx in enumerate(ids)]
    candidates = [{"transaction_id": tx, "receipt_upload_id": receipt, "group_id": group,
                   "match_group_id": group, "allocation_role": "income_payment_channel", "status": "suggested",
                   "evidence": {"method": "income_payment_channels_v1", "atomic_group": True,
                       "complete_receipt_group": True, "group_transaction_ids": ids[:], "group_row_count": count,
                       "channel": "card" if i == 0 else "swish", "channel_transaction_count": 21 if i == 0 else count - 1,
                       "actual_channel_transaction_count": 1 if i == 0 else count - 1,
                       "channel_bank_total": "100" if i == 0 else str(100 * (count - 1)),
                       "channel_description_evidence": ["explicit label", "phone nearby"],
                       "globally_unambiguous": True, "automatic_confirmation_blocked": False,
                       "confirmation_basis": "complete_channels_globally_forced",
                       "alternative_groups": [{"transaction_ids": ids[:], "receipt_id": receipt}],
                       "resource_conflict_groups": []}} for i, tx in enumerate(ids)]
    return {"transactions": rows, "receipts": [{"id": receipt, "amount": count * 100, "type": "income",
                "currency": "SEK", "date": "2026-10-01"}], "deterministic_candidates": candidates}


def evidence_and_decisions(context, recommendation="match"):
    observations, decisions = [], []
    for row in context["deterministic_candidates"]:
        observations.append({"transaction_id": row["transaction_id"], "receipt_upload_ids": [row["receipt_upload_id"]],
            "group_id": row["group_id"], "finding": "complete joint kernel proof; difference not itemized",
            "evidence": ["explicit count + aggregate sum + nearby time + globally forced complete group"], "unresolved": []})
        decisions.append({"transaction_id": row["transaction_id"], "receipt_upload_ids": [row["receipt_upload_id"]],
                          "recommendation": recommendation, "confidence": .96})
    return {**_worker(), "observations": observations}, {**_final(), "decisions": decisions}


def merge(*contexts):
    return {key: [row for context in contexts for row in context[key]]
            for key in ("transactions", "receipts", "deterministic_candidates")}


def single(tx, lane, category="expense", role="direct_expense"):
    return {"transactions": [{"id": tx, "amount": 100, "direction": "credit" if lane == "in" else "debit" if lane == "out" else "invalid",
                              "date": "2026-10-02"}],
            "receipts": [{"id": "r-" + tx, "amount": 100, "type": category}],
            "deterministic_candidates": [{"transaction_id": tx, "receipt_upload_id": "r-" + tx, "allocation_role": role}]}


@pytest.mark.parametrize("count", [2, 3, 4, 12])
def test_complete_channel_bank_rows_are_one_indivisible_budgeted_chunk(count):
    context = channel_context(count)
    before = deepcopy(context)
    chunks = agent._worker_chunks(context)
    assert len(chunks) == 1 and len(chunks[0]["transactions"]) == count
    assert agent._split_worker_chunk(chunks[0]) == []
    assert agent._estimated_tokens(agent._serialized(chunks[0])) <= agent.MAX_WORKER_CHUNK_TOKENS
    assert chunks[0]["cashflow_lane"] == "in"
    assert context == before
    worker, final = evidence_and_decisions(context)
    used_receipts, used_transactions = set(), set()
    assert agent._validate_response(final, worker, chunks[0], used_receipts, used_transactions) == final["decisions"]
    assert used_receipts == {"original:0"}
    assert len(used_transactions) == count


def test_controller_accepts_256_member_group_and_rejects_257():
    assert agent._complete_channel_group(channel_context(256)["deterministic_candidates"])
    assert not agent._complete_channel_group(channel_context(257)["deterministic_candidates"])


def test_channel_role_routes_to_revenue_not_direct_expenses():
    assert agent._strategy({"allocation_role": "income_payment_channel"}) == "revenue_settlements"
    assert agent._strategy({"allocation_role": "payroll"}) == "employee_reimbursements"
    assert agent._strategy({"allocation_role": "corporate_card"}) == "corporate_card_expenses"
    assert agent.SEED_STRATEGY_ORDER[0] == "corporate_card_expenses"


def test_all_paths_partition_cashflow_even_one_strategy_selects_every_transaction():
    context = merge(single("expense", "out"), single("income", "in", "income", "direct_expense"),
                    single("payroll", "out", "payroll", "payroll"), single("unknown", "unknown", "unknown"))
    chunks = agent._worker_chunks(context)
    assert [chunk["cashflow_lane"] for chunk in chunks] == ["in", "out", "unknown"]
    assert all(len({cashflow_facts(tx, source_kind="transaction")["cashflow_lane"] for tx in chunk["transactions"]}) == 1 for chunk in chunks)
    plan = {"tasks": [{"task_id": "mixed", "strategy": "direct_expenses", "transaction_ids": [tx["id"] for tx in context["transactions"]]}]}
    notebooks = AuditNotebooks("lanes", "lanes")
    jobs = agent._scoped_jobs(plan, context, notebooks)
    assert [chunk["cashflow_lane"] for chunk in jobs] == ["in", "out", "unknown"]
    for job in jobs:
        assert set(job["tasks"][0]["transaction_ids"]) == {tx["id"] for tx in job["transactions"]}
    for recommendation in ("match", "suggest"):
        unknown = jobs[-1]
        decision = {"transaction_id": "unknown", "receipt_upload_ids": ["r-unknown"], "recommendation": recommendation, "confidence": .99}
        worker = {"observations": [{"transaction_id": "unknown", "receipt_upload_ids": ["r-unknown"], "group_id": None,
                    "finding": "amount equal", "evidence": ["amount"], "unresolved": []}]}
        assert agent._approved_decisions({"decisions": [decision]}, worker, unknown) == ([decision] if recommendation == "suggest" else [])


@pytest.mark.parametrize("scope", ["transaction", "receipt", "strategy"])
def test_selecting_one_channel_row_extends_task_scope_and_preserves_complete_cache(scope):
    context = channel_context()
    context["source_inventory"] = deepcopy({key: context[key] for key in ("transactions", "receipts")})
    raw = {"task_id": "selected", "strategy": "revenue_settlements", "operation": "compare_details"}
    if scope == "transaction":
        raw["transaction_ids"] = ["bank-2"]
    if scope == "receipt":
        raw["receipt_ids"] = ["original:0"]
    notebooks = AuditNotebooks("scope", "scope")
    notebooks.organize_inventory(context["source_inventory"])
    jobs = agent._scoped_jobs({"tasks": [raw]}, context, notebooks)
    assert len(jobs) == 1
    job = jobs[0]
    assert set(job["tasks"][0]["transaction_ids"]) == {f"bank-{i}" for i in range(4)}
    first = notebooks.prepare(job, job["tasks"])
    second = notebooks.prepare(job, job["tasks"])
    assert len(first["transaction_facts"]) == len(second["transaction_facts"]) == 4
    assert first == second
    assert notebooks.snapshot()["stats"]["cache_hits"] >= 5


@pytest.mark.parametrize("fault", ["omission", "partial_receipt", "mixed_decisions", "duplicate", "wrong_receipt",
                                    "no_observation", "partial_observation", "wrong_group", "unresolved", "low_confidence"])
def test_final_channel_approval_is_all_or_nothing(fault):
    context = channel_context()
    worker, final = evidence_and_decisions(context)
    if fault == "omission":
        final["decisions"].pop()
    elif fault == "partial_receipt":
        final["decisions"][0]["receipt_upload_ids"] = []
    elif fault == "mixed_decisions":
        final["decisions"][0]["recommendation"] = "suggest"
    elif fault == "duplicate":
        final["decisions"].append(deepcopy(final["decisions"][0]))
    elif fault == "wrong_receipt":
        final["decisions"][0]["receipt_upload_ids"] = ["invented:0"]
    elif fault == "no_observation":
        worker["observations"] = []
    elif fault == "partial_observation":
        worker["observations"].pop()
    elif fault == "wrong_group":
        worker["observations"][0]["group_id"] = "other"
    elif fault == "unresolved":
        worker["observations"][0]["unresolved"] = ["currency conflict"]
    else:
        final["decisions"][0]["confidence"] = .87
    receipts, transactions = set(), set()
    chunk = agent._worker_chunks(context)[0]
    assert agent._approved_decisions(final, worker, chunk, receipts, transactions) == []
    assert not receipts and not transactions


@pytest.mark.parametrize("fault", ["missing_member", "wrong_count", "wrong_declared_ids", "missing_group", "different_child", "missing_atomic",
                                    "wrong_method", "duplicate_bank", "too_many"])
def test_malformed_or_oversized_channel_group_does_not_drop_other_valid_chunks(fault):
    bad = channel_context(257 if fault == "too_many" else 4)
    rows = bad["deterministic_candidates"]
    if fault == "missing_member":
        rows.pop()
    elif fault == "wrong_count":
        rows[0]["evidence"]["group_row_count"] = 3
    elif fault == "wrong_declared_ids":
        rows[0]["evidence"]["group_transaction_ids"][-1] = "invented"
    elif fault == "missing_group":
        rows[0]["group_id"] = rows[0]["match_group_id"] = None
    elif fault == "different_child":
        rows[0]["receipt_upload_id"] = "original:1"
        bad["receipts"].append({"id": "original:1", "amount": 100, "type": "income"})
    elif fault == "missing_atomic":
        rows[0]["evidence"]["atomic_group"] = False
    elif fault == "wrong_method":
        rows[0]["evidence"]["method"] = "invented"
    elif fault == "duplicate_bank":
        rows.append(deepcopy(rows[0]))
    good = channel_context(3, "good", "other:0", "other-bank")
    chunks = agent._worker_chunks(merge(bad, good))
    assert len(chunks) == 1
    assert {row["group_id"] for row in chunks[0]["deterministic_candidates"]} == {"good"}
    worker, final = evidence_and_decisions(good)
    assert agent._approved_decisions(final, worker, chunks[0]) == final["decisions"]


@pytest.mark.parametrize("fault", ["blocked", "ambiguous", "unknown_channel", "missing_basis", "unknown_bank", "unknown_receipt"])
def test_unproven_or_unknown_channel_group_can_only_be_suggested_as_a_whole(fault):
    context = channel_context()
    row = context["deterministic_candidates"][0]["evidence"]
    if fault == "blocked":
        row["automatic_confirmation_blocked"] = True
    elif fault == "ambiguous":
        row["globally_unambiguous"] = False
    elif fault == "unknown_channel":
        row["channel"] = "unknown"
    elif fault == "missing_basis":
        row.pop("confirmation_basis")
    elif fault == "unknown_bank":
        for tx in context["transactions"]:
            tx["direction"] = "invalid"
    else:
        context["receipts"][0]["type"] = "unknown"
    chunk = agent._worker_chunks(context)[0]
    worker, final = evidence_and_decisions(context)
    assert agent._approved_decisions(final, worker, chunk) == []
    worker, final = evidence_and_decisions(context, "suggest")
    assert agent._approved_decisions(final, worker, chunk) == final["decisions"]


@pytest.mark.parametrize("confirmed", ["transaction", "receipt", "missing_source", "mixed_lane"])
def test_atomic_exclusion_never_leaves_half_channel_group(confirmed):
    context = channel_context()
    if confirmed == "transaction":
        context["confirmed_transaction_ids"] = ["bank-0"]
    elif confirmed == "receipt":
        context["confirmed_receipt_ids"] = ["original:0"]
    elif confirmed == "missing_source":
        context["transactions"].pop()
    else:
        context["transactions"][0]["direction"] = "debit"
    assert agent._worker_chunks(context) == []


def test_full_run_aliases_restore_ids_and_never_duplicate_original_child():
    context = channel_context()
    context["source_inventory"] = deepcopy({key: context[key] for key in ("transactions", "receipts")})
    worker, final = evidence_and_decisions(context)
    aliases = agent._id_aliases(context)
    worker = agent._worker_result_for_model(worker, aliases)
    final["decisions"] = [{**row, "transaction_id": aliases["tx_forward"][row["transaction_id"]], "receipt_upload_ids": ["R001"]}
                          for row in final["decisions"]]
    plan = {**_plan(), "tasks": [{"task_id": "one-row", "strategy": "revenue_settlements", "transaction_ids": ["T003"]}]}
    client = FakeClient([plan, worker, final])
    result = agent.analyze_agentic_audit(client, audit_id="channel-wire", context=context)
    assert result["worker_batch_count"] == 1 and len(client.calls) == 3
    assert {row["transaction_id"] for row in result["result"]["decisions"]} == {f"bank-{i}" for i in range(4)}
    for call in client.calls[1:]:
        prompt = call["messages"][1]["content"]
        assert all(tx["id"] not in prompt for tx in context["transactions"])
        assert "original:0" not in prompt
        assert all(f"T00{i}" in prompt for i in range(1, 5))
        assert "group_row_count\":4" in prompt
    payload = json.loads(client.calls[1]["messages"][1]["content"].split("<audit_data>")[1].split("</audit_data>")[0])
    assert payload["tasks"][0]["transaction_ids"] == ["T001", "T004", "T002", "T003"]
    assert len(payload["transactions"]) == 4
    assert "显式通道金额和笔数" in client.calls[1]["messages"][0]["content"]


@pytest.mark.parametrize("stage", ["worker", "final"])
def test_truncation_never_splits_atomic_group_and_other_valid_group_continues(stage):
    first = channel_context(4)
    second = channel_context(3, "second", "second:0", "second-bank")
    worker, final = evidence_and_decisions(second)
    responses = [_plan()]
    if stage == "final":
        responses.append(evidence_and_decisions(first)[0])
    responses += ["{truncated", worker, final]
    client = FakeClient(responses, length_calls={2 if stage == "worker" else 3})
    result = agent.analyze_agentic_audit(client, audit_id="atomic-truncation-" + stage, context=merge(first, second))
    assert result["error_code"] == "agentic_output_limit"
    assert result["worker_batch_count"] == 2
    assert result["result"]["decisions"] == final["decisions"]
    assert sum(call["schema_name"] == "ma_shifu_evidence_review" for call in client.calls) == 2


def test_whole_group_token_budget_counts_every_row_without_global_increase(monkeypatch):
    context = channel_context()
    context["transactions"][-1]["memo"] = "large evidence" * 5000
    monkeypatch.setattr(agent, "MAX_WORKER_CHUNK_TOKENS", 1500)
    with pytest.raises(AuditSkillError, match="单个交易候选组"):
        agent._worker_chunks(context)
    assert agent.MAX_WORKER_CHUNK_TRANSACTIONS == 2
    assert agent._audit_input_budgets(None) == (16000, 12000, 18000)
    assert agent._audit_input_budgets("65536") == (48000, 40000, 48000)


def test_runtime_approved_role_skills_only_and_seed_remains_read_only():
    seed = deepcopy(agent.SEED_STRATEGY_ORDER)
    context = {"active_skills": [
        {"skill_key": "company", "owner_agent": "evidence_worker", "status": "active", "version": 3,
         "content": {"guidance": ["latest company channel rule"], "trigger_conditions": ["explicit channels"], "arbitrary_report": "not evidence"}},
        {"skill_key": "retired", "owner_agent": "evidence_worker", "status": "retired", "content": {"guidance": ["old rule"]}},
        {"skill_key": "planner", "owner_agent": "audit_planner", "status": "active", "content": {"guidance": ["planner only"]}},
    ]}
    rules = agent._prompt_skills(context, "evidence_worker")
    assert [row["skill_key"] for row in rules] == ["company"]
    assert rules[0]["version"] == 3
    assert rules[0]["content"]["guidance"] == ["latest company channel rule"]
    assert "arbitrary_report" not in agent._serialized(rules)
    context["active_skills"][0]["content"]["guidance"] = ["updated runtime rule"]
    assert agent._prompt_skills(context, "evidence_worker")[0]["content"]["guidance"] == ["updated runtime rule"]
    assert agent.SEED_STRATEGY_ORDER == seed