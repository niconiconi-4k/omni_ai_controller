"""Offline approval/display regressions: no model, database or production inputs."""
from copy import deepcopy

import pytest

from omni_ai_controller import agentic_audit as agent
from test_agentic_audit import FakeClient, _final, _plan, _small_context
from test_cashflow_channel_agents import channel_context, evidence_and_decisions


def observation(tx="tx", receipts=None, **updates):
    return {"transaction_id": tx, "receipt_upload_ids": receipts or ["receipt"], "group_id": None,
            "finding": "真实活动与结算差额可披露", "evidence": ["完整候选关系、公司金额容差、实际0至3日活动"],
            "unresolved": [], **updates}


def decision(tx="tx", receipts=None, **updates):
    return {"transaction_id": tx, "receipt_upload_ids": receipts or ["receipt"],
            "recommendation": "match", "confidence": .95, **updates}


def approve(context=None, observations=None, decisions=None):
    context = context or _small_context()
    final = {**_final(), "decisions": decisions if decisions is not None else [decision()]}
    worker = {"observations": observations if observations is not None else [observation()]}
    used_receipts, used_transactions = set(), set()
    before = deepcopy(worker)
    approved = agent._approved_decisions(final, worker, agent._worker_chunks(context)[0], used_receipts, used_transactions)
    assert worker == before
    assert final["approval_diagnostics"]["requested_count"] == len(final["decisions"])
    assert final["approval_diagnostics"]["approved_count"] + final["approval_diagnostics"]["rejected_count"] == len(final["decisions"])
    return approved, final, used_receipts, used_transactions


@pytest.mark.parametrize("unresolved", [
    ["差额136.91成因未分解，需语义复核交易方"], ["交易方语义未确认"],
    ["手续费未分解"], ["真实全局同等方案"], ["币种不符"], ["证据冲突"],
])
def test_unresolved_is_not_keyword_cleared_and_rejection_is_visible(unresolved):
    approved, final, receipts, transactions = approve(observations=[observation(unresolved=unresolved)])
    assert approved == [] and not receipts and not transactions
    rejected = final["rejected_decisions"][0]
    assert rejected["reason_code"] == "worker_unresolved"
    assert rejected["worker_unresolved"] == unresolved
    assert rejected["model_recommendation"] == "match" and rejected["model_confidence"] == .95
    assert rejected["transaction_id"] == "tx" and rejected["receipt_upload_ids"] == ["receipt"]
    assert rejected["disposition"] == "rejected"


def test_wire_and_steps_distinguish_model_confirmation_from_actual_approval():
    context = _small_context()
    context["transactions"][0]["id"] = " bank:original-id "
    context["receipts"][0]["id"] = " receipt:original-id "
    context["deterministic_candidates"][0].update(transaction_id=" bank:original-id ", receipt_upload_id=" receipt:original-id ")
    worker = {"observations": [observation("T001", ["R001"], unresolved=["差额136.91成因未分解，需语义复核交易方"])]}
    model_decision = decision("T001", ["R001"], explanation="确认这组")
    client = FakeClient([_plan(), worker, {**_final(), "decisions": [model_decision], "summary": "确认T001/R001"}])
    result = agent.analyze_agentic_audit(client, audit_id="approval-diagnostics-wire", context=context)
    final = result["result"]
    step = result["steps"][-1]
    assert final["decisions"] == step["result"]["decisions"] == []
    assert step["result"]["model_summary"] == "确认T001/R001"
    for assessment in (final, step["result"]):
        assert "实际审批：确认 0 条" in assessment["summary"]
        assert "拒绝 1 条意向" in assessment["summary"]
        assert "模型评估文字（非实际审批结果）" in assessment["summary"]
        assert "worker_unresolved=1" in assessment["summary"]
        rejection = assessment["rejected_decisions"][0]
        assert rejection["transaction_id"] == " bank:original-id "
        assert rejection["receipt_upload_ids"] == [" receipt:original-id "]
        assert rejection["model_explanation"] == "确认这组"
    assert final["approval_diagnostics"]["rejected_count"] == 1
    assert result["usage"] == {"prompt_tokens": 600, "completion_tokens": 60, "total_tokens": 660}
    assert step["usage"] == {"prompt_tokens": 300, "completion_tokens": 30, "total_tokens": 330}
    assert [call["max_tokens"] for call in client.calls] == [3072, 2432, 2432]


@pytest.mark.parametrize("confidence", [.879, True, 1.01, None, "0.99", float("nan")])
def test_li_threshold_never_raised_by_display(confidence):
    approved, final, receipts, transactions = approve(decisions=[decision(confidence=confidence)])
    assert not approved and not receipts and not transactions
    assert final["rejected_decisions"][0]["reason_code"] == "confidence_below_threshold"


@pytest.mark.parametrize("confidence", [.88, .95, 1])
def test_full_evidence_preserves_legitimate_approval(confidence):
    intended = decision(confidence=confidence)
    approved, final, receipts, transactions = approve(decisions=[intended])
    assert approved == [intended] and receipts == {"receipt"} and transactions == {"tx"}
    assert final["rejected_decisions"] == []
    assert final["approval_diagnostics"]["approved_match_count"] == 1


@pytest.mark.parametrize("recommendation", ["match", "suggest"])
@pytest.mark.parametrize("fault,code", [
    ("scope", "outside_scope"), ("candidate", "not_kernel_candidate"),
    ("coverage", "missing_worker_coverage"), ("duplicate_ids", "invalid_receipt_ids"),
])
def test_illegal_or_uncovered_relations_never_enter_allocation(recommendation, fault, code):
    intended = decision(recommendation=recommendation)
    observations = [observation()]
    if fault == "scope":
        intended["transaction_id"] = "invented"
    elif fault == "candidate":
        intended["receipt_upload_ids"] = ["invented"]
    elif fault == "coverage":
        observations = []
    else:
        intended["receipt_upload_ids"] = ["receipt", "receipt"]
    approved, final, receipts, transactions = approve(observations=observations, decisions=[intended])
    assert not approved and not receipts and not transactions
    assert final["rejected_decisions"][0]["reason_code"] == code


@pytest.mark.parametrize("fault,code", [
    ("unknown", "unknown_cashflow"), ("currency", "currency_conflict"),
    ("blocked", "kernel_confirmation_blocked"), ("ambiguous", "kernel_confirmation_blocked"),
])
def test_model_clean_observation_cannot_override_explicit_hard_conflicts(fault, code):
    context = _small_context()
    if fault == "unknown":
        context["transactions"][0]["direction"] = "invalid"
    elif fault == "currency":
        context["transactions"][0]["currency"] = "SEK"
        context["receipts"][0]["currency"] = "EUR"
    else:
        context["deterministic_candidates"][0]["evidence"] = {
            "income_sequence": {"automatic_confirmation_blocked": True} if fault == "blocked"
            else {"globally_unambiguous": False}}
    approved, final, receipts, transactions = approve(context=context)
    assert not approved and not receipts and not transactions
    assert final["rejected_decisions"][0]["reason_code"] == code


@pytest.mark.parametrize("currency", [None, "", "unknown", "?", "123"])
def test_explicit_unknown_currency_is_not_cured_by_clean_model_evidence(currency):
    context = _small_context()
    context["transactions"][0]["currency"] = currency
    context["receipts"][0]["currency"] = "SEK"
    approved, final, receipts, transactions = approve(context=context)
    assert not approved and not receipts and not transactions
    assert "unknown_currency" in final["rejected_decisions"][0]["reason_codes"]


def test_competing_complete_groups_are_not_chosen_by_model_order_or_double_counted():
    first = channel_context(2)
    second = channel_context(2, group="other", receipt="other:0")
    chunk = {"transactions": first["transactions"], "receipts": first["receipts"] + second["receipts"],
             "deterministic_candidates": first["deterministic_candidates"] + second["deterministic_candidates"]}
    worker, final = evidence_and_decisions(first)
    receipts, transactions = set(), set()
    assert agent._approved_decisions(final, worker, chunk, receipts, transactions) == []
    assert not receipts and not transactions
    assert final["approval_diagnostics"]["requested_count"] == final["approval_diagnostics"]["rejected_count"] == 2
    assert all(row["reason_code"] == "kernel_confirmation_blocked" for row in final["rejected_decisions"])


def test_no_completed_assessments_have_explicit_zero_counts():
    result = {"summary": "没有完成评估", **agent._combined_approval_diagnostics([])}
    agent._mark_approval_summary(result)
    assert result["approval_diagnostics"]["approved_match_count"] == 0
    assert result["approval_diagnostics"]["rejected_count"] == 0
    assert "实际审批：确认 0 条" in result["summary"]


def test_rejected_intention_does_not_steal_resources_from_later_approved_relation():
    low, valid = decision(confidence=.5), decision()
    approved, final, receipts, transactions = approve(decisions=[low, valid])
    assert approved == [valid] and receipts == {"receipt"} and transactions == {"tx"}
    assert final["approval_diagnostics"]["rejected_count"] == 1
    assert final["rejected_decisions"][0]["model_confidence"] == .5


def test_empty_suggestion_cannot_allocate_a_transaction_without_candidate_relation():
    intended = decision(recommendation="suggest")
    intended["receipt_upload_ids"] = []
    approved, final, receipts, transactions = approve(decisions=[intended])
    assert not approved and not receipts and not transactions
    assert final["rejected_decisions"][0]["reason_code"] == "not_kernel_candidate"


def test_conflicting_suggestion_cannot_replace_an_approved_match():
    valid, suggested = decision(), decision(recommendation="suggest", confidence=.8)
    approved, final, receipts, transactions = approve(decisions=[valid, suggested])
    assert approved == [valid] and receipts == {"receipt"} and transactions == {"tx"}
    assert final["rejected_decisions"][0]["reason_code"] == "resource_conflict"
    assert final["approval_diagnostics"]["approved_match_count"] == 1
    assert final["approval_diagnostics"]["approved_suggest_count"] == 0


@pytest.mark.parametrize("fault,code", [
    ("half", "incomplete_group_decisions"), ("mixed", "mixed_group_recommendations"),
    ("coverage", "missing_worker_coverage"), ("unresolved", "worker_unresolved"),
    ("blocked", "unproven_channel_group"), ("low", "confidence_below_threshold"),
])
def test_group_rejection_is_atomic_and_visible_on_every_original_row(fault, code):
    context = channel_context()
    worker, final = evidence_and_decisions(context)
    if fault == "half":
        final["decisions"].pop()
    elif fault == "mixed":
        final["decisions"][0]["recommendation"] = "suggest"
    elif fault == "coverage":
        worker["observations"].pop()
    elif fault == "unresolved":
        worker["observations"][0]["unresolved"] = ["证据冲突"]
    elif fault == "blocked":
        context["deterministic_candidates"][0]["evidence"]["automatic_confirmation_blocked"] = True
    else:
        final["decisions"][0]["confidence"] = .8
    approved, saved, receipts, transactions = approve(context, worker["observations"], final["decisions"])
    assert not approved and not receipts and not transactions
    assert len(saved["rejected_decisions"]) == len(final["decisions"])
    assert all(code in row["reason_codes"] and row["group_id"] == "channels" for row in saved["rejected_decisions"])
    assert all(row["receipt_upload_ids"] == ["original:0"] for row in saved["rejected_decisions"])


def test_rejection_payload_and_aggregate_are_bounded_but_counts_complete():
    count = agent.MAX_APPROVAL_REJECTIONS + 25
    intended = [decision(confidence=.5, explanation="long" * 2000) for _ in range(count)]
    worker = observation(unresolved=["X" * 5000 for _ in range(30)])
    approved, final, receipts, transactions = approve(observations=[worker], decisions=intended)
    assert not approved and not receipts and not transactions
    assert len(final["rejected_decisions"]) == agent.MAX_APPROVAL_REJECTIONS
    diagnostics = final["approval_diagnostics"]
    assert diagnostics["rejected_count"] == count and diagnostics["rejected_decisions_omitted"] == 25
    assert diagnostics["rejection_counts"] == {"confidence_below_threshold": count, "worker_unresolved": count}
    for row in final["rejected_decisions"]:
        assert len(row["model_explanation"]) <= agent.MAX_APPROVAL_DETAIL_CHARS
        assert len(row["worker_unresolved"]) == 8 and row["worker_unresolved_omitted"] == 22
        assert all(len(value) == agent.MAX_APPROVAL_DETAIL_CHARS for value in row["worker_unresolved"])
    aggregate = agent._combined_approval_diagnostics([final, final])
    assert len(aggregate["rejected_decisions"]) == agent.MAX_APPROVAL_REJECTIONS
    assert aggregate["approval_diagnostics"]["rejected_count"] == 2 * count
    assert aggregate["approval_diagnostics"]["rejected_decisions_omitted"] == 2 * count - agent.MAX_APPROVAL_REJECTIONS


def test_summary_only_and_forged_diagnostics_cannot_approve_anything():
    final = {**_final(), "decisions": [], "summary": "确认T001/R001全部对应",
             "approval_diagnostics": {"approved_match_count": 999}, "rejected_decisions": ["forged"]}
    assert agent._approved_decisions(final, {"observations": [observation()]}, agent._worker_chunks(_small_context())[0]) == []
    agent._mark_approval_summary(final)
    assert final["rejected_decisions"] == [] and final["approval_diagnostics"]["approved_match_count"] == 0
    assert final["model_summary"] == "确认T001/R001全部对应"
    assert "实际审批：确认 0 条" in final["summary"]


def test_cancellation_preserves_rejection_diagnostics_without_freezing(monkeypatch):
    context = _small_context()
    context["transactions"].append({"id": "tx2", "amount": 1200, "direction": "debit"})
    context["receipts"].append({"id": "receipt2", "amount": 1200, "type": "expense"})
    context["deterministic_candidates"].append({"transaction_id": "tx2", "receipt_upload_id": "receipt2"})
    monkeypatch.setattr(agent, "MAX_WORKER_CHUNK_TRANSACTIONS", 1)

    class CancelAfterRejection(FakeClient):
        def chat_json(self, messages, **kwargs):
            response = super().chat_json(messages, **kwargs)
            if len(self.calls) == 3:
                agent.cancel_audit_run("cancel-rejected-assessment")
            return response

    client = CancelAfterRejection([_plan(), {"observations": [observation(unresolved=["交易方语义未确认"])]},
                                  {**_final(), "decisions": [decision()], "summary": "确认T001/R001"}])
    result = agent.analyze_agentic_audit(client, audit_id="cancel-rejected-assessment", context=context)
    assert result["error_code"] == "agentic_cancelled" and result["result"]["decisions"] == []
    assert result["result"]["approval_diagnostics"]["rejected_count"] == 1
    assert result["result"]["rejected_decisions"][0]["reason_code"] == "worker_unresolved"
    assert "实际审批：确认 0 条" in result["result"]["summary"]


def test_verified_confirmed_resources_are_excluded_not_reapproved_by_model():
    context = _small_context()
    context.update(confirmed_transaction_ids=["tx"], confirmed_receipt_ids=["receipt"])
    assert agent._worker_chunks(context) == []
    for prompt in (agent._WORKER_SYSTEM_PROMPT, agent._FINAL_SYSTEM_PROMPT):
        assert "不需再次由模型批准" in prompt
        assert "POS商户与银行PSP/支付结算方名称不必相同" in prompt
        assert "本身仅是披露风险" in prompt
        assert "真实全局同等方案" in prompt and "半组或证据冲突" in prompt
    assert agent._audit_input_budgets(None) == (16000, 12000, 18000)
    assert agent._audit_input_budgets("65536") == (48000, 40000, 48000)