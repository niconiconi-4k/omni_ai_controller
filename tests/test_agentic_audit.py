import json
from types import SimpleNamespace

import pytest

from omni_ai_controller.agentic_audit import (
    MAX_AGENTIC_SOURCE_CHARS,
    SEED_PLAYBOOK_ID,
    analyze_agentic_audit,
    get_audit_progress,
    _worker_chunks,
)
from omni_ai_controller.audit_skill import AuditSkillError


class FakeClient:
    def __init__(self, responses: list[dict[str, object] | str], *, length_calls: set[int] | None = None) -> None:
        self.responses = list(responses)
        self.length_calls = length_calls or set()
        self.config = SimpleNamespace(model_name="local-qwen")
        self.calls: list[dict[str, object]] = []

    def chat_json(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append({"messages": messages, **kwargs})
        response = self.responses.pop(0)
        content = response if isinstance(response, str) else json.dumps(response, ensure_ascii=False)
        index = len(self.calls)
        return SimpleNamespace(
            content=content,
            raw={
                "id": f"agent-step-{index}",
                "choices": [{"finish_reason": "length" if index in self.length_calls else "stop"}],
                "usage": {
                    "prompt_tokens": 100 * index,
                    "completion_tokens": 10 * index,
                    "total_tokens": 110 * index,
                },
            },
        )


def _plan() -> dict[str, object]:
    return {
        "objective": "先支出后收入完成核对",
        "strategy_order": [
            "corporate_card_expenses",
            "direct_expenses",
            "revenue_settlements",
            "employee_reimbursements",
            "anomaly_review",
        ],
        "tasks": [{
            "task_id": "task-1",
            "strategy": "direct_expenses",
            "objective": "核对支出候选",
            "priority": 100,
            "evidence_requirements": ["金额", "日期"],
        }],
        "deviations": [],
        "risk_focus": ["个人与公司转账"],
    }


def _worker() -> dict[str, object]:
    return {
        "task_results": [{
            "task_id": "task-1",
            "status": "completed",
            "finding": "证据充分",
            "evidence": ["金额一致"],
            "unresolved": [],
        }],
        "decisions": [],
        "summary": "完成证据核对",
        "risks": [],
        "cache_notes": ["按日期索引候选"],
    }


def _final() -> dict[str, object]:
    return {
        "decisions": [],
        "summary": "智能审计完成",
        "risks": [],
        "plan_assessment": "初始作业法适用",
        "skill_candidates": [],
    }


def test_agentic_audit_runs_li_ma_li_with_shared_seed_playbook() -> None:
    client = FakeClient([_plan(), _worker(), _final()])

    result = analyze_agentic_audit(
        client,
        audit_id="case-1",
        context={
            "profile": {"business_type": "电商"},
            "transactions": [{"id": "tx-1"}],
            "receipts": [{"id": "receipt-1"}],
            "deterministic_candidates": [{
                "transaction_id": "tx-1",
                "receipt_upload_id": "receipt-1",
                "allocation_role": "direct_expense",
            }],
        },
    )

    assert result["process_mode"] == "agentic"
    assert result["seed_playbook"] == SEED_PLAYBOOK_ID
    assert [step["agent_kind"] for step in result["steps"]] == [
        "audit_planner", "evidence_worker", "audit_planner",
    ]
    assert [call["schema_name"] for call in client.calls] == [
        "li_shifu_audit_plan",
        "ma_shifu_evidence_review",
        "li_shifu_final_assessment",
    ]
    assert result["usage"]["total_tokens"] == 660
    assert "accounting-expense-first-v1" in client.calls[0]["messages"][0]["content"]
    assert "马师傅" in client.calls[1]["messages"][0]["content"]
    snapshot = get_audit_progress("case-1")
    assert snapshot["status"] == "completed"
    assert len(snapshot["steps"]) == 3


def test_agentic_audit_rejects_oversized_context_before_any_call() -> None:
    client = FakeClient([_plan(), _worker(), _final()])

    with pytest.raises(AuditSkillError, match="超过智能流程工作预算"):
        analyze_agentic_audit(
            client,
            audit_id="case-1",
            context={"payload": "x" * (MAX_AGENTIC_SOURCE_CHARS + 1)},
        )

    assert client.calls == []


def test_agentic_audit_fails_closed_on_invalid_agent_response() -> None:
    client = FakeClient(["not-json"])

    with pytest.raises(AuditSkillError, match="返回格式无效"):
        analyze_agentic_audit(
            client,
            audit_id="case-1",
            context={
                "transactions": [{"id": "tx-1"}],
                "receipts": [{"id": "receipt-1"}],
                "deterministic_candidates": [{
                    "transaction_id": "tx-1",
                    "receipt_upload_id": "receipt-1",
                }],
            },
        )


def test_agentic_audit_splits_large_evidence_by_transaction() -> None:
    from omni_ai_controller.agentic_audit import MAX_WORKER_CHUNK_TOKENS

    client = FakeClient([_plan(), _worker(), _final(), _worker(), _final()])
    # Each receipt fits independently; both exceed the configured chunk budget.
    long_text = "凭证内容" * (MAX_WORKER_CHUNK_TOKENS // 8)
    context = {
        "profile": {},
        "transactions": [{"id": "tx-1"}, {"id": "tx-2"}],
        "receipts": [
            {"id": "receipt-1", "ocr_excerpt": long_text},
            {"id": "receipt-2", "ocr_excerpt": long_text},
        ],
        "deterministic_candidates": [
            {
                "transaction_id": "tx-1",
                "receipt_upload_id": "receipt-1",
                "allocation_role": "direct_expense",
            },
            {
                "transaction_id": "tx-2",
                "receipt_upload_id": "receipt-2",
                "allocation_role": "direct_expense",
            },
        ],
    }

    result = analyze_agentic_audit(client, audit_id="case-1", context=context)

    assert result["worker_batch_count"] == 2
    assert len(result["steps"]) == 5
    assert [call["schema_name"] for call in client.calls].count(
        "ma_shifu_evidence_review"
    ) == 2


def test_agentic_failure_preserves_completed_plan_and_failure_stage() -> None:
    client = FakeClient([_plan(), "not-json"])
    context = {
        "_run_id": "failed-run",
        "transactions": [{"id": "tx"}],
        "receipts": [{"id": "receipt"}],
        "deterministic_candidates": [{"transaction_id": "tx", "receipt_upload_id": "receipt"}],
    }
    with pytest.raises(AuditSkillError, match="JSON 在字符"):
        analyze_agentic_audit(client, audit_id="audit", context=context)
    progress = get_audit_progress("failed-run")
    assert progress["status"] == "failed"
    assert progress["stage"] == "evidence_review"
    assert len(progress["steps"]) == 1
    assert progress["steps"][0]["result"]["objective"] == _plan()["objective"]
    assert progress["last_response"]["output_characters"] == 8


def test_agentic_rejects_length_finish_reason_even_for_valid_json() -> None:
    client = FakeClient([_plan(), _plan()], length_calls={1, 2})
    result = analyze_agentic_audit(client, audit_id="truncated", context=_small_context())
    assert result["error_code"] == "agentic_output_limit"
    assert result["result"]["decisions"] == []
    assert len(client.calls) == 2
    assert all(step["status"] == "failed" for step in result["steps"])
    assert get_audit_progress("truncated")["status"] == "partial"


def test_worker_chunks_limit_output_candidates_without_splitting_transaction() -> None:
    context = {
        "transactions": [{"id": f"tx-{i}"} for i in range(10)],
        "receipts": [{"id": f"receipt-{i}"} for i in range(10)],
        "deterministic_candidates": [
            {"transaction_id": f"tx-{i}", "receipt_upload_id": f"receipt-{i}"}
            for i in range(10)
        ],
    }
    chunks = _worker_chunks(context)
    assert [len(chunk["deterministic_candidates"]) for chunk in chunks] == [2, 2, 2, 2, 2]
    assert {item["transaction_id"] for chunk in chunks for item in chunk["deterministic_candidates"]} == {f"tx-{i}" for i in range(10)}


def _small_context() -> dict:
    return {
        "transactions": [{"id": "tx", "amount": 1200, "direction": "debit"}],
        "receipts": [{"id": "receipt", "amount": 1200, "type": "expense"}],
        "deterministic_candidates": [{"transaction_id": "tx", "receipt_upload_id": "receipt"}],
    }


def test_replanned_round_keeps_prior_steps_usage_and_round_number():
    prior = [{"sequence_number": 1, "agent_kind": "audit_planner", "step_kind": "initial_plan", "status": "completed", "usage": {"total_tokens": 7}}]
    context = {**_small_context(), "_prior_steps": prior, "iteration": {"number": 2, "phase": "exact_amount"}, "strategy": "amount_first_iterative_v1"}
    result = analyze_agentic_audit(FakeClient([_plan(), _worker(), _final()]), audit_id="round-2", context=context)
    assert [step["sequence_number"] for step in result["steps"]] == [1, 2, 3, 4]
    assert result["usage"]["total_tokens"] == 667
    assert result["steps"][1]["input_summary"]["iteration"] == 2
    assert result["progress"]["iteration"] == 2
    assert len(prior) == 1


@pytest.mark.parametrize("worker_receipt", [None, "invented"])
def test_planner_cannot_promote_without_worker_kernel_evidence(worker_receipt):
    decision = {"transaction_id": "tx", "receipt_upload_ids": ["receipt"], "recommendation": "match", "confidence": 0.99}
    worker = _worker()
    if worker_receipt:
        worker["decisions"] = [{**decision, "receipt_upload_ids": [worker_receipt]}]
    final = {**_final(), "decisions": [decision]}
    result = analyze_agentic_audit(FakeClient([_plan(), worker, final]), audit_id="unbacked", context=_small_context())
    assert result["result"]["decisions"] == []


def test_worker_and_planner_can_approve_same_kernel_relation():
    decision = {"transaction_id": "tx", "receipt_upload_ids": ["receipt"], "recommendation": "match", "confidence": 0.95}
    result = analyze_agentic_audit(FakeClient([_plan(), {**_worker(), "decisions": [decision]}, {**_final(), "decisions": [decision]}]), audit_id="backed", context=_small_context())
    assert result["result"]["decisions"] == [decision]


def _grouped_context() -> dict:
    return {
        "transactions": [{"id": "tx-1", "amount": 1200, "direction": "debit"}, {"id": "tx-2", "amount": 900, "direction": "debit"}],
        "receipts": [{"id": f"receipt-{index}", "amount": 1200, "type": "expense"} for index in range(3)],
        "deterministic_candidates": [
            {"transaction_id": "tx-1", "receipt_upload_id": "receipt-0"},
            {"transaction_id": "tx-1", "receipt_upload_id": "receipt-1"},
            {"transaction_id": "tx-2", "receipt_upload_id": "receipt-2"},
        ],
    }


def test_truncated_plan_retries_with_bounded_output_and_tracks_all_usage() -> None:
    client = FakeClient([_plan(), _plan(), _worker(), _final()], length_calls={1})
    result = analyze_agentic_audit(client, audit_id="plan-retry", context=_small_context())
    assert "error_code" not in result
    assert client.calls[1]["max_tokens"] > client.calls[0]["max_tokens"]
    assert result["steps"][0]["status"] == "failed"
    assert result["usage"]["total_tokens"] == 1100
    assert [step["sequence_number"] for step in result["steps"]] == [1, 2, 3, 4]


@pytest.mark.parametrize("stage", ["worker", "final"])
def test_truncated_single_transaction_retries_once(stage: str) -> None:
    responses = (
        [_plan(), _worker(), _worker(), _final()] if stage == "worker"
        else [_plan(), _worker(), _final(), _final()]
    )
    client = FakeClient(responses, length_calls={2 if stage == "worker" else 3})
    result = analyze_agentic_audit(client, audit_id=f"{stage}-retry", context=_small_context())
    assert "error_code" not in result
    retry_index = 2 if stage == "worker" else 3
    assert client.calls[retry_index - 1]["max_tokens"] < client.calls[retry_index]["max_tokens"] <= 8192
    assert len(client.calls) == 4
    assert get_audit_progress(f"{stage}-retry")["status"] == "completed"


@pytest.mark.parametrize("stage", ["worker", "final"])
def test_truncated_batch_splits_only_between_transactions(stage: str) -> None:
    responses = [_plan()]
    if stage == "final":
        responses.append(_worker())
    responses += (["{truncated", _worker(), _final(), _worker(), _final()] if stage == "worker"
                  else ["{truncated", _final(), _final()])
    client = FakeClient(responses, length_calls={2 if stage == "worker" else 3})
    result = analyze_agentic_audit(client, audit_id=f"{stage}-split", context=_grouped_context())
    assert "error_code" not in result
    assert result["worker_batch_count"] == 2
    completed_workers = [
        step for step in result["steps"]
        if step["step_kind"] == "evidence_review" and step["status"] == "completed"
    ]
    counts = [step["input_summary"]["candidate_count"] for step in completed_workers]
    assert counts == ([2, 1] if stage == "worker" else [3])
    schemas = [call["schema"] for call in client.calls if call["schema_name"] == "ma_shifu_evidence_review"]
    assert all("decisions" not in schema["properties"] for schema in schemas)
    assert [schema["properties"]["observations"]["maxItems"] for schema in schemas][-2:] == ([2, 1] if stage == "worker" else [3])
    if stage == "final":
        assert result["agent_state"]["stats"]["evidence_reuses"] == 2


def test_failed_single_group_does_not_discard_other_approved_batches(monkeypatch) -> None:
    monkeypatch.setattr("omni_ai_controller.agentic_audit.MAX_WORKER_CHUNK_CANDIDATES", 1)
    approved = {**_final(), "decisions": [{"transaction_id": "tx-1", "recommendation": "suggest"}]}
    client = FakeClient([_plan(), _worker(), approved, "{truncated", "{truncated"], length_calls={4, 5})
    result = analyze_agentic_audit(client, audit_id="partial-batches", context=_grouped_context())
    assert result["error_code"] == "agentic_output_limit"
    assert result["result"]["decisions"] == approved["decisions"]
    assert result["result"]["risks"]
    assert len(client.calls) == 5
    assert result["usage"]["total_tokens"] == 1650
    assert get_audit_progress("partial-batches")["status"] == "partial"


def test_agentic_output_bounds_do_not_mutate_shared_legacy_schema() -> None:
    from omni_ai_controller.agentic_audit import _WORKER_SCHEMA
    from omni_ai_controller.audit_skill import AUDIT_DECISIONS_SCHEMA

    client = FakeClient([_plan(), _worker(), _final()])
    analyze_agentic_audit(client, audit_id="schema-bounds", context=_small_context())
    schema = client.calls[1]["schema"]
    assert schema["properties"]["summary"]["maxLength"] == 320
    assert "decisions" not in schema["properties"]
    assert schema["properties"]["observations"]["maxItems"] == 1
    assert "maxLength" not in _WORKER_SCHEMA["properties"]["summary"]
    assert AUDIT_DECISIONS_SCHEMA["maxItems"] == 200


def test_exhausted_final_retry_never_applies_unapproved_worker_decisions() -> None:
    worker = {**_worker(), "decisions": [{"transaction_id": "tx", "recommendation": "match"}]}
    client = FakeClient([_plan(), worker, _final(), _final()], length_calls={3, 4})
    result = analyze_agentic_audit(client, audit_id="unapproved", context=_small_context())
    assert result["result"]["decisions"] == []
    assert result["error_code"] == "agentic_output_limit"
    assert len(client.calls) == 4


def test_split_budget_exhaustion_is_partial_not_an_unbounded_loop(monkeypatch) -> None:
    monkeypatch.setattr("omni_ai_controller.agentic_audit.MAX_AGENTIC_CHUNKS", 1)
    client = FakeClient([_plan(), "{truncated"], length_calls={2})
    result = analyze_agentic_audit(client, audit_id="split-limit", context=_grouped_context())
    assert result["error_code"] == "agentic_output_limit"
    assert result["worker_batch_count"] == 1
    assert result["result"]["decisions"] == []
    assert len(client.calls) == 2


def test_large_audit_is_planned_as_bounded_two_transaction_tasks():
    context = {
        "transactions": [{"id": f"t{i}"} for i in range(40)],
        "receipts": [{"id": f"r{i}"} for i in range(40)],
        "deterministic_candidates": [{"transaction_id": f"t{i}", "receipt_upload_id": f"r{i}"} for i in range(40)],
    }
    chunks = _worker_chunks(context)
    assert len(chunks) == 20
    assert sum(len(chunk["deterministic_candidates"]) for chunk in chunks) == 40
    assert all(len(chunk["transactions"]) <= 2 for chunk in chunks)


def test_slow_batch_is_split_and_other_transactions_continue():
    from omni_ai_controller.client import ServerRequestTimeout

    class SlowOnce(FakeClient):
        def chat_json(self, messages, **kwargs):
            if len(self.calls) == 1:
                self.calls.append({"schema_name": kwargs["schema_name"]})
                raise ServerRequestTimeout("simulated slow worker")
            return super().chat_json(messages, **kwargs)

    client = SlowOnce([_plan(), _worker(), _final(), _worker(), _final()])
    result = analyze_agentic_audit(client, audit_id="slow-split", context=_grouped_context())
    assert not result.get("error_code")
    assert result["worker_batch_count"] == 2
    assert len([step for step in result["steps"] if step["step_kind"] == "final_assessment" and step["status"] == "completed"]) == 2
    assert any(step.get("error_code") == "agentic_step_timeout" for step in result["steps"])
    assert all(call.get("timeout", 240) <= 240 for call in client.calls)


def test_shared_deadline_does_not_submit_more_model_requests():
    client = FakeClient([_plan(), _worker(), _final()])
    result = analyze_agentic_audit(client, audit_id="expired-deadline", context={**_small_context(), "_deadline": 1})
    assert result["error_code"] == "agentic_iteration_budget"
    assert result["result"]["decisions"] == []
    assert client.calls == []


def test_deadline_retains_completed_final_approvals(monkeypatch):
    from omni_ai_controller import agentic_audit as module

    class Clocked(FakeClient):
        def chat_json(self, messages, **kwargs):
            response = super().chat_json(messages, **kwargs)
            if len(self.calls) == 3:
                monkeypatch.setattr(module, "time", lambda: 10000)
            return response

    decision = {"transaction_id": "tx-1", "receipt_upload_ids": ["receipt-1"], "recommendation": "match", "confidence": .95}
    worker = {**_worker(), "decisions": [decision]}
    final = {**_final(), "decisions": [decision]}
    monkeypatch.setattr(module, "time", lambda: 10)
    monkeypatch.setattr(module, "MAX_WORKER_CHUNK_CANDIDATES", 1)
    client = Clocked([_plan(), worker, final])
    result = analyze_agentic_audit(client, audit_id="deadline-after-approval", context={**_grouped_context(), "_deadline": 200})
    assert result["error_code"] == "agentic_iteration_budget"
    assert result["result"]["decisions"] == [decision]
    assert len(client.calls) == 3


def test_parent_cancellation_preserves_approvals_and_skips_remaining_tasks(monkeypatch):
    from omni_ai_controller import agentic_audit as module

    class CancellingClient(FakeClient):
        def chat_json(self, messages, **kwargs):
            response = super().chat_json(messages, **kwargs)
            if len(self.calls) == 3:
                assert module.cancel_audit_run("cancel-after-approval")
            return response

    decision = {"transaction_id": "tx-1", "receipt_upload_ids": ["receipt-1"], "recommendation": "match", "confidence": .95}
    monkeypatch.setattr(module, "MAX_WORKER_CHUNK_CANDIDATES", 1)
    client = CancellingClient([_plan(), {**_worker(), "decisions": [decision]}, {**_final(), "decisions": [decision]}])
    result = analyze_agentic_audit(client, audit_id="cancel-after-approval", context=_grouped_context())
    assert result["error_code"] == "agentic_cancelled"
    assert result["result"]["decisions"] == [decision]
    assert len(client.calls) == 3


def test_model_prompts_use_short_codes_and_restore_real_ids() -> None:
    transaction_id = "6f06d153-04ce-4ba6-8870-36efce455f8d"
    receipt_id = "f81f6dd0-c0d2-4f58-b7f0-cf00f66ec2b0"
    worker = {
        **_worker(),
        "observations": [{
            "transaction_id": "T001",
            "receipt_upload_ids": ["R001"],
            "group_id": None,
            "finding": "金额一致",
            "evidence": ["精确金额唯一"],
            "unresolved": [],
        }],
    }
    final = {
        **_final(),
        "decisions": [{
            "transaction_id": "T001",
            "receipt_upload_ids": ["R001"],
            "recommendation": "match",
            "confidence": 0.95,
        }],
    }
    context = {
        "transactions": [{"id": transaction_id, "amount": 1200, "direction": "debit"}],
        "receipts": [{"id": receipt_id, "amount": 1200, "type": "expense"}],
        "deterministic_candidates": [{
            "transaction_id": transaction_id,
            "receipt_upload_id": receipt_id,
        }],
    }

    client = FakeClient([_plan(), worker, final])
    result = analyze_agentic_audit(client, audit_id="alias-case", context=context)

    assert result["result"]["decisions"] == [{
        "transaction_id": transaction_id,
        "receipt_upload_ids": [receipt_id],
        "recommendation": "match",
        "confidence": 0.95,
    }]
    planner_payload = str(client.calls[0]["messages"][1]["content"])
    worker_payload = str(client.calls[1]["messages"][1]["content"])
    final_payload = str(client.calls[2]["messages"][1]["content"])
    assert "T001" in planner_payload and "R001" in planner_payload
    assert transaction_id not in worker_payload and receipt_id not in worker_payload
    assert "T001" in worker_payload and "R001" in worker_payload
    assert transaction_id not in final_payload and receipt_id not in final_payload
    assert "T001" in final_payload and "R001" in final_payload


def test_income_match_can_disclose_unitemized_difference_without_blocking_relation() -> None:
    decision = {
        "transaction_id": "tx",
        "receipt_upload_ids": ["receipt"],
        "recommendation": "match",
        "confidence": 0.92,
    }
    worker = {
        **_worker(),
        "observations": [{
            "transaction_id": "tx",
            "receipt_upload_ids": ["receipt"],
            "group_id": None,
            "finding": "交易净额与收入日报毛额存在处理商抽成差额",
            "evidence": ["金额差额稳定", "日期角色一致"],
            "unresolved": [],  # Missing breakdown is disclosure, not a relationship conflict.
        }],
    }
    context = {
        "transactions": [{"id": "tx", "amount": 1170, "direction": "credit"}],
        "receipts": [{"id": "receipt", "amount": 1200, "type": "income"}],
        "deterministic_candidates": [{
            "transaction_id": "tx",
            "receipt_upload_id": "receipt",
            "allocation_role": "income_settlement",
            "category": "income",
        }],
    }

    result = analyze_agentic_audit(
        FakeClient([_plan(), worker, {**_final(), "decisions": [decision]}]),
        audit_id="income-fee-allowed",
        context=context,
    )
    assert result["result"]["decisions"] == [decision]


def test_income_fee_words_do_not_bypass_actual_unresolved_conflicts() -> None:
    decision = {
        "transaction_id": "tx",
        "receipt_upload_ids": ["receipt"],
        "recommendation": "match",
        "confidence": 0.99,
    }
    worker = {
        **_worker(),
        "observations": [{
            "transaction_id": "tx",
            "receipt_upload_ids": ["receipt"],
            "group_id": None,
            "finding": "存在处理商结算抽成差额",
            "evidence": ["金额链路完整"],
            "unresolved": ["手续费可能存在，但结算日期矛盾，币种不符"],
        }],
    }
    context = {
        "transactions": [{"id": "tx"}],
        "receipts": [{"id": "receipt"}],
        "deterministic_candidates": [{
            "transaction_id": "tx",
            "receipt_upload_id": "receipt",
            "allocation_role": "income_settlement",
            "category": "income",
        }],
    }

    result = analyze_agentic_audit(
        FakeClient([_plan(), worker, {**_final(), "decisions": [decision]}]),
        audit_id="income-fee-lower-confidence",
        context=context,
    )
    assert result["result"]["decisions"] == []


def test_unique_income_suggestion_is_not_silently_promoted_or_confidence_invented() -> None:
    worker = {
        **_worker(),
        "observations": [{
            "transaction_id": "tx",
            "receipt_upload_ids": ["receipt"],
            "group_id": None,
            "finding": "收入净额与凭证毛额存在抽成差额",
            "evidence": ["日期角色一致", "候选关系唯一"],
            "unresolved": ["手续费拆分未逐项列示"],
        }],
    }
    final = {
        **_final(),
        "decisions": [{
            "transaction_id": "tx",
            "receipt_upload_ids": ["receipt"],
            "recommendation": "suggest",
            "confidence": 0.86,
        }],
    }
    context = {
        "transactions": [{"id": "tx"}],
        "receipts": [{"id": "receipt"}],
        "deterministic_candidates": [{
            "transaction_id": "tx",
            "receipt_upload_id": "receipt",
            "allocation_role": "income_settlement",
            "category": "income",
        }],
    }

    result = analyze_agentic_audit(
        FakeClient([_plan(), worker, final]),
        audit_id="income-auto-promote",
        context=context,
    )
    assert result["result"]["decisions"] == final["decisions"]
    assert result["result"]["decisions"][0]["recommendation"] == "suggest"
    assert result["result"]["decisions"][0]["confidence"] == 0.86


def test_income_suggest_not_promoted_when_multiple_candidates_exist() -> None:
    worker = {
        **_worker(),
        "observations": [{
            "transaction_id": "tx",
            "receipt_upload_ids": ["receipt-1"],
            "group_id": None,
            "finding": "可解释但并非唯一候选",
            "evidence": ["同日多张收入凭证"],
            "unresolved": ["需在多候选中消歧"],
        }],
    }
    final = {
        **_final(),
        "decisions": [{
            "transaction_id": "tx",
            "receipt_upload_ids": ["receipt-1"],
            "recommendation": "suggest",
            "confidence": 0.91,
        }],
    }
    context = {
        "transactions": [{"id": "tx"}],
        "receipts": [{"id": "receipt-1"}, {"id": "receipt-2"}],
        "deterministic_candidates": [
            {
                "transaction_id": "tx",
                "receipt_upload_id": "receipt-1",
                "allocation_role": "income_settlement",
                "category": "income",
            },
            {
                "transaction_id": "tx",
                "receipt_upload_id": "receipt-2",
                "allocation_role": "income_settlement",
                "category": "income",
            },
        ],
    }

    result = analyze_agentic_audit(
        FakeClient([_plan(), worker, final]),
        audit_id="income-no-auto-promote",
        context=context,
    )
    assert result["result"]["decisions"][0]["recommendation"] == "suggest"


def test_chunk_local_date_order_without_amount_checks_cannot_promote_suggestions() -> None:
    worker = {
        **_worker(),
        "observations": [
            {
                "transaction_id": "tx-1",
                "receipt_upload_ids": ["receipt-1"],
                "group_id": None,
                "finding": "时间顺序与第一笔收入凭证一致，净额有抽成偏差",
                "evidence": ["销售日与到账序列一致"],
                "unresolved": ["含手续费和小费导致差额"],
            },
            {
                "transaction_id": "tx-2",
                "receipt_upload_ids": ["receipt-2"],
                "group_id": None,
                "finding": "时间顺序与第二笔收入凭证一致，净额有抽成偏差",
                "evidence": ["销售日与到账序列一致"],
                "unresolved": ["含手续费和小费导致差额"],
            },
        ],
    }
    final = {
        **_final(),
        "decisions": [
            {
                "transaction_id": "tx-1",
                "receipt_upload_ids": ["receipt-1"],
                "recommendation": "suggest",
                "confidence": 0.84,
            },
            {
                "transaction_id": "tx-2",
                "receipt_upload_ids": ["receipt-2"],
                "recommendation": "suggest",
                "confidence": 0.85,
            },
        ],
    }
    context = {
        "transactions": [
            {"id": "tx-1", "event_date": "2026-09-01"},
            {"id": "tx-2", "event_date": "2026-09-02"},
        ],
        "receipts": [
            {"id": "receipt-1", "event_date": "2026-09-01"},
            {"id": "receipt-2", "event_date": "2026-09-02"},
        ],
        "deterministic_candidates": [
            {
                "transaction_id": "tx-1",
                "receipt_upload_id": "receipt-1",
                "allocation_role": "income_settlement",
                "category": "income",
            },
            {
                "transaction_id": "tx-1",
                "receipt_upload_id": "receipt-2",
                "allocation_role": "income_settlement",
                "category": "income",
            },
            {
                "transaction_id": "tx-2",
                "receipt_upload_id": "receipt-1",
                "allocation_role": "income_settlement",
                "category": "income",
            },
            {
                "transaction_id": "tx-2",
                "receipt_upload_id": "receipt-2",
                "allocation_role": "income_settlement",
                "category": "income",
            },
        ],
    }

    result = analyze_agentic_audit(
        FakeClient([_plan(), worker, final]),
        audit_id="income-time-order-promote",
        context=context,
    )
    recommendations = [item["recommendation"] for item in result["result"]["decisions"]]
    assert recommendations == ["suggest", "suggest"]
    assert result["result"]["decisions"] == final["decisions"]


def test_short_codes_remain_stable_after_confirmed_sources_leave_residual():
    from omni_ai_controller.agentic_audit import _id_aliases

    inventory = {"transactions": [{"id": "first"}, {"id": "second"}],
                 "receipts": [{"id": "first-receipt"}, {"id": "second-receipt"}]}
    initial = _id_aliases({"source_inventory": inventory, **inventory})
    residual = _id_aliases({"source_inventory": inventory,
        "transactions": [{"id": "second"}], "receipts": [{"id": "second-receipt"}]})
    assert initial == residual
    assert residual["tx_forward"]["second"] == "T002"
    assert residual["receipt_forward"]["second-receipt"] == "R002"


def test_short_codes_are_deterministic_without_full_inventory():
    from omni_ai_controller.agentic_audit import _id_aliases

    first = {"transactions": [{"id": "b"}, {"id": "a"}],
             "receipts": [{"id": "y"}, {"id": "x"}]}
    second = {"transactions": first["transactions"][::-1], "receipts": first["receipts"][::-1]}
    assert _id_aliases(first) == _id_aliases(second)


def test_all_agents_use_general_global_income_strategy_not_itemized_fee_requirement():
    from omni_ai_controller.agentic_audit import (
        _PLANNER_SYSTEM_PROMPT, _WORKER_SYSTEM_PROMPT, _FINAL_SYSTEM_PROMPT,
    )

    assert "每条待核流水" in _PLANNER_SYSTEM_PROMPT
    assert "金额附近一对一最优解" in _PLANNER_SYSTEM_PROMPT
    assert "具体构成不必逐项可得" in _WORKER_SYSTEM_PROMPT
    assert "income_sequence" in _WORKER_SYSTEM_PROMPT
    assert "同币种、同方向" in _FINAL_SYSTEM_PROMPT
    assert "不能仅按第几条或本分片顺序硬配" in _FINAL_SYSTEM_PROMPT
    assert "不依赖某一行业、店铺或预设平台费率" in _FINAL_SYSTEM_PROMPT

