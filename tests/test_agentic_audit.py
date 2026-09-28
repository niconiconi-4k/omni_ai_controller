import json
from types import SimpleNamespace

import pytest

from omni_ai_controller.agentic_audit import (
    MAX_AGENTIC_SOURCE_CHARS,
    SEED_PLAYBOOK_ID,
    analyze_agentic_audit,
)
from omni_ai_controller.audit_skill import AuditSkillError


class FakeClient:
    def __init__(self, responses: list[dict[str, object] | str]) -> None:
        self.responses = list(responses)
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
    client = FakeClient([_plan(), _worker(), _final(), _worker(), _final()])
    long_text = "凭证内容" * 1800
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
