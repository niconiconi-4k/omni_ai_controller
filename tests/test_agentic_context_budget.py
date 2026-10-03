import pytest

from omni_ai_controller import agentic_audit as audit
from omni_ai_controller.audit_skill import AuditSkillError


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