import json
from types import SimpleNamespace

import pytest

from omni_ai_controller.audit_skill import (
    AUDIT_SKILL_SYSTEM_PROMPT,
    MAX_AUDIT_CONTEXT_CHARS,
    AuditSkillError,
    analyze_audit,
)


class FakeClient:
    def __init__(self, content: object) -> None:
        self.content = content
        self.config = SimpleNamespace(model_name="local-qwen")
        self.calls: list[dict[str, object]] = []

    def chat_json(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append({"messages": messages, **kwargs})
        return SimpleNamespace(
            content=self.content,
            raw={"id": "audit-1", "usage": {"total_tokens": 42}},
        )


def valid_result() -> dict[str, object]:
    return {
        "decisions": [],
        "summary": "无待确认候选",
        "risks": [],
    }


def test_audit_skill_uses_strict_schema_and_disables_unbounded_output() -> None:
    client = FakeClient(json.dumps(valid_result(), ensure_ascii=False))

    result = analyze_audit(client, audit_id="case-1", context={"transactions": []})

    assert result["model"] == "local-qwen"
    assert result["serialized"] is True
    assert result["usage"]["total_tokens"] == 42
    call = client.calls[0]
    assert call["schema_name"] == "local_audit_reconciliation"
    assert call["max_tokens"] == 8192
    assert call["schema"]["additionalProperties"] is False
    assert "不得虚构" in AUDIT_SKILL_SYSTEM_PROMPT
    assert "不要自行重新计算金额" in AUDIT_SKILL_SYSTEM_PROMPT


def test_audit_skill_rejects_oversized_context_before_model_call() -> None:
    client = FakeClient(json.dumps(valid_result()))

    with pytest.raises(AuditSkillError, match="超过本地模型安全预算"):
        analyze_audit(
            client,
            audit_id="case-1",
            context={"ocr": "x" * (MAX_AUDIT_CONTEXT_CHARS + 1)},
        )

    assert client.calls == []


def test_audit_skill_rejects_non_json_model_response() -> None:
    client = FakeClient("not-json")

    with pytest.raises(AuditSkillError, match="返回格式无效"):
        analyze_audit(client, audit_id="case-1", context={})
