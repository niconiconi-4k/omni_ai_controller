from copy import deepcopy
import json
from threading import Event, Lock

import pytest

from omni_ai_controller import agentic_audit as module
from omni_ai_controller.agentic_audit import analyze_agentic_audit, get_audit_progress
from omni_ai_controller.audit_notebooks import AuditNotebooks, MAX_NOTEBOOK_BYTES, serialized
from omni_ai_controller.audit_skill import AuditSkillError
from test_agentic_audit import FakeClient, _final, _plan, _small_context, _worker


def _observation(tx="tx", receipts=None, **updates):
    return {"transaction_id": tx, "receipt_upload_ids": receipts or ["receipt"], "group_id": None,
            "finding": "内核候选金额相等", "evidence": ["候选金额1200及原付款日期"], "unresolved": [], **updates}


def _evidence(*observations):
    return {"observations": list(observations), "task_results": [], "summary": "仅候选事实", "risks": [], "cache_notes": []}


def _decision(tx="tx", receipts=None, **updates):
    return {"transaction_id": tx, "receipt_upload_ids": receipts or ["receipt"], "recommendation": "match", "confidence": .95, **updates}


def _two_context():
    return {"transactions": [{"id": "t1", "amount": -1200}, {"id": "t2", "amount": -900}],
            "receipts": [{"id": "r1", "amount": 1200}, {"id": "r2", "amount": 900}],
            "deterministic_candidates": [{"transaction_id": "t1", "receipt_upload_id": "r1"}, {"transaction_id": "t2", "receipt_upload_id": "r2"}]}


def test_new_wire_has_no_worker_decisions_and_only_li_decides():
    decision = _decision()
    client = FakeClient([_plan(), _evidence(_observation()), {**_final(), "decisions": [decision]}])
    result = analyze_agentic_audit(client, audit_id="wire", context=_small_context())
    assert result["result"]["decisions"] == [decision]
    schema = client.calls[1]["schema"]
    assert "decisions" not in schema["properties"] and "observations" in schema["required"]
    assert "confidence" not in serialized(schema) and "recommendation" not in serialized(schema)
    assert "decisions" not in result["steps"][1]["result"]
    final_prompt = client.calls[2]["messages"][1]["content"]
    assert "kernel_candidates" in final_prompt and "observations" in final_prompt
    assert "notebooks" not in final_prompt
    assert result["agent_state"] == result["progress"]["agent_state"]
    assert result["agent_state"] == get_audit_progress("wire")["agent_state"]


@pytest.mark.parametrize("recommendation", ["match", "suggest", "leave_unmatched"])
@pytest.mark.parametrize("invent", ["transaction", "receipt", "observation"])
def test_all_li_recommendations_fail_closed_on_invented_or_cross_case_relations(recommendation, invent):
    decision = _decision(recommendation=recommendation)
    observation = _observation()
    if invent == "transaction":
        decision["transaction_id"] = "other-case-tx"
    elif invent == "receipt":
        decision["receipt_upload_ids"] = ["other-case-receipt"]
    else:
        observation["receipt_upload_ids"] = ["other-case-receipt"]
    result = analyze_agentic_audit(FakeClient([_plan(), _evidence(observation), {**_final(), "decisions": [decision]}]),
                                   audit_id="invent", context=_small_context())
    assert result["result"]["decisions"] == []


@pytest.mark.parametrize("confidence,unresolved,evidence,expected", [
    (.879, [], ["金额一致"], False), (.88, [], ["金额一致"], True),
    (.99, ["日期矛盾"], ["金额一致"], False), (.99, [], [], False),
])
def test_li_threshold_and_evidence_checks_do_not_need_ma_confidence(confidence, unresolved, evidence, expected):
    decision = _decision(confidence=confidence)
    result = analyze_agentic_audit(FakeClient([_plan(), _evidence(_observation(unresolved=unresolved, evidence=evidence)),
                                             {**_final(), "decisions": [decision]}]), audit_id="threshold", context=_small_context())
    assert result["result"]["decisions"] == ([decision] if expected else [])


def _groups_context():
    return {"transactions": [{"id": "tx"}], "receipts": [{"id": f"r{i}"} for i in range(1, 5)],
            "deterministic_candidates": [{"transaction_id": "tx", "receipt_upload_id": f"r{i}",
                                           "group_id": "A" if i < 3 else "B", "allocation_role": "employee_reimbursement"}
                                          for i in range(1, 5)]}


@pytest.mark.parametrize("receipts,group_id,expected", [
    (["r1", "r2"], "A", True), (["r1"], "A", False), (["r1", "r3"], "A", False),
    (["r1", "r2", "r3", "r4"], "A", False), (["r1", "r2"], None, False),
])
def test_group_observations_and_decisions_must_be_complete_and_not_mixed(receipts, group_id, expected):
    observation = _observation(receipts=receipts, group_id=group_id)
    decision = _decision(receipts=receipts)
    result = analyze_agentic_audit(FakeClient([_plan(), _evidence(observation), {**_final(), "decisions": [decision]}]),
                                   audit_id="groups", context=_groups_context())
    assert result["result"]["decisions"] == ([decision] if expected else [])


def test_no_ungrouped_reimbursement_combination_or_duplicate_receipts():
    context = _two_context()
    context["deterministic_candidates"].append({"transaction_id": "t2", "receipt_upload_id": "r1"})
    first = _decision("t1", ["r1"])
    second = _decision("t2", ["r1"])
    worker = _evidence(_observation("t1", ["r1"]), _observation("t2", ["r1"]))
    result = analyze_agentic_audit(FakeClient([_plan(), worker, {**_final(), "decisions": [first, second]}]), audit_id="duplicate", context=context)
    assert result["result"]["decisions"] == [first]
    context = _small_context()
    context["receipts"].append({"id": "second"})
    context["deterministic_candidates"].append({"transaction_id": "tx", "receipt_upload_id": "second"})
    fabricated_group = _observation(receipts=["receipt", "second"])
    result = analyze_agentic_audit(FakeClient([_plan(), _evidence(fabricated_group), {**_final(), "decisions": [_decision(receipts=["receipt", "second"])]}]), audit_id="no-combine", context=context)
    assert result["result"]["decisions"] == []


def test_scope_priority_dependency_and_legacy_defaults_are_actually_dispatched():
    context = _two_context()
    tasks = [
        {"task_id": "low", "strategy": "direct_expenses", "objective": "低优先级", "prio": 5, "transaction_ids": ["t1"], "depends_on": []},
        {"task_id": "dependent", "strategy": "direct_expenses", "objective": "依赖低任务", "priority": 100,
         "operation": "compare_details", "transaction_ids": ["t1"], "receipt_ids": ["r1"], "amount": None, "depends_on": ["low"]},
        {"task_id": "high", "strategy": "direct_expenses", "objective": "高优先级", "priority": 90,
         "operation": "search_amount", "transaction_ids": ["t2"], "receipt_ids": ["r2"], "amount": 900, "depends_on": []},
    ]
    plan = {**_plan(), "tasks": tasks}
    client = FakeClient([plan, _worker(), _final(), _worker(), _final(), _worker(), _final()])
    result = analyze_agentic_audit(client, audit_id="scope", context=context)
    prompts = [json.loads(call["messages"][1]["content"].split("<audit_data>")[1].split("</audit_data>")[0])
               for call in client.calls if call["schema_name"] == "ma_shifu_evidence_review"]
    assert [prompt["tasks"][0]["parent_task_id"] for prompt in prompts] == ["high", "low", "dependent"]
    assert [prompt["tasks"][0]["transaction_ids"] for prompt in prompts] == [["t2"], ["t1"], ["t1"]]
    assert prompts[1]["tasks"][0]["operation"] == "search_amount"
    assert prompts[2]["tasks"][0]["depends_on"] == ["low"]
    assert "李师傅计划" not in client.calls[1]["messages"][1]["content"]
    assert result["worker_batch_count"] == 3
    assert all(task["status"] == "completed" for task in result["agent_state"]["task_lists"]["audit_planner"])


def test_duplicate_scope_is_removed_and_dependency_cycles_are_blocked():
    task = deepcopy(_plan()["tasks"][0])
    plan = {**_plan(), "tasks": [task, {**task, "task_id": "duplicate"}]}
    result = analyze_agentic_audit(FakeClient([plan, _worker(), _final()]), audit_id="dedup-task", context=_small_context())
    assert len(result["agent_state"]["task_lists"]["evidence_worker"]) == 1
    blocked_plan = {**_plan(), "tasks": [{**task, "depends_on": ["unknown"]}]}
    client = FakeClient([blocked_plan])
    result = analyze_agentic_audit(client, audit_id="blocked", context=_small_context())
    assert result["error_code"] == "agentic_output_limit"
    assert result["agent_state"]["task_lists"]["audit_planner"][0]["status"] == "blocked"
    assert len(client.calls) == 1


def test_confirmed_items_are_excluded_from_both_scope_and_worker_search():
    context = {**_two_context(), "confirmed_receipt_ids": ["r1"], "confirmed_transaction_ids": ["t1"]}
    client = FakeClient([_plan(), _evidence(_observation("t2", ["r2"])), {**_final(), "decisions": [_decision("t1", ["r1"]), _decision("t2", ["r2"])]}])
    result = analyze_agentic_audit(client, audit_id="confirmed", context=context)
    worker_prompt = client.calls[1]["messages"][1]["content"]
    assert '"t1"' not in worker_prompt and '"r1"' not in worker_prompt
    assert result["result"]["decisions"] == [_decision("t2", ["r2"])]


def test_real_event_barrier_proves_worker_prepare_overlaps_li_not_model_calls(monkeypatch):
    monkeypatch.setattr(module, "MAX_WORKER_CHUNK_CANDIDATES", 1)
    worker_started, li_entered, worker_done = Event(), Event(), Event()
    original = AuditNotebooks.prepare
    active_lock = Lock()
    active_workers = 0

    def prepare(self, chunk, tasks, stopped):
        nonlocal active_workers
        with active_lock:
            active_workers += 1
            assert active_workers == 1
        try:
            if chunk["transactions"][0]["id"] == "t2":
                worker_started.set()
                assert li_entered.wait(5), "Li must enter while next worker preparation is active"
            result = original(self, chunk, tasks, stopped)
            if chunk["transactions"][0]["id"] == "t2":
                worker_done.set()
            return result
        finally:
            with active_lock:
                active_workers -= 1

    class BarrierClient(FakeClient):
        active_models = 0

        def chat_json(self, messages, **kwargs):
            self.active_models += 1
            assert self.active_models == 1
            try:
                if kwargs["schema_name"] == "li_shifu_final_assessment" and not li_entered.is_set():
                    assert worker_started.wait(5)
                    li_entered.set()
                    assert worker_done.wait(5)
                return super().chat_json(messages, **kwargs)
            finally:
                self.active_models -= 1

    monkeypatch.setattr(AuditNotebooks, "prepare", prepare)
    client = BarrierClient([_plan(), _worker(), _final(), _worker(), _final()])
    result = analyze_agentic_audit(client, audit_id="barrier", context=_two_context())
    assert worker_started.is_set() and worker_done.is_set()
    assert result["worker_batch_count"] == 2
    assert [step["sequence_number"] for step in result["steps"]] == [1, 2, 3, 4, 5]


def test_final_split_reuses_observations_and_never_reruns_worker():
    context = _two_context()
    first, second = _decision("t1", ["r1"]), _decision("t2", ["r2"])
    client = FakeClient([_plan(), _evidence(_observation("t1", ["r1"]), _observation("t2", ["r2"])), "{truncated",
                         {**_final(), "decisions": [first]}, {**_final(), "decisions": [second]}], length_calls={3})
    result = analyze_agentic_audit(client, audit_id="reuse", context=context)
    assert result["result"]["decisions"] == [first, second]
    assert [call["schema_name"] for call in client.calls].count("ma_shifu_evidence_review") == 1
    assert result["agent_state"]["stats"]["evidence_reuses"] == 2
    for call in client.calls[3:]:
        payload = json.loads(call["messages"][1]["content"].split("<agent_results>")[1].split("</agent_results>")[0])
        assert len(payload["worker_result"]["observations"]) == 1


@pytest.mark.parametrize("stop", ["cancel", "deadline"])
def test_stop_preserves_completed_future_notebooks_and_only_li_approvals(monkeypatch, stop):
    monkeypatch.setattr(module, "MAX_WORKER_CHUNK_CANDIDATES", 1)
    prepared = Event()
    original = AuditNotebooks.prepare
    monkeypatch.setattr(module, "time", lambda: 10)

    def prepare(self, chunk, tasks, stopped):
        result = original(self, chunk, tasks, stopped)
        if chunk["transactions"][0]["id"] == "t2":
            prepared.set()
        return result

    class StoppingClient(FakeClient):
        def chat_json(self, messages, **kwargs):
            response = super().chat_json(messages, **kwargs)
            if kwargs["schema_name"] == "li_shifu_final_assessment":
                assert prepared.wait(5)
                if stop == "cancel":
                    assert module.cancel_audit_run("stopping")
                else:
                    monkeypatch.setattr(module, "time", lambda: 10000)
            return response

    monkeypatch.setattr(AuditNotebooks, "prepare", prepare)
    decision = _decision("t1", ["r1"])
    context = {**_two_context(), "_deadline": 200}
    client = StoppingClient([_plan(), _evidence(_observation("t1", ["r1"])), {**_final(), "decisions": [decision]}])
    result = analyze_agentic_audit(client, audit_id="stopping", context=context)
    assert len(client.calls) == 3 and result["result"]["decisions"] == [decision]
    assert result["error_code"] == ("agentic_cancelled" if stop == "cancel" else "agentic_iteration_budget")
    assert any(entry["key"] == "receipt:r2:basic" for entry in result["agent_state"]["notebooks"]["evidence_worker"]["entries"])
    assert context["_agent_state"] == result["agent_state"]
    assert get_audit_progress("stopping")["agent_state"] == result["agent_state"]


def test_state_resumes_cache_outside_4m_source_budget_but_is_never_model_input():
    context = _small_context()
    notebooks = AuditNotebooks("memory", "memory")
    notebooks.put("audit_planner", "large-cache", "hypothesis", "记" * 1500000)
    context["_agent_state"] = notebooks.snapshot()
    first = FakeClient([_plan(), _worker(), _final()])
    result = analyze_agentic_audit(first, audit_id="memory", context=context)
    assert result["context_characters"] < 1000
    assert all("large-cache" not in serialized(call["messages"]) and "notebooks" not in serialized(call["messages"]) for call in first.calls)
    second = analyze_agentic_audit(FakeClient([_plan(), _worker(), _final()]), audit_id="memory", context=context)
    assert second["agent_state"]["stats"]["cache_hits"] >= 2
    assert all(book["used_bytes"] <= MAX_NOTEBOOK_BYTES for book in result["agent_state"]["notebooks"].values())


def test_invalid_model_response_preserves_real_tasks_and_notebook_progress():
    context = _small_context()
    with pytest.raises(AuditSkillError):
        analyze_agentic_audit(FakeClient([_plan(), "not-json"]), audit_id="bad-worker", context=context)
    state = get_audit_progress("bad-worker")["agent_state"]
    assert state == context["_agent_state"]
    assert state["notebooks"]["evidence_worker"]["entries"]
    assert any(task["status"] == "failed" for task in state["task_lists"]["evidence_worker"])


def test_current_run_confirmation_excludes_later_planned_scope_without_rerunning_ma():
    task = deepcopy(_plan()["tasks"][0])
    plan = {**_plan(), "tasks": [task, {**task, "task_id": "later-details", "operation": "compare_details",
                                      "priority": 1, "depends_on": [task["task_id"]]}]}
    decision = _decision()
    client = FakeClient([plan, _evidence(_observation()), {**_final(), "decisions": [decision]}])
    result = analyze_agentic_audit(client, audit_id="current-confirmed", context=_small_context())
    assert len(client.calls) == 3 and result["result"]["decisions"] == [decision]
    tasks = result["agent_state"]["task_lists"]["evidence_worker"]
    assert tasks[-1]["exclusion_reason"] == "already_confirmed_by_planner"
    assert all(task["status"] == "completed" for task in tasks)


def test_every_task_transition_is_published_as_real_agent_state(monkeypatch):
    events = []
    original = module._publish_progress

    def publish(run_id, **updates):
        if "agent_state" in updates:
            events.append(deepcopy(updates["agent_state"]))
        original(run_id, **updates)

    monkeypatch.setattr(module, "_publish_progress", publish)
    analyze_agentic_audit(FakeClient([_plan(), _evidence(_observation()), _final()]), audit_id="events", context=_small_context())
    states = [state["task_lists"]["evidence_worker"][0]["status"] for state in events if state["task_lists"]["evidence_worker"]]
    transitions = [value for index, value in enumerate(states) if index == 0 or states[index - 1] != value]
    assert transitions == ["pending", "preparing", "prepared", "running", "completed"]
    assert all(state["version"] == 1 and state["audit_id"] == "events" for state in events)


def test_income_adjustment_does_not_require_explicit_fee_and_role_prompts_remain_conservative():
    context = _small_context()
    context["deterministic_candidates"][0].update(allocation_role="revenue_settlement", amount_delta="-30", company_support=True)
    worker = _evidence(_observation(finding="销售1200，净到账1170，差额30仅推断扣费", evidence=["内核差额及公司/日期支持，无明确费用标签"]))
    decision = _decision(kind="revenue_settlement", discrepancy_note="差额30，扣费只是推断")
    client = FakeClient([_plan(), worker, {**_final(), "decisions": [decision]}])
    result = analyze_agentic_audit(client, audit_id="net-income", context=context)
    assert result["result"]["decisions"] == [decision]
    assert "无须票面明确" in client.calls[1]["messages"][0]["content"]
    assert "工资差额只能 suspected" in client.calls[1]["messages"][0]["content"]
    assert "最后保留异常" in client.calls[1]["messages"][0]["content"]


def test_partially_confirmed_or_missing_receipt_group_cannot_become_partial_match():
    context = _groups_context()
    context["confirmed_receipt_ids"] = ["r1"]
    client = FakeClient([_plan(), _evidence(_observation(receipts=["r2"], group_id="A")), {**_final(), "decisions": [_decision(receipts=["r2"])]}])
    result = analyze_agentic_audit(client, audit_id="partial-group", context=context)
    assert result["result"]["decisions"] == []
    prompt = client.calls[1]["messages"][1]["content"]
    assert '"r1"' not in prompt and '"r2"' not in prompt
    context = _groups_context()
    context["receipts"] = [receipt for receipt in context["receipts"] if receipt["id"] != "r1"]
    client = FakeClient([_plan(), _worker(), _final()])
    analyze_agentic_audit(client, audit_id="missing-group", context=context)
    assert '"r2"' not in client.calls[1]["messages"][1]["content"]


def test_model_receives_only_selected_basic_or_detail_projection_not_duplicate_raw_source():
    context = _small_context()
    context["transactions"][0].update(amount=-1200, counterparty="bank-party")
    context["receipts"][0].update(amount=1200, party="receipt-party", ocr_excerpt="private-detail-marker")
    client = FakeClient([_plan(), _worker(), _final()])
    result = analyze_agentic_audit(client, audit_id="basic-projection", context=context)
    prompt = client.calls[1]["messages"][1]["content"]
    assert "private-detail-marker" not in prompt and "receipt-party" not in prompt and "bank-party" not in prompt
    assert result["agent_state"]["stats"]["exact_amount_hits"] == 1
    plan = {**_plan(), "tasks": [{**_plan()["tasks"][0], "operation": "compare_details"}]}
    client = FakeClient([plan, _worker(), _final()])
    analyze_agentic_audit(client, audit_id="details-projection", context=context)
    prompt = client.calls[1]["messages"][1]["content"]
    assert prompt.count("private-detail-marker") == 1
    assert prompt.count("receipt-party") == 1 and prompt.count("bank-party") == 1


def test_large_scope_catalog_is_bounded_without_reintroducing_64_limit():
    context = {"transactions": [{"id": f"t{i}"} for i in range(200)],
               "receipts": [{"id": f"r{i}"} for i in range(200)],
               "deterministic_candidates": [{"transaction_id": f"t{i}", "receipt_upload_id": f"r{i}"} for i in range(200)]}
    client = FakeClient([_plan(), _plan()], length_calls={1, 2})
    result = analyze_agentic_audit(client, audit_id="large-preview", context=context)
    assert result["error_code"] == "agentic_output_limit"
    assert result["worker_batch_count"] == 100
    prompt = client.calls[0]["messages"][1]["content"]
    summary = json.loads(prompt.split("<audit_summary>")[1].split("</audit_summary>")[0])
    assert summary["counts"]["deterministic_candidates"] == 200
    assert summary["scope_catalog_omitted"] > 0
    assert module._estimated_tokens(prompt) + module._estimated_tokens(client.calls[0]["messages"][0]["content"]) < module.MAX_PLANNER_INPUT_TOKENS


def test_unexpected_prepare_failure_still_preserves_state_and_fails_closed(monkeypatch):
    original = AuditNotebooks.prepare

    def failed_prepare(self, chunk, tasks, stopped):
        original(self, chunk, tasks, stopped)
        raise RuntimeError("fake CPU failure")

    monkeypatch.setattr(AuditNotebooks, "prepare", failed_prepare)
    context = _small_context()
    with pytest.raises(AuditSkillError, match="保全双笔记"):
        analyze_agentic_audit(FakeClient([_plan()]), audit_id="prepare-failed", context=context)
    snapshot = get_audit_progress("prepare-failed")
    assert snapshot["status"] == "failed"
    assert snapshot["agent_state"] == context["_agent_state"]
    assert snapshot["agent_state"]["notebooks"]["evidence_worker"]["entries"]


def test_legacy_suggestion_is_read_only_evidence_not_an_automatic_match():
    old = _decision(recommendation="suggest", confidence=.7)
    worker = {**_worker(), "decisions": [old]}
    suggestion = _decision(recommendation="suggest")
    client = FakeClient([_plan(), worker, {**_final(), "decisions": [suggestion]}])
    result = analyze_agentic_audit(client, audit_id="legacy-evidence", context=_small_context())
    assert result["result"]["decisions"] == [suggestion]
    assert "decisions" not in result["steps"][1]["result"]
    assert result["steps"][1]["result"]["observations"][0]["unresolved"]
    result = analyze_agentic_audit(FakeClient([_plan(), worker, {**_final(), "decisions": [_decision()]}]),
                                   audit_id="legacy-no-promote", context=_small_context())
    assert result["result"]["decisions"] == []


def test_full_kernel_group_is_not_accidentally_capped_by_six_text_items():
    ids = [f"r{i}" for i in range(7)]
    context = {"transactions": [{"id": "tx"}], "receipts": [{"id": value} for value in ids],
               "deterministic_candidates": [{"transaction_id": "tx", "receipt_upload_id": value, "group_id": "batch-7"} for value in ids]}
    decision = _decision(receipts=ids)
    client = FakeClient([_plan(), _evidence(_observation(receipts=ids, group_id="batch-7")), {**_final(), "decisions": [decision]}])
    result = analyze_agentic_audit(client, audit_id="whole-group", context=context)
    assert result["result"]["decisions"] == [decision]
    assert result["worker_batch_count"] == 1
    assert client.calls[1]["schema"]["properties"]["observations"]["items"]["properties"]["receipt_upload_ids"]["maxItems"] == 12
    assert client.calls[2]["schema"]["properties"]["decisions"]["items"]["properties"]["receipt_upload_ids"]["maxItems"] == 12