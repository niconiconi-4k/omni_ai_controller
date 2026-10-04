from copy import deepcopy
import json
from uuid import UUID as UUIDValue

import pytest

from omni_ai_controller import agentic_audit as agent
from omni_ai_controller.audit_inventory import canonical_bytes, fingerprint, register_inventory
from omni_ai_controller.audit_notebooks import AuditNotebooks
from omni_ai_controller.audit_references import HASH, SHORT, UUID, ShortReferenceCodec
from omni_ai_controller.audit_skill import analyze_audit
from test_agentic_audit import FakeClient, _final, _plan, _worker
from test_audit_skill import FakeClient as LegacyClient
from test_cashflow_channel_agents import channel_context, evidence_and_decisions


def identity(number):
    return str(UUIDValue(int=number))


def assert_private(state):
    display = {key: value for key, value in state.items() if key != "identity_map"}
    document = json.dumps(display, ensure_ascii=False)
    assert UUID.search(document) is None
    assert HASH.search(document) is None
    mapping = state["identity_map"]
    assert all(mapping["reverse"][token] == raw for raw, token in mapping["forward"].items())
    for book in state["notebooks"].values():
        assert book["used_bytes"] == len(canonical_bytes(book)) <= book["limit_bytes"]
        assert all(SHORT.fullmatch(entry["key"]) for entry in book["entries"])
        assert all(not entry["source_fingerprint"] or SHORT.fullmatch(entry["source_fingerprint"]) for entry in book["entries"])
    assert all(SHORT.fullmatch(task["task_id"]) for tasks in state["task_lists"].values() for task in tasks)


def inventory():
    return {"transactions": [{"id": identity(1), "amount": "1200.00", "direction": "debit",
                               "currency": "SEK", "date": "2026-10-02"},
                              {"id": identity(2), "amount": "900.00", "direction": "debit"}],
            "receipts": [{"id": identity(3) + ":1", "upload_id": identity(3), "amount": "1200.00",
                          "type": "expense", "currency": "SEK", "payment_date": "2026-10-01",
                          "party": "receipt-party", "ocr_excerpt": "source=" + identity(4)},
                         {"id": identity(5) + ":1", "amount": "900.00", "type": "expense"}]}


def test_codec_nested_keys_prose_compound_uploads_and_financial_values_roundtrip():
    codec = ShortReferenceCodec()
    sources = inventory()
    codec.seed({"source_inventory": sources})
    value = {"group_id": identity(6), "nested": {identity(4): {"upload_id": identity(3),
             "receipt_ids": [sources["receipts"][0]["id"]]}},
             "prose": "证据 " + identity(3) + ":1；上传 " + identity(3),
             "amount": "1200.00", "date": "2026-10-01", "party": "receipt-party"}
    encoded = codec.encode(value)
    assert UUID.search(json.dumps(encoded)) is None
    assert encoded["amount"] == "1200.00" and encoded["date"] == "2026-10-01"
    assert encoded["party"] == "receipt-party"
    assert codec.decode(encoded) == value
    assert codec.identity_map["forward"][sources["receipts"][0]["id"]] == "R001"
    assert codec.identity_map["forward"][identity(3)].startswith("U")


def test_incremental_restore_never_renumbers_full_inventory_or_residual():
    sources = inventory()
    codec = ShortReferenceCodec()
    codec.seed({"source_inventory": sources})
    before = deepcopy(codec.identity_map["forward"])
    restored = ShortReferenceCodec(codec.identity_map)
    reordered = {plural: rows[::-1] for plural, rows in sources.items()}
    restored.seed({"source_inventory": reordered, "transactions": sources["transactions"][1:],
                   "receipts": sources["receipts"][1:]})
    restored.seed({"transactions": [{"id": identity(9)}]})
    assert all(restored.identity_map["forward"][raw] == token for raw, token in before.items())
    assert restored.reference(identity(9), "T") == "T003"
    aliases = agent._id_aliases({"source_inventory": reordered}, restored)
    assert aliases["tx_forward"][identity(1)] == "T001"


def test_existing_short_ids_and_numeric_task_ids_preserve_money_dates():
    codec = ShortReferenceCodec()
    codec.seed({"transactions": [{"id": "T001"}], "receipts": [{"id": "R001"}]})
    codec.reference("1200", "K")
    codec.reference("2026-10-01", "K")
    assert codec.text("1200 / 2026-10-01 / T001 / R001") == "1200 / 2026-10-01 / T001 / R001"
    assert codec.model_result({"transaction_id": "T001"}) == {"transaction_id": "T001"}


def test_short_namespace_grows_beyond_three_digits():
    codec = ShortReferenceCodec()
    for number in range(1, 1002):
        assert codec.reference(identity(number), "T") == f"T{number:03d}"
    assert ShortReferenceCodec(codec.identity_map).reference(identity(1100), "T") == "T1002"


def test_literal_legacy_short_id_cannot_collide_with_another_real_source():
    codec = ShortReferenceCodec()
    codec.seed({"transactions": [{"id": identity(1)}, {"id": "T001"}]})
    assert codec.reference("T001", "T") == "T001"
    assert codec.reference(identity(1), "T") == "T002"
    prior = ShortReferenceCodec()
    prior.seed({"transactions": [{"id": identity(1)}]})
    with pytest.raises(ValueError, match="冲突"):
        prior.seed({"transactions": [{"id": "T001"}]})


def test_non_uuid_id_words_do_not_rewrite_financial_evidence():
    codec = ShortReferenceCodec()
    value = {"group_id": "income", "task_id": "payment", "amount": "1200.00",
             "finding": "income payment amount 1200.00 on 2026-10-01"}
    projected = codec.encode(value)
    assert projected["group_id"] == "G001" and projected["task_id"] == "K001"
    assert projected["finding"] == value["finding"]


def test_financial_reference_literals_are_lossless_across_cache_and_restore():
    sources = inventory()
    receipt = {**sources["receipts"][0], "reference_numbers": ["T001", "R001", "~R001", "~~"],
               "memo": "reference R001; see " + sources["transactions"][0]["id"]}
    books = AuditNotebooks(identity(20), identity(21), source_inventory=sources)
    first = books.facts(receipt, details=True)
    second = books.facts(receipt, details=True)
    restored = AuditNotebooks(identity(20), identity(21), books.snapshot())
    third = restored.facts(receipt, details=True)
    assert first == second == third
    assert third["reference_numbers"] == receipt["reference_numbers"]
    assert third["memo"] == receipt["memo"]
    projected = restored.codec.encode(third)
    assert projected["reference_numbers"] == receipt["reference_numbers"]
    assert_private(restored.snapshot())


def test_without_inventory_candidate_order_only_changes_display_not_mapping():
    candidates = [{"transaction_id": identity(2), "receipt_upload_id": identity(4)},
                  {"transaction_id": identity(1), "receipt_upload_id": identity(3)}]
    first, second = ShortReferenceCodec(), ShortReferenceCodec()
    first.seed({"deterministic_candidates": candidates})
    second.seed({"deterministic_candidates": candidates[::-1]})
    assert first.identity_map == second.identity_map


@pytest.mark.parametrize("fault", ["reverse", "duplicate", "malformed", "version"])
def test_corrupt_maps_fail_closed(fault):
    codec = ShortReferenceCodec()
    codec.reference(identity(1), "T")
    mapping = deepcopy(codec.identity_map)
    if fault == "reverse":
        mapping["reverse"]["T001"] = identity(2)
    elif fault == "duplicate":
        mapping["forward"][identity(2)] = "T001"
    elif fault == "malformed":
        mapping["forward"] = []
    else:
        mapping["version"] = 99
    with pytest.raises(ValueError):
        ShortReferenceCodec(mapping)


def test_v1_migration_keeps_full_evidence_cache_hits_and_invalidates_both_projections():
    receipt = inventory()["receipts"][0]
    original = AuditNotebooks(identity(20), identity(21))
    original.facts(receipt)
    details = original.facts(receipt, details=True)
    original.task("audit_planner", {"task_id": identity(22), "objective": "复核 " + receipt["id"],
                                    "receipt_ids": [receipt["id"]], "depends_on": []}, "completed")
    legacy = {**original.snapshot(), **original.internal_snapshot(), "version": 1,
              "audit_id": identity(20), "run_id": identity(21)}
    legacy.pop("identity_map")
    legacy.pop("reference_schema")
    frozen = deepcopy(legacy)
    resumed = AuditNotebooks(identity(20), identity(21), legacy)
    assert_private(resumed.snapshot())
    assert resumed.facts(receipt, details=True) == details
    assert resumed.snapshot()["stats"]["cache_hits"] == 1
    changed = {**receipt, "amount": "1199.00", "party": "new party"}
    assert resumed.facts(changed)["amount"] == "1199.00"
    assert resumed.snapshot()["stats"]["invalidations"] == 1
    raw_entries = resumed.internal_snapshot()["notebooks"]["evidence_worker"]["entries"]
    assert not any(entry["key"].endswith(":details") for entry in raw_entries)
    assert resumed.facts(changed, details=True)["party"] == "new party"
    assert legacy == frozen
    assert_private(resumed.snapshot())


def test_inventory_register_restore_exact_fingerprint_and_cache_invalidation():
    sources = inventory()
    books = AuditNotebooks(identity(20), identity(21))
    books.organize_inventory(sources)
    state = books.snapshot()
    assert_private(state)
    before = deepcopy(state["identity_map"]["forward"])
    resumed = AuditNotebooks(identity(20), identity(21), state)
    resumed.organize_inventory(sources)
    assert resumed.snapshot()["stats"]["cache_hits"] == 4
    assert resumed.facts(sources["receipts"][0], details=True)["ocr_excerpt"].endswith(identity(4))
    changed = deepcopy(sources)
    changed["receipts"][0]["amount"] = "1199.99"
    resumed.organize_inventory(changed)
    assert resumed.snapshot()["stats"]["invalidations"] == 1
    assert resumed.facts(changed["receipts"][0])["amount"] == "1199.99"
    assert all(resumed.codec.identity_map["forward"][raw] == token for raw, token in before.items())
    assert_private(resumed.snapshot())
    # Same-audit new run retains labels, not run-bound cached evidence.
    new_run = AuditNotebooks(identity(20), identity(25), resumed.snapshot())
    assert new_run.codec.reference(identity(1), "T") == "T001"
    assert not new_run.snapshot()["notebooks"]["evidence_worker"]["entries"]
    direct = deepcopy(state)
    register_inventory(direct, sources)
    assert_private(direct)


def test_put_and_prepare_short_keys_search_ids_hashes_but_internal_real_sources():
    sources = inventory()
    books = AuditNotebooks(identity(20), identity(21))
    books.organize_inventory(sources)
    task = {"task_id": identity(22), "operation": "compare_details", "amount": "1200.00",
            "receipt_ids": [item["id"] for item in sources["receipts"]],
            "transaction_ids": [item["id"] for item in sources["transactions"]], "depends_on": []}
    prepared = books.prepare(sources, [task])
    assert prepared["amount_searches"][0]["receipt_ids"] == [sources["receipts"][0]["id"]]
    books.put("audit_planner", "scope:" + identity(8), "cache_notes", {identity(7): "原文 " + identity(8)}, fingerprint(sources))
    state = books.snapshot()
    assert_private(state)
    assert any(entry["key"].startswith("Q") for entry in state["notebooks"]["evidence_worker"]["entries"])
    assert books.task_statuses("evidence_worker")[identity(22)] == "prepared"


def test_all_agent_prompts_short_nested_channel_groups_restore_real_approval():
    context = channel_context(2, group=identity(6), receipt=identity(3) + ":1", prefix=identity(1))
    for index, row in enumerate(context["transactions"]):
        row["id"] = identity(index + 10)
        context["deterministic_candidates"][index]["transaction_id"] = row["id"]
    tx_ids = [row["id"] for row in context["transactions"]]
    for candidate in context["deterministic_candidates"]:
        candidate["evidence"]["group_transaction_ids"] = tx_ids
        candidate["evidence"]["alternative_groups"][0]["transaction_ids"] = tx_ids
        candidate["evidence"]["scope_by_id"] = {identity(7): {"source_id": identity(8)}}
    context["source_inventory"] = deepcopy({key: context[key] for key in ("transactions", "receipts")})
    context["_run_id"] = identity(21)
    worker, final = evidence_and_decisions(context)
    worker["summary"] = "证据 " + identity(7)
    worker["observations"][0]["finding"] += " " + identity(6)
    model_codec = ShortReferenceCodec()
    model_codec.seed(context)
    plan = _plan()
    plan["tasks"][0].update(strategy="revenue_settlements", operation="compare_details",
                            transaction_ids=[], receipt_ids=[], amount=None, depends_on=[])
    client = FakeClient([plan, model_codec.encode(worker), model_codec.encode(final)])
    result = agent.analyze_agentic_audit(client, audit_id=identity(20), context=context)
    for call in client.calls:
        assert UUID.search(json.dumps(call["messages"])) is None
        assert HASH.search(json.dumps(call["messages"])) is None
    decisions = result["result"]["decisions"]
    assert len(decisions) == 2 and all(item["recommendation"] == "match" for item in decisions)
    assert {item["transaction_id"] for item in decisions} == set(tx_ids)
    assert all(item["receipt_upload_ids"] == [identity(3) + ":1"] and item["confidence"] == .96 for item in decisions)
    observations = next(step["result"]["observations"] for step in result["steps"] if step["step_kind"] == "evidence_review")
    assert all(item["group_id"] == identity(6) for item in observations)
    assert_private(result["agent_state"])


@pytest.mark.parametrize("field,token", [("transaction_id", "T999"), ("receipt_upload_ids", "R999"), ("group_id", "G999")])
def test_unknown_model_references_cannot_approve(field, token):
    context = channel_context(2, group=identity(6), receipt=identity(3) + ":1")
    worker, final = evidence_and_decisions(context)
    codec = ShortReferenceCodec()
    codec.seed(context)
    worker = codec.encode(worker)
    worker["observations"][0][field] = [token] if field == "receipt_upload_ids" else token
    plan = _plan()
    plan["tasks"][0].update(strategy="revenue_settlements", transaction_ids=[], receipt_ids=[])
    result = agent.analyze_agentic_audit(FakeClient([plan, worker, codec.encode(final)]), audit_id=identity(20), context=context)
    assert result["result"]["decisions"] == []


def test_legacy_model_input_short_output_real_and_unknown_failclosed():
    sources = inventory()
    context = {**sources, "source_inventory": deepcopy(sources), "group_id": identity(6),
               "notes": "流水 " + identity(1), "nested": {identity(8): {"source_id": identity(9)}}}
    client = LegacyClient(json.dumps({"decisions": [
        {"transaction_id": "T001", "receipt_upload_ids": ["R001"], "confidence": .91},
        {"transaction_id": "T999", "receipt_upload_ids": ["R999"], "confidence": .99}], "summary": "完成", "risks": []}))
    result = analyze_audit(client, audit_id=identity(20), context=context)
    assert UUID.search(json.dumps(client.calls[0]["messages"])) is None
    assert result["result"]["decisions"][0]["transaction_id"] == identity(1)
    assert result["result"]["decisions"][0]["receipt_upload_ids"] == [identity(3) + ":1"]
    assert result["result"]["decisions"][0]["confidence"] == .91
    assert result["result"]["decisions"][1]["transaction_id"] == "unknown-reference:T999"


def test_two_real_audit_rounds_reuse_inventory_cache_and_keep_residual_codes():
    sources = inventory()
    previous = None
    states = []
    for round_number in (1, 2):
        index = round_number - 1
        transaction, receipt = sources["transactions"][index], sources["receipts"][index]
        context = {"source_inventory": deepcopy(sources), "transactions": [transaction], "receipts": [receipt],
                   "deterministic_candidates": [{"transaction_id": transaction["id"], "receipt_upload_id": receipt["id"]}],
                   "_run_id": identity(21), "iteration": {"number": round_number}}
        if previous is not None:
            context.update(_agent_state=previous, confirmed_transaction_ids=[sources["transactions"][0]["id"]],
                           confirmed_receipt_ids=[sources["receipts"][0]["id"]])
        plan = _plan()
        plan["tasks"][0].update(strategy="direct_expenses", transaction_ids=[], receipt_ids=[], depends_on=[])
        tx_code, receipt_code = f"T{round_number:03d}", f"R{round_number:03d}"
        worker = {**_worker(), "observations": [{"transaction_id": tx_code, "receipt_upload_ids": [receipt_code],
                  "group_id": None, "finding": "amount/date/currency agree", "evidence": ["kernel facts"], "unresolved": []}]}
        final = {**_final(), "decisions": [{"transaction_id": tx_code, "receipt_upload_ids": [receipt_code],
                 "recommendation": "suggest", "confidence": .91}]}
        client = FakeClient([plan, worker, final])
        result = agent.analyze_agentic_audit(client, audit_id=identity(20), context=context)
        assert result["result"]["decisions"][0]["transaction_id"] == transaction["id"]
        assert result["result"]["decisions"][0]["receipt_upload_ids"] == [receipt["id"]]
        worker_prompt = next(call["messages"][1]["content"] for call in client.calls if call["schema_name"] == "ma_shifu_evidence_review")
        payload = json.loads(worker_prompt.split("<audit_data>")[1].split("</audit_data>")[0])
        assert payload["transactions"][0]["id"] == tx_code
        assert payload["receipts"][0]["id"] == receipt_code
        assert UUID.search(json.dumps([call["messages"] for call in client.calls])) is None
        previous = result["agent_state"]
        assert_private(previous)
        states.append(previous)
    assert states[1]["stats"]["cache_hits"] > states[0]["stats"]["cache_hits"]
    assert all(states[1]["identity_map"]["forward"][raw] == token for raw, token in states[0]["identity_map"]["forward"].items())