from copy import deepcopy

import pytest

from omni_ai_controller import agentic_audit as agent
from omni_ai_controller.audit_notebooks import AuditNotebooks
from omni_ai_controller.audit_references import ShortReferenceCodec


@pytest.mark.parametrize("field,prefix", [("group_id", "G"), ("task_id", "K"), ("key", "B")])
def test_raw_collision_rejected_but_existing_encoded_projection_idempotent(field, prefix):
    codec = ShortReferenceCodec()
    original = {field: "raw-original"}
    encoded = codec.encode(original, raw=True)
    token = encoded[field]
    saved = deepcopy(codec.identity_map)
    for restored in (codec, ShortReferenceCodec(saved)):
        assert restored.encode(encoded) == encoded
        assert restored.reference(token, prefix, encoded=True) == token
        assert restored.decode(encoded, text=False) == original
        with pytest.raises(ValueError, match="冲突"):
            restored.reference(token, prefix, encoded=False)
        with pytest.raises(ValueError, match="冲突"):
            restored.encode(encoded, raw=True)
        with pytest.raises(ValueError, match="冲突"):
            restored.encode(encoded, storage=True)
        with pytest.raises(ValueError, match="冲突"):
            restored.seed({"nested": encoded})
        assert restored.identity_map == saved


@pytest.mark.parametrize("field,prefix", [("group_id", "G"), ("task_id", "K"), ("key", "B")])
def test_seed_reserves_nested_literal_before_allocating_same_namespace(field, prefix):
    codec = ShortReferenceCodec()
    token = prefix + "001"
    codec.seed({"rows": [{field: "raw-original"}, {field: token}]})
    assert codec.identity_map["forward"][token] == token
    assert codec.identity_map["forward"]["raw-original"] == prefix + "002"
    assert all(codec.identity_map["reverse"][short] == raw
               for raw, short in codec.identity_map["forward"].items())


def test_notebook_raw_b_collision_cannot_overwrite_entry_and_read_restore_remains_valid():
    books = AuditNotebooks("audit-original", "run-original")
    books.put("evidence_worker", "raw-key", "facts", {"group_id": "raw-group", "amount": "10"})
    books.task("evidence_worker", {"task_id": "raw-task", "objective": "keep"}, "completed")
    saved = books.snapshot()
    with pytest.raises(ValueError, match="冲突"):
        books.put("evidence_worker", "B001", "facts", {"amount": "999"})
    with pytest.raises(ValueError, match="冲突"):
        books.task("evidence_worker", {"task_id": "K001", "objective": "replace"}, "completed")
    with pytest.raises(ValueError, match="冲突"):
        books.put("evidence_worker", "raw-key", "facts", {"group_id": "G001"})
    assert books.snapshot() == saved
    restored = AuditNotebooks("audit-original", "run-original", saved)
    assert restored.snapshot() == saved
    projection = restored.codec.encode(saved["notebooks"])
    assert restored.codec.encode(projection) == projection
    assert restored.internal_snapshot()["notebooks"]["evidence_worker"]["entries"][0]["content"]["amount"] == "10"
    new_run = AuditNotebooks("audit-original", "new-run", saved)
    assert not new_run.snapshot()["notebooks"]["evidence_worker"]["entries"]
    assert all(new_run.codec.identity_map["forward"][raw] == token
               for raw, token in saved["identity_map"]["forward"].items())


@pytest.mark.parametrize("mutate", [
    lambda m: m.update(version=True), lambda m: m.pop("next"),
    lambda m: m.update(forward=[]), lambda m: m.update(reverse=[]),
    lambda m: m.update(next=[]), lambda m: m["next"].update(G=True),
    lambda m: m["next"].update(G="2"), lambda m: m["next"].update(G=0),
    lambda m: m["next"].update(G=1), lambda m: m["next"].update(GG=2),
    lambda m: m.update(extra={}),
    lambda m: (m["forward"].update(G001="K001"), m["reverse"].update(K001="G001"), m["next"].update(K=2)),
])
def test_constructor_strict_maps_and_counters_without_repair(mutate):
    codec = ShortReferenceCodec()
    codec.reference("raw-group", "G", encoded=False)
    mapping = deepcopy(codec.identity_map)
    mutate(mapping)
    saved = deepcopy(mapping)
    with pytest.raises(ValueError):
        ShortReferenceCodec(mapping)
    with pytest.raises(ValueError):
        AuditNotebooks("audit", "run", {"version": 2, "identity_map": mapping})
    assert mapping == saved


def test_allocator_never_uses_an_existing_raw_identity_as_another_alias():
    codec = ShortReferenceCodec()
    codec.reference("G001", "S", encoded=False)
    assert codec.reference("raw-group", "G", encoded=False) == "G002"
    assert ShortReferenceCodec(codec.identity_map).identity_map == codec.identity_map


def worker_context():
    return {"source_inventory": {
        "transactions": [None, {}, {"id": "tx", "amount": "10", "direction": "debit"}],
        "receipts": [None, {}, {"id": "receipt", "amount": "10", "type": "expense"}]},
        "deterministic_candidates": [{"transaction_id": "tx", "receipt_upload_id": "receipt"}]}


def test_worker_inventory_only_fallback_filters_invalid_rows():
    context = worker_context()
    saved = deepcopy(context)
    chunks = agent._worker_chunks(context)
    assert len(chunks) == 1
    assert [row["id"] for row in chunks[0]["transactions"]] == ["tx"]
    assert [row["id"] for row in chunks[0]["receipts"]] == ["receipt"]
    assert context == saved


@pytest.mark.parametrize("plural", ["transactions", "receipts", "both"])
def test_worker_explicit_empty_residual_does_not_resurrect_inventory(plural):
    context = worker_context()
    for key in ("transactions", "receipts") if plural == "both" else (plural,):
        context[key] = []
    assert agent._worker_chunks(context) == []


@pytest.mark.parametrize("field,identity", [("confirmed_transaction_ids", "tx"), ("confirmed_receipt_ids", "receipt")])
def test_worker_inventory_only_still_excludes_confirmed(field, identity):
    context = worker_context()
    context[field] = [identity]
    assert agent._worker_chunks(context) == []