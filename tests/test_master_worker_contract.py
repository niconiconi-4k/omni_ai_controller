from omni_ai_controller.agentic_audit import _kernel_relations, _worker_chunks
from omni_ai_controller.audit_notebooks import AuditNotebooks, category


def test_main_match_group_alias_preserves_whole_group_and_exclusion():
    context = {
        "transactions": [{"id": "tx"}],
        "receipts": [{"id": "a"}, {"id": "b"}],
        "deterministic_candidates": [
            {"transaction_id": "tx", "receipt_upload_id": key, "match_group_id": "batch"}
            for key in ("a", "b")
        ],
    }
    chunks = _worker_chunks(context)
    assert _kernel_relations(chunks[0]) == {("tx", frozenset({"a", "b"}), "batch")}
    assert _worker_chunks({**context, "confirmed_receipt_ids": ["a"]}) == []


def test_refund_facts_and_detail_account_ownership_survive_cache():
    source = {"id": "refund", "type": "expense", "amount": "1200",
              "financial_facts": {"amount_decimal": "-1200", "amount_effect": "refund"},
              "account_suffixes": ["1234"]}
    notebook = AuditNotebooks("audit", "run")
    assert category(source) == "refund"
    assert notebook.facts(source)["category"] == "refund"
    assert notebook.facts(source, details=True)["account_suffixes"] == ["1234"]
    notebook.facts(source)
    state = notebook.snapshot()
    resumed = AuditNotebooks("audit", "run", state)
    resumed.facts(source)
    assert resumed.snapshot()["stats"]["cache_hits"] == state["stats"]["cache_hits"] + 1