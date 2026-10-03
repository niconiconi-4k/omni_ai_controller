"""Deterministic v1 inventory contract; mirrored by Main, never a model output.

Source hashes cover the complete canonical compact source, not its projection.
Only rebuildable individual entries are evicted; source inputs remain immutable.
"""
from collections import Counter
from copy import deepcopy
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import json
from typing import Any

MAX_NOTEBOOK_BYTES = 8 * 1024 * 1024
OWNERS = ("audit_planner", "evidence_worker")
SEED_PLAYBOOK_ID = "accounting-expense-first-v1"
SEED_STRATEGY_ORDER = ["corporate_card_expenses", "direct_expenses", "revenue_settlements",
                       "employee_reimbursements", "anomaly_review"]
SEED_OBJECTIVES = ["整理全部流水与独立子票基础信息，按类型、币种和实际事件日期索引",
                   "只检索未确认内核候选，金额优先，身份详情按需提取",
                   "李师傅唯一审批；员工报销不凑单，工资差额仅疑似，异常最后复核"]
BASIC_FIELDS = ("cashflow_lane", "signed_cashflow_amount", "amount", "signed_amount", "total_amount", "currency", "currency_source",
                "date", "date_role", "date_evidence", "time", "type", "category", "refund",
                "document_type", "receipt_type", "direction", "payment_date", "value_date",
                "transaction_date", "actual_payment_date", "sale_date", "event_date",
                "issue_date", "due_date", "booking_date", "amount_decimal", "amount_text", "amount_effect",
                "transaction_time_text", "transaction_time_iso", "transaction_time_role", "document_kind",
                "document_date_iso", "due_date_iso", "amount_components")
NON_EVENT_ROLES = {"unknown", "issue", "issued", "creation", "due", "invoice_date", "issue_date", "due_date",
                   "document_issue", "document_creation", "document_due", "payment_due", "invoice_issue", "invoice_due"}


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"),
                      sort_keys=True, allow_nan=False).encode("utf-8")


def fingerprint(value: Any) -> str:
    return sha256(canonical_bytes(value)).hexdigest()


def source_amount(item: dict[str, Any]) -> Any:
    financial = item.get("financial_facts") or {}
    # Compact amount is a matching magnitude; signed_amount/raw facts retain
    # the original economic sign (in particular negative refund sources).
    mappings = ((item, ("signed_amount",)), (financial, ("amount_decimal",)),
                (item, ("amount", "total_amount", "amount_decimal")))
    for mapping, keys in mappings:
        if isinstance(mapping, dict):
            for key in keys:
                if mapping.get(key) is not None:
                    return mapping[key]
    return None


def _is_transaction(item: dict[str, Any], source_kind: str | None = None) -> bool:
    if source_kind is not None:
        return source_kind == "transaction"
    return any(key in item for key in ("direction", "booking_date", "value_date"))


def _decimal_amount(item: dict[str, Any]) -> Decimal | None:
    raw = source_amount(item)
    if raw is None or isinstance(raw, bool):
        return None
    try:
        value = Decimal(str(raw))
        return value if value.is_finite() else None
    except (InvalidOperation, TypeError, ValueError):
        return None


def _receipt_category(item: dict[str, Any]) -> str:
    financial = item.get("financial_facts") if isinstance(item.get("financial_facts"), dict) else {}
    words = " ".join(str(item.get(key) or "") for key in (
        "type", "document_type", "receipt_type", "allocation_role",
    )).lower() + " " + str(financial.get("document_kind") or "").lower()
    if any(word in words for word in ("payroll", "salary", "工资")):
        return "payroll"
    if any(word in words for word in ("income", "revenue", "sales_report", "收入")):
        return "income"
    if any(word in words for word in ("expense", "tax_voucher", "loan_interest_voucher", "支出")):
        return "expense"
    return "unknown"


def cashflow_facts(item: dict[str, Any], *, source_kind: str | None = None) -> dict[str, Any]:
    """Economic sign, separate from matching magnitude and receipt reversal sign.

    Bank positive magnitudes obey direction. Negative raw debit is valid;
    credit plus negative raw is contradictory. Unknown direction/type is never
    guessed from a document brand, filename, invoice date or cached lane.
    """
    unknown = {"cashflow_lane": "unknown", "signed_cashflow_amount": None}
    value = _decimal_amount(item)
    if value is None or value == 0:
        return unknown
    if _is_transaction(item, source_kind):
        direction = str(item.get("direction") or "").strip().lower()
        if direction in {"credit", "in", "inflow", "incoming"}:
            if value < 0:
                return unknown
            lane = "in"
        elif direction in {"debit", "out", "outflow", "outgoing"}:
            if item.get("signed_amount") is not None and value > 0:
                return unknown
            lane = "out"
        elif direction:
            return unknown
        else:
            lane = "out" if value < 0 else "in"
    else:
        typ = _receipt_category(item)
        explicit = item.get("category")
        if explicit == "unknown":
            return unknown
        if typ == "unknown" and explicit in {"income", "expense", "payroll"}:
            typ = explicit
        if typ == "unknown":
            return unknown
        financial = item.get("financial_facts") if isinstance(item.get("financial_facts"), dict) else {}
        effect = item.get("amount_effect") or financial.get("amount_effect")
        reversal = item.get("refund") is True or explicit == "refund" or value < 0 or effect in {"reversal", "refund", "decrease", "credit"}
        if value < 0 and effect == "normal":
            return unknown
        lane = "in" if (typ == "income") != reversal else "out"
    magnitude = value.copy_abs()
    signed = magnitude if lane == "in" else magnitude.copy_negate()
    return {"cashflow_lane": lane, "signed_cashflow_amount": str(signed)}


def category(item: dict[str, Any], *, source_kind: str | None = None) -> str:
    financial = item.get("financial_facts") if isinstance(item.get("financial_facts"), dict) else {}
    if _is_transaction(item, source_kind):
        explicit = item.get("category")
        if explicit in ("expense", "income", "refund", "payroll", "unknown"):
            return explicit
        return {"in": "income", "out": "expense"}.get(cashflow_facts(item, source_kind="transaction")["cashflow_lane"], "unknown")
    effect = item.get("amount_effect") or financial.get("amount_effect")
    if item.get("refund") is True or effect in ("reversal", "refund", "decrease", "credit"):
        return "refund"
    explicit = item.get("category")
    if explicit in ("expense", "income", "refund", "payroll", "unknown"):
        return explicit
    value = _decimal_amount(item)
    if value is not None and value < 0:
        return "refund"
    words = " ".join(str(item.get(key) or "") for key in (
        "allocation_role", "type", "document_type", "receipt_type", "direction", "kind",
    )).lower() + " " + str(financial.get("document_kind") or "").lower()
    if any(word in words for word in ("refund", "credit_note", "退款")):
        return "refund"
    if any(word in words for word in ("payroll", "salary", "工资")):
        return "payroll"
    if any(word in words for word in ("income", "revenue", "credit", "inflow", "incoming", "sales_report", "收入")) or item.get("direction") == "in":
        return "income"
    return _receipt_category(item)


def cashflow_sort_key(item: dict[str, Any]) -> tuple:
    """Lane/currency/actual event precede subtype; stable ties, never UUID."""
    lane = item.get("cashflow_lane")
    if lane not in {"in", "out", "unknown"}:
        lane = cashflow_facts(item)["cashflow_lane"]
    actual = event_date(item)
    return ({"in": 0, "out": 1, "unknown": 2}[lane], str(item.get("currency") or ""),
            not actual, actual, str(item.get("category") or ""), str(item.get("type") or ""))


def partition_cashflow(items: list[dict[str, Any]], *, source_kind: str | None = None) -> dict[str, list[dict[str, Any]]]:
    """Integration hook: separate worker lanes without splitting candidate groups.

    Call on transaction sources, then keep each transaction's candidates atomic.
    Unknown is a separate review bucket, not an income or expense fallback.
    """
    lanes: dict[str, list[dict[str, Any]]] = {"in": [], "out": [], "unknown": []}
    for item in items:
        projected = {**item, **cashflow_facts(item, source_kind=source_kind)}
        lanes[projected["cashflow_lane"]].append(projected)
    for rows in lanes.values():
        rows.sort(key=cashflow_sort_key)
    return lanes


def event_date(item: dict[str, Any]) -> str:
    financial = item.get("financial_facts") if isinstance(item.get("financial_facts"), dict) else {}
    if "event_date" in item:
        role = str(item.get("date_role") or item.get("transaction_time_role") or financial.get("transaction_time_role") or "").lower()
        return str(item.get("event_date") or "") if role not in NON_EVENT_ROLES else ""
    for mapping in (financial, item):
        if mapping.get("transaction_time_role") in ("actual_payment", "sales_activity", "settlement") and mapping.get("transaction_time_iso"):
            return str(mapping["transaction_time_iso"])
    for key in ("actual_payment_date", "payment_date", "transaction_date", "sale_date"):
        if item.get(key) or financial.get(key):
            return str(item.get(key) or financial[key])
    role = str(item.get("date_role") or item.get("transaction_time_role") or financial.get("transaction_time_role") or "").lower()
    if role in NON_EVENT_ROLES:
        return ""
    return str(item.get("date") or item.get("booking_date") or "")


def base_facts(source: dict[str, Any], *, source_kind: str | None = None) -> dict[str, Any]:
    content = {key: deepcopy(source[key]) for key in BASIC_FIELDS if key in source}
    financial = source.get("financial_facts")
    if isinstance(financial, dict):
        content["financial_facts"] = {key: deepcopy(financial[key]) for key in BASIC_FIELDS if key in financial}
    content.update(id=source.get("id"), category=category(source, source_kind=source_kind), event_date=event_date(source),
                   source_amount=source_amount(source), **cashflow_facts(source, source_kind=source_kind))
    return content


def fit_book(book: dict[str, Any]) -> None:
    """Linear-size accounting, including commas, envelope and counter digits."""
    entries = book["entries"]
    sizes = [len(canonical_bytes(entry)) for entry in entries]
    total = sum(sizes)
    removed = 0
    while True:
        size = 0
        for _ in range(8):
            measured = len(canonical_bytes({**book, "entries": [], "used_bytes": size})) + total + max(0, len(entries) - removed - 1)
            if measured == size:
                break
            size = measured
        if size <= MAX_NOTEBOOK_BYTES or removed == len(entries):
            break
        total -= sizes[removed]
        removed += 1
        book["evicted_entries"] += 1
    if removed:
        del entries[:removed]
    book["used_bytes"] = size


def inventory_task(command: dict[str, Any] | None = None, *, iteration: int = 1) -> dict[str, Any]:
    return {"task_id": "inventory:base" if command is None else f"inventory:plan:{iteration}:{command['task_id']}",
            "operation": "organize", "status": "completed", "priority": 100,
            "objective": SEED_OBJECTIVES[0], "depends_on": [], "receipt_ids": [], "transaction_ids": [],
            "amount": None, "strategy": "source_inventory", "scope": "all_source_inventory",
            "status_source": "deterministic", "origin": "planner_seed" if command is None else "li_organize_command",
            "derived_from_task_id": None if command is None else command["task_id"]}


def register_inventory(state: dict[str, Any], inventory: dict[str, Any],
                       command: dict[str, Any] | None = None, *, iteration: int = 1) -> None:
    """Register all sources, including confirmed IDs, without touching decisions."""
    sources = [(kind, source) for plural, kind in (("transactions", "transaction"), ("receipts", "receipt"))
               for source in inventory.get(plural) or []]
    projections = [(f"{kind}:{source['id']}:basic", fingerprint(source), base_facts(source, source_kind=kind))
                   for kind, source in sources]
    summary = {"counts": {kind: len(inventory.get(kind) or []) for kind in ("transactions", "receipts")},
               "category_counts": {kind: dict(Counter(category(item, source_kind=kind[:-1]) for item in inventory.get(kind) or []))
                                   for kind in ("transactions", "receipts")},
               "cashflow_lane_counts": {kind: dict(Counter(cashflow_facts(item, source_kind=kind[:-1])["cashflow_lane"] for item in inventory.get(kind) or []))
                                        for kind in ("transactions", "receipts")},
               "purpose": "deterministic_cache_only", "include_in_model_prompt": False,
               "ordering": "cashflow_lane/currency/actual_event_date/category/type; unknown_last; ties_retained; no_id_tiebreak"}
    seed = {"id": SEED_PLAYBOOK_ID, "strategy_order": SEED_STRATEGY_ORDER,
            "objectives": SEED_OBJECTIVES, "origin": "planner_seed", "status_source": "deterministic",
            "not_model_output": True}
    index = sorted(
        [{"source_key": key, "id": fact["id"], "category": fact["category"], "type": fact.get("type"),
          "cashflow_lane": fact["cashflow_lane"], "signed_cashflow_amount": fact["signed_cashflow_amount"],
          "currency": fact.get("currency"), "amount": fact["source_amount"], "event_date": fact["event_date"]}
         for key, _, fact in projections],
        key=lambda item: ({"in": 0, "out": 1, "unknown": 2}[item["cashflow_lane"]], str(item["currency"] or ""),
                          not item["event_date"], item["event_date"], item["category"], str(item["type"] or "")),
    )
    inventory_hash = fingerprint([(key, source_hash) for key, source_hash, _ in projections])
    for owner in OWNERS:
        book = state["notebooks"][owner]
        entries = {entry["key"]: entry for entry in book["entries"]}

        def put(key: str, kind: str, content: Any, source_hash: str) -> None:
            old = entries.get(key)
            if old and old["source_fingerprint"] == source_hash and old["content"] == content:
                return
            entry = {"key": key, "kind": kind, "content": deepcopy(content), "source_fingerprint": source_hash}
            probe = {**book, "entries": [entry], "used_bytes": 0}
            fit_book(probe)
            if not probe["entries"]:
                book["overflow"] += 1
                return
            entries[key] = entry

        for key, source_hash, content in projections:
            old = entries.get(key)
            if owner == "evidence_worker":
                stat = "cache_hits" if old and old["source_fingerprint"] == source_hash else "cache_misses"
                state["stats"][stat] = state["stats"].get(stat, 0) + 1
            if old and old["source_fingerprint"] != source_hash:
                prefix = key.rsplit(":", 1)[0] + ":"
                entries = {k: v for k, v in entries.items() if not k.startswith(prefix)}
                if owner == "evidence_worker":
                    state["stats"]["invalidations"] = state["stats"].get("invalidations", 0) + 1
            put(key, "source_basic", content, source_hash)
        put("inventory:seed", "planner_seed", seed, fingerprint(seed))
        # Independently bounded/sharded indices, never one all-inventory entry.
        for offset in range(0, len(index), 128):
            shard = index[offset:offset + 128]
            put(f"inventory:index:{offset // 128}", "source_index", shard, fingerprint(shard))
        put("inventory:summary", "source_inventory_summary", summary, inventory_hash)
        book["entries"] = list(entries.values())
        fit_book(book)
        task = {**inventory_task(command, iteration=iteration), "source_counts": summary["counts"],
                "source_fingerprint": inventory_hash}
        tasks = state["task_lists"][owner]
        previous = next((item for item in tasks if item["task_id"] == task["task_id"]), None)
        if previous is not None:
            previous.update(task)
        else:
            if len(tasks) >= 512:
                removable = next((item for item in tasks if item["status"] in
                                  ("completed", "partial", "blocked", "cancelled", "failed", "superseded")), None)
                if removable is None:
                    state["stats"]["task_overflow"] = state["stats"].get("task_overflow", 0) + 1
                    raise ValueError("智能体任务列表超过有界预算")
                tasks.remove(removable)
                state["stats"]["task_evictions"] = state["stats"].get("task_evictions", 0) + 1
            tasks.append(task)