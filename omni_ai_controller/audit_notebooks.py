"""Bounded, case/run-bound working memory. Cache facts, never accounting rules.

Source documents and approved decisions remain in their authoritative inputs /
steps; notebooks contain rebuildable derivatives only. No notebook is a model
prompt. UTF-8 limits include the complete notebook envelope, not just content.
"""
from __future__ import annotations

from copy import deepcopy
from decimal import Decimal, InvalidOperation
import json
from threading import RLock
from typing import Any, Callable
from .audit_references import ShortReferenceCodec, REFERENCE_SCHEMA

from .audit_inventory import (
    base_facts, category, event_date, fingerprint, fit_book, register_inventory, source_amount,
)


MAX_NOTEBOOK_BYTES = 8 * 1024 * 1024
MAX_TASK_ENTRIES = 512
OWNERS = ("audit_planner", "evidence_worker")
OPERATIONS = ("organize", "search_amount", "compare_details", "review_anomaly")


def serialized(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def amount_key(value: Any) -> str | None:
    try:
        number = Decimal(str(value))
        return str(abs(number).normalize()) if number.is_finite() else None
    except (InvalidOperation, ValueError, TypeError):
        return None


def confirmed_ids(context: dict[str, Any]) -> tuple[set[str], set[str]]:
    transactions = {str(value) for value in context.get("confirmed_transaction_ids") or []}
    receipts = {str(value) for value in context.get("confirmed_receipt_ids") or []}
    receipts.update(str(value) for value in context.get("confirmed_receipt_upload_ids") or [])
    # Main's actually accepted IDs are the only cross-round authority. Historical
    # model proposals (even high-confidence matches) are not confirmations.
    return transactions, receipts


def candidate_receipts(candidate: dict[str, Any]) -> set[str]:
    return {str(value) for value in candidate.get("receipt_upload_ids") or []} | (
        {str(candidate["receipt_upload_id"])} if candidate.get("receipt_upload_id") else set()
    )


class AuditNotebooks:
    def __init__(self, audit_id: str, run_id: str, previous: Any = None,
                 publish: Callable[[dict[str, Any]], None] | None = None,
                 *, source_inventory: dict[str, Any] | None = None) -> None:
        self.lock = RLock()
        self.publish = publish
        old_codec = ShortReferenceCodec(previous.get("identity_map") if isinstance(previous, dict) else None)
        same_audit = isinstance(previous, dict) and old_codec.decode(previous.get("audit_id"), text=False) == audit_id
        self.codec = old_codec if same_audit else ShortReferenceCodec()
        valid = (same_audit and previous.get("version") in (1, 2)
                 and self.codec.decode(previous.get("run_id"), text=False) == run_id)
        if source_inventory is not None:
            self.codec.seed({"source_inventory": source_inventory})
        if valid and previous.get("version") == 1:
            # Seed legacy source entries before generic id traversal; retain all
            # evidence and full original fingerprint for exact cache hits.
            for book in (previous.get("notebooks") or {}).values():
                for entry in book.get("entries") or []:
                    content = entry.get("content")
                    key = str(entry.get("key") or "")
                    if isinstance(content, dict) and content.get("id") and key.startswith(("transaction:", "receipt:")):
                        self.codec.reference(content["id"], "T" if key.startswith("transaction:") else "R", encoded=False)
        self.state: dict[str, Any] = {
            "version": 2, "reference_schema": REFERENCE_SCHEMA,
            "audit_id": self.codec.reference(audit_id, "A", encoded=False), "run_id": self.codec.reference(run_id, "N", encoded=False),
            "identity_map": self.codec.identity_map,
            "notebooks": {}, "task_lists": {},
            "stats": {"cache_hits": 0, "cache_misses": 0, "invalidations": 0,
                      "amount_searches": 0, "exact_amount_hits": 0, "duplicate_candidates": 0,
                      "task_evictions": 0, "task_overflow": 0, "evidence_reuses": 0},
        }
        if valid:
            for key in self.state["stats"]:
                count = (previous.get("stats") or {}).get(key)
                if type(count) is int and count >= 0:
                    self.state["stats"][key] = count
        for owner in OWNERS:
            old = (previous.get("notebooks") or {}).get(owner, {}) if valid else {}
            self.state["notebooks"][owner] = {
                "limit_bytes": MAX_NOTEBOOK_BYTES, "used_bytes": 0,
                "entries": self.codec.encode(self.codec.decode(old.get("entries") or []) if valid and previous.get("version") == 2 else old.get("entries") or [], storage=True),
                "evicted_entries": int(old.get("evicted_entries") or 0),
                "overflow": int(old.get("overflow") or 0),
            }
            tasks = (previous.get("task_lists") or {}).get(owner, []) if valid else []
            self.state["task_lists"][owner] = self.codec.encode(self.codec.decode(tasks[-MAX_TASK_ENTRIES:]) if valid and previous.get("version") == 2 else tasks[-MAX_TASK_ENTRIES:], storage=True)
            self._fit(owner)

    def _measure(self, notebook: dict[str, Any]) -> int:
        # used_bytes itself contributes digits to the serialized size.
        for _ in range(8):
            size = len(serialized(notebook).encode("utf-8"))
            if notebook["used_bytes"] == size:
                return size
            notebook["used_bytes"] = size
        return len(serialized(notebook).encode("utf-8"))

    def _fit(self, owner: str) -> None:
        fit_book(self.state["notebooks"][owner])

    def organize_inventory(self, inventory: dict[str, Any], command: dict[str, Any] | None = None,
                           *, iteration: int = 1) -> None:
        with self.lock:
            register_inventory(self.state, inventory, command, iteration=iteration, codec=self.codec)
            self.emit()

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return deepcopy(self.state)

    def task_statuses(self, owner: str) -> dict[str, str]:
        with self.lock:
            return {self.codec.decode(task["task_id"], text=False): task["status"] for task in self.state["task_lists"][owner]}

    def internal_snapshot(self) -> dict[str, Any]:
        """Internal scheduling only; never publish this projection."""
        with self.lock:
            return {"notebooks": self.codec.decode(self.state["notebooks"]),
                    "task_lists": self.codec.decode(self.state["task_lists"])}

    def emit(self) -> None:
        with self.lock:
            if self.publish:
                self.publish(self.snapshot())

    def put(self, owner: str, key: str, kind: str, content: Any,
            source_fingerprint: str = "") -> bool:
        with self.lock:
            notebook = self.state["notebooks"][owner]
            short_key = self.codec.reference(key, "Q" if key.startswith("search:") else "B", encoded=False)
            entry = {"key": short_key, "kind": self.codec.text(kind), "content": self.codec.encode(content, storage=True),
                     "source_fingerprint": self.codec.reference(source_fingerprint, "F", encoded=False) if source_fingerprint else ""}
            probe = {**notebook, "entries": [entry], "used_bytes": 0}
            if self._measure(probe) > MAX_NOTEBOOK_BYTES:
                notebook["overflow"] += 1
                self._fit(owner)
                return False
            notebook["entries"] = [item for item in notebook["entries"] if item.get("key") != short_key]
            notebook["entries"].append(entry)
            self._fit(owner)
            return True

    def facts(self, source: dict[str, Any], *, details: bool = False, source_kind: str = "receipt") -> dict[str, Any]:
        """Cache basic amount/date/type first; extract identities only on demand."""
        key = f"{source_kind}:{source.get('id')}:" + ("details" if details else "basic")
        source_hash = fingerprint(source)
        with self.lock:
            self.codec.reference(source.get("id"), "T" if source_kind == "transaction" else "R", encoded=False)
            short_key = self.codec.reference(key, "B", encoded=False)
            entries = self.state["notebooks"]["evidence_worker"]["entries"]
            cached = next((entry for entry in entries if entry.get("key") == short_key), None)
            if cached and self.codec.decode(cached.get("source_fingerprint"), text=False) == source_hash:
                self.state["stats"]["cache_hits"] += 1
                return self.codec.decode(cached["content"])
            if cached:
                self.state["stats"]["invalidations"] += 1
                # Invalidate BOTH basic and detail projections on a source change.
                prefix = f"{source_kind}:{source.get('id')}:"
                entries[:] = [entry for entry in entries if not str(self.codec.decode(entry.get("key"), text=False)).startswith(prefix)]
            self.state["stats"]["cache_misses"] += 1
            content = base_facts(source, source_kind=source_kind)
            financial = source.get("financial_facts")
            if details:
                for name in ("party", "parties", "accounts", "account", "reference", "references", "taxes",
                             "merchant", "counterparty", "payment_method", "payment_breakdown", "employee",
                             "payer", "payee", "payment", "account_numbers", "account_suffixes", "reference_numbers", "description", "memo",
                             "ocr_excerpt", "text_excerpt"):
                    if name in source:
                        content[name] = deepcopy(source[name])
                    if isinstance(financial, dict) and name in financial:
                        content.setdefault("financial_facts", {})[name] = deepcopy(financial[name])
            self.put("evidence_worker", key, "receipt_details" if details else "source_basic", content, source_hash)
            return content

    def task(self, owner: str, task: dict[str, Any], status: str) -> None:
        with self.lock:
            task = self.codec.encode(task, storage=True)
            tasks = self.state["task_lists"][owner]
            entry = next((item for item in tasks if item.get("task_id") == task["task_id"]), None)
            if entry is None:
                if len(tasks) >= MAX_TASK_ENTRIES:
                    removable = next((item for item in tasks if item.get("status") in (
                        "completed", "partial", "blocked", "cancelled", "failed", "superseded")), None)
                    if removable is None:
                        self.state["stats"]["task_overflow"] += 1
                        raise ValueError("智能体任务列表超过有界预算")
                    tasks.remove(removable)
                    self.state["stats"]["task_evictions"] += 1
                entry = deepcopy(task)
                tasks.append(entry)
            else:
                entry.update(deepcopy(task))
            entry["status"] = status
            self.emit()  # Ordered publication of every task transition.

    def prepare(self, chunk: dict[str, Any], tasks: list[dict[str, Any]],
                stopped: Callable[[], bool] = lambda: False) -> dict[str, Any]:
        """Deterministic worker stage, run on the single bounded executor."""
        if stopped():
            return {"tasks": tasks, "cancelled": True}
        for task in tasks:
            self.task("evidence_worker", task, "preparing")
        details = any(task["operation"] in ("compare_details", "review_anomaly") for task in tasks)
        receipts = [self.facts(item) for item in chunk.get("receipts") or [] if not stopped()]
        transactions = [self.facts(item, source_kind="transaction") for item in chunk.get("transactions") or [] if not stopped()]
        # The in-memory index is rebuilt from fingerprint-validated notebook facts,
        # not raw OCR. Both exact amount lookup and duplicate retrieval use it.
        index: dict[str, list[dict[str, Any]]] = {}
        for receipt in receipts:
            value = amount_key(source_amount(receipt))
            if value is not None:
                index.setdefault(value, []).append(receipt)
        searches = []
        for task in tasks:
            if stopped():
                break
            target_amounts = [task["amount"]] if task.get("amount") is not None else [
                source_amount(item) for item in transactions
                if str(item.get("id")) in task["transaction_ids"]
            ]
            for amount in dict.fromkeys(amount_key(value) for value in target_amounts):
                if amount is None:
                    continue
                found = sorted(index.get(amount, []), key=lambda item: (item["category"], item["event_date"], str(item["id"])))
                ids = [str(item["id"]) for item in found if str(item["id"]) in task["receipt_ids"]]
                searches.append({"task_id": task["task_id"], "amount": amount, "receipt_ids": ids,
                                 "duplicate_candidates": ids if len(ids) > 1 else []})
                with self.lock:
                    self.state["stats"]["amount_searches"] += 1
                    self.state["stats"]["exact_amount_hits"] += len(ids)
                    self.state["stats"]["duplicate_candidates"] += max(0, len(ids) - 1)
                if len(ids) > 1:
                    details = True  # Same amount: identity/date disambiguation.
        if details:
            receipts = [self.facts(item, details=True) for item in chunk.get("receipts") or [] if not stopped()]
            for receipt in receipts:
                receipt.update(deepcopy((chunk.get("detail_excerpts") or {}).get(str(receipt["id"])) or {}))
            transactions = [self.facts(item, details=True, source_kind="transaction")
                            for item in chunk.get("transactions") or [] if not stopped()]
        result = {"tasks": tasks, "receipt_facts": sorted(receipts, key=lambda item: (item["category"], item["event_date"], str(item["id"]))),
                  "transaction_facts": sorted(transactions, key=lambda item: (item["category"], item["event_date"], str(item["id"]))),
                  "amount_searches": searches, "cancelled": stopped()}
        self.put("evidence_worker", "search:" + fingerprint(tasks), "search_results", result,
                 fingerprint({"receipts": chunk.get("receipts"), "transactions": chunk.get("transactions")}))
        for task in tasks:
            self.task("evidence_worker", task, "cancelled" if stopped() else "prepared")
        return result