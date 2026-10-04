"""Persistent display references. Identity maps are transport metadata, never prompts.

One raw identity has one reference, even when it occurs in multiple scopes.
Financial values are not identities: only typed ID fields, UUIDs and hashes
are registered. No truncation, financial inference or confidence adjustment.
"""
from __future__ import annotations

from copy import deepcopy
import re
from threading import RLock
from typing import Any

REFERENCE_SCHEMA = "audit-short-references-v1"
UUID = re.compile(r"(?<![0-9a-fA-F])[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}(?::\d+)?(?![0-9a-fA-F])")
HASH = re.compile(r"(?<![0-9a-fA-F])[0-9a-fA-F]{64}(?![0-9a-fA-F])")
SHORT = re.compile(r"^[A-Z][0-9]+$")
FIELDS = {
    "audit_id": "A", "run_id": "N", "_run_id": "N",
    "group_id": "G", "match_group_id": "G",
    "upload_id": "U", "upload_ids": "U", "source_id": "S", "source_ids": "S",
    "transaction_id": "T", "transaction_ids": "T", "group_transaction_ids": "T",
    "shared_transaction_ids": "T", "confirmed_transaction_ids": "T",
    "approved_transaction_ids": "T", "receipt_id": "R", "receipt_ids": "R",
    "receipt_upload_id": "R", "receipt_upload_ids": "R", "confirmed_receipt_ids": "R",
    "confirmed_receipt_upload_ids": "R", "duplicate_candidates": "R",
    "task_id": "K", "parent_task_id": "K", "derived_from_task_id": "K", "depends_on": "K",
    "key": "B", "source_key": "B", "source_fingerprint": "F", "fingerprint": "F",
    "search_id": "Q", "skill_key": "S", "id": "S",
}


class ShortReferenceCodec:
    def __init__(self, identity_map: Any = None) -> None:
        self.lock = RLock()
        self._text_pattern: Any = None
        self._text_revision = -1
        self.identity_map = {"version": 1, "forward": {}, "reverse": {}, "next": {}}
        if identity_map is not None:
            if (not isinstance(identity_map, dict)
                    or set(identity_map) != {"version", "forward", "reverse", "next"}
                    or type(identity_map.get("version")) is not int or identity_map["version"] != 1):
                raise ValueError("短编号映射版本无效")
            forward, reverse, counters = (identity_map[name] for name in ("forward", "reverse", "next"))
            # Fail closed on corrupt/ambiguous maps; never silently reassign.
            if not all(isinstance(part, dict) for part in (forward, reverse, counters)):
                raise ValueError("短编号映射格式无效")
            for raw, token in forward.items():
                if not isinstance(raw, str) or not raw or not isinstance(token, str) or not SHORT.fullmatch(token) or reverse.get(token) != raw:
                    raise ValueError("短编号映射不一致")
            if len(forward) != len(reverse):
                raise ValueError("短编号映射不唯一")
            for prefix, count in counters.items():
                if (not isinstance(prefix, str) or not re.fullmatch(r"[A-Z]", prefix)
                        or type(count) is not int or count < 1):
                    raise ValueError("短编号计数无效")
            for token, raw in reverse.items():
                if (not isinstance(token, str) or not SHORT.fullmatch(token)
                        or not isinstance(raw, str) or forward.get(raw) != token):
                    raise ValueError("短编号映射不一致")
                if counters.get(token[0], 0) <= int(token[1:]):
                    raise ValueError("短编号计数无效")
                if raw in reverse and reverse[raw] != raw:
                    raise ValueError("真实标识与已有短编号冲突")
            self.identity_map = deepcopy(identity_map)

    def reference(self, value: Any, prefix: str = "S", *, encoded: bool = True) -> str:
        """Cache-compatible projection; raw callers must pass encoded=False."""
        raw = str(value)
        if not raw:
            return raw
        if not isinstance(prefix, str) or not re.fullmatch(r"[A-Z]", prefix):
            raise ValueError("短编号前缀无效")
        with self.lock:
            forward, reverse, counters = (self.identity_map[name] for name in ("forward", "reverse", "next"))
            if raw in forward:
                return forward[raw]
            if raw in reverse:
                if encoded:
                    return raw
                raise ValueError("真实标识与已有短编号冲突")
            number = int(counters.get(prefix, 1))
            token = raw if SHORT.fullmatch(raw) and raw[0] == prefix else f"{prefix}{number:03d}"
            while token in reverse or (token in forward and token != raw):
                number += 1
                token = f"{prefix}{number:03d}"
            forward[raw], reverse[token] = token, raw
            counters[prefix] = max(number + 1, int(token[1:]) + 1)
            return token

    def seed(self, context: dict[str, Any]) -> None:
        inventory = context.get("source_inventory") or {}
        # Reserve all typed literal identities before allocating labels;
        # an incremental raw identity colliding with an existing display token
        # is ambiguous and must fail closed, never alias two real resources.
        literal_ids = [str(item["id"]) for plural in ("transactions", "receipts")
                       for item in [*(inventory.get(plural) or []), *(context.get(plural) or [])]
                       if isinstance(item, dict) and item.get("id") and SHORT.fullmatch(str(item["id"]))]
        for candidate in context.get("deterministic_candidates") or []:
            if isinstance(candidate, dict):
                values = [candidate.get("transaction_id"), candidate.get("receipt_upload_id"),
                          *(candidate.get("receipt_upload_ids") or [])]
                literal_ids.extend(str(value) for value in values if value and SHORT.fullmatch(str(value)))
        raw_context = {key: value for key, value in context.items()
                       if key not in ("_agent_state", "identity_map", "_prior_steps")}

        def collect(value: Any, field: str = "") -> None:
            if isinstance(value, dict):
                for key, child in value.items():
                    if key not in ("_agent_state", "identity_map", "_prior_steps"):
                        collect(child, str(key))
            elif isinstance(value, (list, tuple)):
                for child in value:
                    collect(child, field)
            elif (isinstance(value, str) and SHORT.fullmatch(value)
                  and (field in FIELDS or field.endswith("_id") or field.endswith("_ids"))):
                literal_ids.append(value)

        collect(raw_context)
        for raw in dict.fromkeys(literal_ids):
            reverse = self.identity_map["reverse"]
            if raw in reverse and reverse[raw] != raw and raw not in self.identity_map["forward"]:
                raise ValueError("真实标识与已有短编号冲突")
            self.reference(raw, raw[0], encoded=False)
        for plural, prefix in (("transactions", "T"), ("receipts", "R")):
            rows = inventory.get(plural) or []
            fallback = context.get(plural) or []
            if not inventory:
                values = {str(item["id"]) for item in fallback if isinstance(item, dict) and item.get("id")}
                for candidate in context.get("deterministic_candidates") or []:
                    if not isinstance(candidate, dict):
                        continue
                    if plural == "transactions" and candidate.get("transaction_id"):
                        values.add(str(candidate["transaction_id"]))
                    if plural == "receipts":
                        values.update(str(value) for value in candidate.get("receipt_upload_ids") or [])
                        if candidate.get("receipt_upload_id"):
                            values.add(str(candidate["receipt_upload_id"]))
                fallback = [{"id": value} for value in sorted(values)]
            for item in [*rows, *fallback]:
                if isinstance(item, dict) and item.get("id"):
                    self.reference(item["id"], prefix, encoded=False)
        # Register nested IDs only AFTER full T/R inventory (including confirmed).
        self.encode(raw_context, raw=True)

    def text(self, value: str) -> str:
        with self.lock:
            # Match whole compound identities before bare upload UUIDs.
            forward = self.identity_map["forward"]
            if self._text_revision != len(forward):
                # Free prose replaces UUID identities only. A task/group named
                # "income" must never rewrite financial evidence saying income.
                # Other IDs are shortened in typed fields, not word substitution.
                known = [raw for raw, token in forward.items() if raw != token
                         and UUID.fullmatch(raw)]
                expression = "|".join(re.escape(raw) for raw in sorted(known, key=len, reverse=True))
                self._text_pattern = re.compile(r"(?<![\w-])(?:" + expression + r")(?![\w-])") if expression else None
                self._text_revision = len(forward)
            if self._text_pattern is not None:
                value = self._text_pattern.sub(lambda match: forward[match.group()], value)
            value = UUID.sub(lambda match: self.reference(match.group(), "S"), value)
            return HASH.sub(lambda match: self.reference(match.group(), "F"), value)

    def encode(self, value: Any, field: str = "", *, storage: bool = False, raw: bool = False) -> Any:
        """Project short caches idempotently; raw inputs/storage must not alias tokens."""
        with self.lock:
            if isinstance(value, dict):
                return {self.encode(str(key), storage=storage, raw=raw): self.encode(child, str(key), storage=storage, raw=raw) for key, child in value.items()
                        if key not in ("identity_map", "_agent_state")}
            if isinstance(value, (list, tuple)):
                return [self.encode(child, field, storage=storage, raw=raw) for child in value]
            if isinstance(value, str):
                if value and field in FIELDS:
                    prefix = "Q" if field == "key" and value.startswith("search:") else FIELDS[field]
                    return self.reference(value, prefix, encoded=not (raw or storage))
                if value and (field.endswith("_id") or field.endswith("_ids")):
                    return self.reference(value, "S", encoded=not (raw or storage))
                if storage:
                    # Accounting references may literally say R001/T001. Keep
                    # those facts distinct from our UUID display labels through
                    # cache/restore. Escapes never enter model projections.
                    value = value.replace("~", "~~")
                    value = re.sub(r"(?<!\w)[A-Z][0-9]+(?!\w)", lambda match: "~" + match.group(), value)
                return self.text(value)
            return deepcopy(value)

    def decode(self, value: Any, *, text: bool = True) -> Any:
        """Trusted stored derivatives only; unknown model references remain unknown."""
        if isinstance(value, dict):
            return {self.decode(key, text=text): self.decode(child, text=text) for key, child in value.items()}
        if isinstance(value, list):
            return [self.decode(child, text=text) for child in value]
        if isinstance(value, str):
            reverse = self.identity_map["reverse"]
            if value in reverse:
                return reverse[value]
            if text:
                # Consume escaped literal segments atomically; only unescaped
                # display references restore original identities.
                def restore(match: Any) -> str:
                    token = match.group()
                    if token == "~~":
                        return "~"
                    if token.startswith("~"):
                        return token[1:]
                    return reverse.get(token, token)
                return re.sub(r"~~|~[A-Z][0-9]+|(?<!\w)[A-Z][0-9]+(?!\w)", restore, value)
        return deepcopy(value)

    def model_result(self, value: Any, field: str = "") -> Any:
        """Restore typed boundary IDs, not prose; unknown IDs cannot alias raw scope."""
        if isinstance(value, dict):
            return {key: self.model_result(child, key) for key, child in value.items()}
        if isinstance(value, list):
            return [self.model_result(child, field) for child in value]
        if isinstance(value, str) and (field in FIELDS or field.endswith("_id") or field.endswith("_ids")):
            if SHORT.fullmatch(value) and value not in self.identity_map["reverse"]:
                return "unknown-reference:" + value
            return self.identity_map["reverse"].get(value, value)
        return deepcopy(value)