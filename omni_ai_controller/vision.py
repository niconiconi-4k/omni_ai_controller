from __future__ import annotations

import base64
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

SUPPORTED_VISION_MODELS: dict[str, dict[str, object]] = {
    "gpt-4o": {
        "label": "GPT-4o",
        "description": "成熟的多模态识图模型，128K 上下文",
        "default": True,
    },
    "gpt-4.1": {
        "label": "GPT-4.1",
        "description": "指令遵循更强，约 1M 上下文",
        "default": False,
    },
    "gpt-6-sol": {
        "label": "GPT-6 Sol",
        "description": "适合复杂凭证与银行流水识别，约 1M 上下文",
        "default": False,
    },
}
DEFAULT_VISION_MODEL = "gpt-4o"
OPENAI_CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"
MAX_VISION_IMAGE_BYTES = 20 * 1024 * 1024
MAX_VISION_DOCUMENT_BYTES = 32 * 1024 * 1024
RECEIPT_DOCUMENT_TYPES = (
    "income_voucher",
    "expense_voucher",
    "payroll_voucher",
    "loan_interest_voucher",
    "tax_voucher",
)
VOUCHER_CLASSIFICATION_INSTRUCTION = (
    "Classify into exactly one of five voucher categories only when both the document type and "
    "transaction direction are supported by strong evidence. income_voucher means income from goods "
    "or services sold by the subject company. expense_voucher means goods, services, or operating "
    "expenses purchased by or charged to the subject company. payroll_voucher means salary, payroll, "
    "or employer payroll declarations. loan_interest_voucher means loans, interest, financing, or "
    "other primary banking business, excluding bank transaction statements. tax_voucher means VAT, "
    "tax returns, customs, or import/export tax documents. For faktura, kreditfaktura, kreditnota, "
    "räkning, and betalningsavi, determine transaction direction before document type. An invoice is "
    "not inherently income. A charge from the subject company to a customer is income; a charge from "
    "a supplier or another company to the subject company is expense. A Kreditfaktura/Kreditnota from "
    "the subject company to its customer reduces sales and remains income_voucher; one from a supplier "
    "to the subject company reduces purchases and remains expense_voucher. Treat kundfaktura, a sales "
    "invoice or sales receipt issued by the subject company, and POS/Z-rapport as income evidence. Treat "
    "leverantörsfaktura, purchase receipts, räkning, betalningsavi, utilities, rent, phone, and software "
    "bills charged by suppliers as expense evidence. Use Säljare, Köpare, Kund, Leverantör, "
    "Fakturamottagare, company names, organization numbers, payment details, payer/payee facts, and body "
    "text to establish direction. Never classify from a keyword alone and never assume the subject "
    "company is the seller when identity or direction is insufficient. Never return bank_voucher or "
    "uncategorized. Set classification.is_certain=true only when direction and category are unambiguous "
    "and confidence is at least 0.85; otherwise return document_type=null and "
    "status=needs_manual_confirmation."
)

PAYMENT_CANDIDATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "text": {"type": "string"},
        "amount": {"type": "string"},
        "currency": {"type": ["string", "null"]},
        "keyword": {"type": ["string", "null"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["text", "amount", "currency", "keyword", "confidence"],
}

PARTY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "name": {"type": ["string", "null"]},
        "organization_number": {"type": ["string", "null"]},
    },
    "required": ["name", "organization_number"],
}

FINANCIAL_FACTS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "amount_text": {"type": ["string", "null"]},
        "amount_decimal": {"type": ["string", "null"]},
        "currency": {"type": ["string", "null"]},
        "reference_numbers": {"type": "array", "items": {"type": "string"}},
        "account_numbers": {"type": "array", "items": {"type": "string"}},
        "transaction_time_text": {"type": ["string", "null"]},
        "transaction_time_iso": {"type": ["string", "null"]},
        "payer": PARTY_SCHEMA,
        "payee": PARTY_SCHEMA,
    },
    "required": [
        "amount_text", "amount_decimal", "currency", "reference_numbers",
        "account_numbers", "transaction_time_text", "transaction_time_iso",
        "payer", "payee",
    ],
}

CLASSIFICATION_RESULT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "document_type": {
            "type": ["string", "null"],
            "enum": [
                "income_voucher", "expense_voucher", "payroll_voucher",
                "loan_interest_voucher", "tax_voucher", None,
            ],
        },
        "is_certain": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string"},
        "evidence": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["document_type", "is_certain", "confidence", "reason", "evidence"],
}

RECEIPT_RESULT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "page_start": {"type": "integer", "minimum": 1, "maximum": 12},
        "page_end": {"type": "integer", "minimum": 1, "maximum": 12},
        "relevance": {
            "type": "string",
            "enum": ["unassessed"],
        },
        "relevance_reason": {"type": "string"},
        "document_date_text": {"type": ["string", "null"]},
        "document_date_iso": {"type": ["string", "null"]},
        "due_date_text": {"type": ["string", "null"]},
        "due_date_iso": {"type": ["string", "null"]},
        "text": {"type": "string"},
        "payment_candidates": {"type": "array", "items": PAYMENT_CANDIDATE_SCHEMA},
        "financial_facts": FINANCIAL_FACTS_SCHEMA,
        "classification": CLASSIFICATION_RESULT_SCHEMA,
    },
    "required": [
        "page_start", "page_end", "relevance", "relevance_reason",
        "document_date_text", "document_date_iso", "due_date_text", "due_date_iso",
        "text", "payment_candidates", "financial_facts", "classification",
    ],
}


class VisionSettingsError(RuntimeError):
    """Raised when protected OpenAI vision settings cannot be read or changed."""


class VisionRequestError(RuntimeError):
    def __init__(self, message: str, *, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


class VisionSettingsStore:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        if self.path.is_symlink():
            raise VisionSettingsError("OpenAI 识图配置不能是符号链接")
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise VisionSettingsError("无法读取 OpenAI 识图配置") from exc
        if not isinstance(payload, dict):
            raise VisionSettingsError("OpenAI 识图配置格式无效")
        return payload

    def status(self) -> dict[str, Any]:
        payload = self._read()
        model = str(payload.get("model") or DEFAULT_VISION_MODEL)
        if model not in SUPPORTED_VISION_MODELS:
            model = DEFAULT_VISION_MODEL
        return {
            "provider": "openai",
            "model": model,
            "configured": bool(payload.get("api_key")),
            "updated_at": payload.get("updated_at"),
            "models": [
                {"id": model_id, **metadata}
                for model_id, metadata in SUPPORTED_VISION_MODELS.items()
            ],
        }

    def save(self, *, model: str, api_key: str | None = None) -> dict[str, Any]:
        if model not in SUPPORTED_VISION_MODELS:
            raise VisionSettingsError("不支持该 OpenAI 识图模型")
        current = self._read()
        normalized_key = api_key.strip() if api_key is not None else ""
        if api_key is not None:
            if len(normalized_key) < 20 or len(normalized_key) > 512:
                raise VisionSettingsError("OpenAI API 密钥长度无效")
            if any(character.isspace() for character in normalized_key):
                raise VisionSettingsError("OpenAI API 密钥不能包含空白字符")
        else:
            normalized_key = str(current.get("api_key") or "")
        if not normalized_key:
            raise VisionSettingsError("首次配置必须填写 OpenAI API 密钥")

        payload = {
            "provider": "openai",
            "model": model,
            "api_key": normalized_key,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            if self.path.parent.is_symlink():
                raise VisionSettingsError("OpenAI 识图配置目录不能是符号链接")
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=".openai-vision-",
                suffix=".json",
                dir=self.path.parent,
                text=True,
            )
            try:
                os.fchmod(descriptor, 0o600)
                with os.fdopen(descriptor, "w", encoding="utf-8") as target:
                    json.dump(payload, target, ensure_ascii=False)
                    target.write("\n")
                    target.flush()
                    os.fsync(target.fileno())
                os.replace(temporary_name, self.path)
                os.chmod(self.path, 0o600)
            except Exception:
                try:
                    os.unlink(temporary_name)
                except OSError:
                    pass
                raise
        except VisionSettingsError:
            raise
        except OSError as exc:
            raise VisionSettingsError("无法保存 OpenAI 识图配置") from exc
        return self.status()

    def remove_key(self) -> dict[str, Any]:
        try:
            if self.path.is_symlink():
                raise VisionSettingsError("OpenAI 识图配置不能是符号链接")
            self.path.unlink(missing_ok=True)
        except OSError as exc:
            raise VisionSettingsError("无法删除 OpenAI API 密钥") from exc
        return self.status()

    def credentials(self) -> tuple[str, str]:
        payload = self._read()
        api_key = str(payload.get("api_key") or "")
        model = str(payload.get("model") or DEFAULT_VISION_MODEL)
        if not api_key:
            raise VisionRequestError("OpenAI 识图模型尚未配置 API 密钥", status_code=503)
        if model not in SUPPORTED_VISION_MODELS:
            raise VisionRequestError("OpenAI 识图模型配置无效", status_code=503)
        return model, api_key


class OpenAIVisionClient:
    def __init__(self, settings: VisionSettingsStore) -> None:
        self.settings = settings

    def recognize(
        self,
        image: bytes,
        *,
        filename: str,
        content_type: str,
        model_override: str | None = None,
        classify: bool = True,
        subject_company_name: str | None = None,
        audit_period_start: str | None = None,
        audit_period_end: str | None = None,
    ) -> dict[str, Any]:
        return self.recognize_document(
            [(image, filename, content_type, 1)],
            document_text=None,
            model_override=model_override,
            classify=classify,
            subject_company_name=subject_company_name,
            audit_period_start=audit_period_start,
            audit_period_end=audit_period_end,
        )

    def recognize_document(
        self,
        pages: list[tuple[bytes, str, str, int]],
        *,
        document_text: str | None,
        page_count_override: int | None = None,
        model_override: str | None = None,
        classify: bool = True,
        subject_company_name: str | None = None,
        audit_period_start: str | None = None,
        audit_period_end: str | None = None,
    ) -> dict[str, Any]:
        page_count = page_count_override or len(pages)
        if page_count < 1 or page_count > 12:
            raise VisionRequestError("文档页数必须介于 1 到 12 页", status_code=400)
        if not pages and not (document_text or "").strip():
            raise VisionRequestError("文档不包含可识别文字或页面", status_code=400)
        if pages and page_count_override is not None and page_count_override != len(pages):
            raise VisionRequestError("渲染页面数量与文档页数不一致", status_code=400)
        if len(pages) > 12:
            raise VisionRequestError("文档最多支持 12 页", status_code=413)
        total_bytes = 0
        for image, _, content_type, page_number in pages:
            if content_type not in {"image/jpeg", "image/png", "image/webp"}:
                raise VisionRequestError(
                    f"第 {page_number} 页不是支持的 JPEG、PNG 或 WebP 图片",
                    status_code=415,
                )
            if not image or len(image) > MAX_VISION_IMAGE_BYTES:
                raise VisionRequestError(
                    f"第 {page_number} 页图片必须介于 1 字节和 20 MiB 之间",
                    status_code=413,
                )
            total_bytes += len(image)
        if total_bytes > MAX_VISION_DOCUMENT_BYTES:
            raise VisionRequestError("文档渲染页面总大小超过 32 MiB", status_code=413)
        configured_model, api_key = self.settings.credentials()
        model = model_override or configured_model
        if model not in SUPPORTED_VISION_MODELS:
            raise VisionRequestError("不支持该 OpenAI 识图模型", status_code=422)
        classification_instruction = (
            VOUCHER_CLASSIFICATION_INSTRUCTION
            if classify
            else "Do not classify this document. Set classification.document_type=null, "
            "classification.is_certain=false, classification.confidence=0, explain that local "
            "classification was requested in classification.reason, and use an empty evidence "
            "array. Determine status only from whether the image can be read reliably."
        )
        user_content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "Analyze the ordered PDF page content as a document package that may contain multiple distinct "
                    "invoices, receipts, credit notes, reminders, or unrelated historical documents. Split "
                    "every distinct financial document into one receipts item and give its inclusive page "
                    "range. Every identified receipt must receive its own complete financial facts and "
                    "classification; never collapse multiple receipts into one result. Do not treat "
                    "advertisements, generic terms, legal boilerplate, navigation, "
                    "repeated headers/footers, or long transaction-history appendices as separate receipts. "
                    "For each receipt, output only a compact financial summary of at most 500 "
                    "characters: issuer and recipient, document/invoice/reference number, document date, "
                    "due date, final paid or payable total, currency, payment account/OCR/reference, and any "
                    "line necessary to justify classification. Financial facts must contain that receipt's "
                    "final payable or paid amount whenever it is visible. For payroll documents, use the net "
                    "final payment labelled Nettolön, Netto lön, Att utbetala, Utbetalas, Utbetalt, or the "
                    "equivalent—not Bruttolön or Månadslön—and include at most three amount candidates per "
                    "receipt. Set relevance=unassessed and "
                    "relevance_reason='deferred_to_audit' for every financial receipt. Do not decide audit-period, "
                    "company-ownership or duplicate relevance during recognition. "
                    "Do not transcribe advertisements, "
                    "terms, policies, explanatory prose, or full transaction tables. "
                    + (
                        "The PDF text layer passed below has already passed local quality checks. Treat its page "
                        "markers and text as the complete source; no rendered images are supplied or needed. "
                        if not pages else
                        "Inspect every supplied rendered page when the PDF text layer is incomplete or unreliable. "
                    )
                    + "Preserve visible masked account/card "
                    "characters exactly and never merge unrelated rows or amounts. "
                    "The service-side subject company name is "
                    + json.dumps((subject_company_name or "").strip()[:255], ensure_ascii=False)
                    + ". Treat company identity as classification context only when it is specific and visibly matches. "
                    "Select one 1-based primary_receipt_index as a backward-compatible representative receipt. "
                    "This does not make other receipts secondary or optional: every receipt must still be extracted "
                    "and classified independently. Top-level text must be "
                    "a package summary under 500 characters and must not repeat receipt excerpts; top-level "
                    "financial_facts and classification must describe only the selected primary receipt. "
                    + classification_instruction
                    + " Use needs_reupload only when page quality prevents reliable reading. Return only "
                    "data matching the supplied JSON schema."
                ),
            }
        ]
        if document_text:
            layout_guidance = (
                "Use rendered pages to resolve columns, tables and visual conflicts; "
                if pages else
                "No rendered pages are supplied because this text passed local quality checks; "
            )
            user_content.append(
                {
                    "type": "text",
                    "text": (
                        "This layout-preserving text came directly from the PDF text layer. Page markers "
                        "and order are authoritative. "
                        + layout_guidance
                        + "do not flatten adjacent columns into one row:\n\n"
                        + document_text[:100_000]
                    ),
                }
            )
        for image, filename, content_type, page_number in pages:
            encoded = base64.b64encode(image).decode("ascii")
            user_content.extend(
                [
                    {
                        "type": "text",
                        "text": f"Rendered page {page_number}/{len(pages)} ({filename}).",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{content_type};base64,{encoded}",
                            "detail": "high",
                        },
                    },
                ]
            )
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": user_content,
                }
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "receipt_vision_result",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "status": {
                                "type": "string",
                                "enum": ["accepted", "needs_manual_confirmation", "needs_reupload"],
                            },
                            "reasons": {"type": "array", "items": {"type": "string"}},
                            "primary_receipt_index": {"type": ["integer", "null"], "minimum": 1},
                            "text": {"type": "string"},
                            "receipts": {
                                "type": "array", "minItems": 1, "maxItems": 36,
                                "items": RECEIPT_RESULT_SCHEMA,
                            },
                            "financial_facts": FINANCIAL_FACTS_SCHEMA,
                            "classification": CLASSIFICATION_RESULT_SCHEMA,
                        },
                        "required": [
                            "status", "reasons", "primary_receipt_index", "text", "receipts",
                            "financial_facts", "classification",
                        ],
                    },
                },
            },
        }
        if model == "gpt-6-sol":
            payload["reasoning_effort"] = "none"
            payload["temperature"] = 0
            payload["max_completion_tokens"] = 8192
        else:
            payload["temperature"] = 0
            payload["max_tokens"] = 8192
        request = Request(
            OPENAI_CHAT_COMPLETIONS_URL,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urlopen(request, timeout=180) as response:
                raw = response.read()
        except HTTPError as exc:
            messages = {
                400: "OpenAI 拒绝了图片或请求参数",
                401: "OpenAI API 密钥无效或没有模型访问权限",
                403: "OpenAI API 账户无权调用该模型",
                404: "所选 OpenAI 模型当前不可用",
                413: "图片超过 OpenAI 接口限制",
                429: "OpenAI API 已达到速率或额度限制",
            }
            raise VisionRequestError(
                messages.get(exc.code, "OpenAI 识图服务返回错误"),
                status_code=exc.code if exc.code in messages else 502,
            ) from exc
        except (URLError, TimeoutError) as exc:
            raise VisionRequestError("无法连接 OpenAI 识图服务", status_code=504) from exc

        try:
            response_payload = json.loads(raw.decode("utf-8"))
            content = response_payload["choices"][0]["message"]["content"]
            if isinstance(content, list):
                content = "".join(
                    str(item.get("text") or "") for item in content if isinstance(item, dict)
                )
            result = json.loads(str(content))
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise VisionRequestError("OpenAI 返回了无效的识图结果") from exc
        if not isinstance(result, dict):
            raise VisionRequestError("OpenAI 返回了无效的识图结果")

        def normalize_candidates(raw_candidates: Any) -> list[dict[str, Any]]:
            normalized: list[dict[str, Any]] = []
            if not isinstance(raw_candidates, list):
                return normalized
            for index, candidate in enumerate(raw_candidates):
                if not isinstance(candidate, dict) or not candidate.get("amount"):
                    continue
                currency = str(candidate.get("currency") or "").strip().casefold()
                keyword = str(candidate.get("keyword") or "total").strip().casefold()
                try:
                    confidence = max(0.0, min(float(candidate.get("confidence") or 0), 1.0))
                except (TypeError, ValueError, OverflowError):
                    confidence = 0.0
                normalized.append(
                    {
                        "line_index": index,
                        "text": str(candidate.get("text") or "")[:1000],
                        "amounts": [str(candidate["amount"])[:100]],
                        "keywords": [keyword[:100]] if keyword else [],
                        "currencies": [currency[:20]] if currency else [],
                        "confidence": confidence,
                    }
                )
            return normalized

        raw_receipts = result.get("receipts")
        uses_receipt_schema = isinstance(raw_receipts, list) and bool(raw_receipts)
        if not isinstance(raw_receipts, list) or not raw_receipts:
            raw_receipts = [{
                "page_start": 1,
                "page_end": page_count,
                "relevance": "unassessed",
                "relevance_reason": "deferred_to_audit",
                "document_date_text": None,
                "document_date_iso": None,
                "due_date_text": None,
                "due_date_iso": None,
                "text": result.get("text") or "",
                "payment_candidates": result.get("payment_candidates") or [],
                "financial_facts": result.get("financial_facts") or {},
                "classification": result.get("classification") or {},
            }]
        receipts: list[dict[str, Any]] = []
        for index, raw_receipt in enumerate(raw_receipts, start=1):
            if not isinstance(raw_receipt, dict):
                continue
            try:
                page_start = max(1, min(int(raw_receipt.get("page_start") or 1), page_count))
                page_end = max(page_start, min(int(raw_receipt.get("page_end") or page_start), page_count))
            except (TypeError, ValueError, OverflowError):
                page_start = page_end = 1
            receipt_facts = raw_receipt.get("financial_facts")
            receipt_classification = raw_receipt.get("classification")
            normalized_receipt = {
                "index": index,
                "page_start": page_start,
                "page_end": page_end,
                "relevance": "unassessed",
                "relevance_reason": "deferred_to_audit",
                "document_date_text": raw_receipt.get("document_date_text"),
                "document_date_iso": raw_receipt.get("document_date_iso"),
                "due_date_text": raw_receipt.get("due_date_text"),
                "due_date_iso": raw_receipt.get("due_date_iso"),
                "text": str(raw_receipt.get("text") or "")[:500],
                "lines": [],
                "payment_candidates": normalize_candidates(
                    raw_receipt.get("payment_candidates")
                )[:3],
                "financial_facts": receipt_facts if isinstance(receipt_facts, dict) else {},
                "classification": (
                    receipt_classification if isinstance(receipt_classification, dict) else {}
                ),
            }
            classification_type = str(
                normalized_receipt["classification"].get("document_type") or ""
            )
            normalized_facts = normalized_receipt["financial_facts"]
            has_amount = bool(str(normalized_facts.get("amount_decimal") or "").strip())
            extraction_complete = classification_type != "payroll_voucher" or has_amount
            normalized_receipt["financial_extraction_complete"] = extraction_complete
            normalized_receipt["financial_extraction_issue"] = (
                None if extraction_complete else "payroll_final_amount_missing"
            )
            receipts.append(normalized_receipt)
        requested_primary = result.get("primary_receipt_index")
        primary = next(
            (
                receipt for receipt in receipts
                if receipt["index"] == requested_primary
                and receipt["relevance"] in {"unassessed", "included"}
            ),
            None,
        )
        if primary is None:
            primary = next(
                (
                    receipt for receipt in receipts
                    if receipt["relevance"] in {"unassessed", "included"}
                ),
                None,
            )
        status = str(result.get("status") or "needs_manual_confirmation")
        if status not in {"accepted", "needs_manual_confirmation", "needs_reupload"}:
            status = "needs_manual_confirmation"
        all_receipts_excluded = bool(receipts) and all(
            receipt["relevance"] == "excluded" for receipt in receipts
        )
        if all_receipts_excluded:
            status = "accepted"
        elif primary is None and status == "accepted":
            status = "needs_manual_confirmation"
        response_model = str(response_payload.get("model") or model)
        financial_facts = (
            primary.get("financial_facts")
            if primary else ({} if uses_receipt_schema else result.get("financial_facts"))
        )
        if not isinstance(financial_facts, dict):
            financial_facts = {}
        classification = (
            primary.get("classification")
            if primary else ({} if uses_receipt_schema else result.get("classification"))
        )
        if not isinstance(classification, dict):
            classification = {}
        reasons = [str(value) for value in (result.get("reasons") or [])]
        if primary is None and receipts and not all_receipts_excluded:
            reasons.append("没有可自动纳入当前审计期间的主票据")
        elif all_receipts_excluded:
            reasons.append("所有票据均已明确排除，无需选择主票据")
        return {
            "request_id": str(response_payload.get("id") or "openai-vision"),
            "status": status,
            "reasons": reasons,
            "model": {"provider": "openai", "vision": response_model},
            "image": {
                "filename": pages[0][1] if pages else "pdf-text-layer",
                "content_type": pages[0][2] if pages else "application/pdf",
                "size_bytes": sum(len(page[0]) for page in pages),
                "page_count": page_count,
                "recognition_mode": "rendered_pages" if pages else "text_layer",
            },
            "primary_receipt_index": primary.get("index") if primary else None,
            "receipts": receipts,
            "financial_facts": financial_facts,
            "classification": classification,
            "processing_ms": round((time.perf_counter() - started) * 1000, 2),
            "usage": response_payload.get("usage") or {},
        }
