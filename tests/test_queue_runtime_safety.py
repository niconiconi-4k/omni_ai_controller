"""Offline regression coverage for terminal cleanup and integration contracts."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from email.utils import format_datetime
from io import BytesIO
from threading import Event
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import HTTPError, URLError

import httpx
import pytest

from omni_ai_controller.client import ModelServerClient, ServerRequestError
from omni_ai_controller.model_queue import CURRENT_QUEUES, REQUEST_CANCELLED, ModelQueues, QueueSettings, run_model_call
from omni_ai_controller.model_transport import _retry_after
from omni_ai_controller.request_errors import ModelCallCancelled
from omni_ai_controller.vision import VisionRequestError, normalize_user_supplement
from test_client import config
from test_model_queue import stalled_peer, submit
from test_service import FakeController, client, headers


AUDIT_CONTEXT = {
    "transactions": [{"id": "tx-1"}], "receipts": [{"id": "receipt-1"}],
    "deterministic_candidates": [{"transaction_id": "tx-1", "receipt_upload_id": "receipt-1",
                                  "allocation_role": "direct_expense"}],
}


@pytest.mark.parametrize("stream", [False, True])
def test_explicit_cancellation_prevents_json_submission(stream):
    queues = ModelQueues()
    token = CURRENT_QUEUES.set(queues)
    try:
        with patch("omni_ai_controller.client.urlopen") as opener:
            with pytest.raises(ModelCallCancelled):
                ModelServerClient(config()).chat_json([], schema_name="test", schema={},
                    stream=stream, cancelled=lambda: True)
            opener.assert_not_called()
        assert queues.local.counts()["running"] == 0
    finally:
        CURRENT_QUEUES.reset(token)


def test_nonstream_json_explicit_cancellation_interrupts_stalled_headers(stalled_peer):
    from dataclasses import replace
    url, entered = stalled_peer
    queues = ModelQueues()
    token = CURRENT_QUEUES.set(queues)
    cancel = Event()
    try:
        with ThreadPoolExecutor(1) as executor:
            job = submit(executor, ModelServerClient(replace(config(), base_url=url)).chat_json,
                         [], schema_name="test", schema={}, cancelled=cancel.is_set)
            assert entered.wait(2)
            cancel.set()
            with pytest.raises(ModelCallCancelled):
                job.result(1)
        assert queues.local.counts()["running"] == 0
    finally:
        cancel.set()
        CURRENT_QUEUES.reset(token)


def test_repeated_async_cancellation_waits_for_terminal_transport_close():
    async def run():
        queues = ModelQueues()
        token = CURRENT_QUEUES.set(queues)
        entered, cancelled, release, exited = Event(), Event(), Event(), Event()
        def work():
            with queues.local.slot():
                entered.set()
                assert REQUEST_CANCELLED.get().wait(2)
                cancelled.set()
                assert release.wait(3)
            exited.set()
            raise ModelCallCancelled("offline terminal cleanup")
        task = asyncio.create_task(run_model_call(work))
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            task.cancel()
            assert await asyncio.to_thread(cancelled.wait, 2)
            task.cancel()
            await asyncio.to_thread(lambda: None)
            assert not task.done()
            assert queues.local.counts()["running"] == 1
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 1)
            assert exited.is_set() and queues.local.counts()["running"] == 0
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
            CURRENT_QUEUES.reset(token)
    asyncio.run(run())


def test_finish_reason_does_not_release_admission_before_done_and_close():
    queues = ModelQueues()
    token = CURRENT_QUEUES.set(queues)
    waiting, release = Event(), Event()
    class Stream(BytesIO):
        def readline(self, *args):
            if self.tell() > 0:
                waiting.set()
                assert release.wait(3)
            return super().readline(*args)
        def close(self):
            assert queues.local.counts()["running"] == 1
            super().close()
    stream = Stream(b'data: {"choices":[{"delta":{"content":"{}"},"finish_reason":"stop"}]}\n'
                    b'data: {"choices":[],"usage":{"total_tokens":7}}\n'
                    b'data: [DONE]\n')
    try:
        with patch("omni_ai_controller.client.urlopen", return_value=stream), ThreadPoolExecutor(1) as executor:
            job = submit(executor, ModelServerClient(config()).chat_json, [], schema_name="test", schema={}, stream=True)
            assert waiting.wait(2)
            assert queues.local.counts()["running"] == 1 and not job.done()
            release.set()
            assert job.result(2).raw["usage"] == {"total_tokens": 7}
        assert stream.closed and queues.local.counts()["running"] == 0
    finally:
        release.set()
        CURRENT_QUEUES.reset(token)


@pytest.mark.parametrize("endpoint,channel,patch_target,body", [
    ("/internal/support/chat", "local", "client", {"messages": [{"role": "user", "content": "private prompt"}]}),
    ("/internal/quantization/classify", "local", "client", {"text": "private prompt"}),
    ("/internal/audit/agentic", "local", "client", {"audit_id": "retry-test", "context": AUDIT_CONTEXT}),
    ("/internal/vision/receipts", "external", "vision", {"source_kind": "text", "document_text": "private prompt"}),
    ("/internal/vision/bank-statements", "external", "bank_statement", {
        "filename": "bank.csv", "source_kind": "spreadsheet", "document_text": "private prompt"}),
])
@pytest.mark.parametrize("status", [429, 503])
def test_upstream_retry_status_is_preserved_and_error_body_closed_without_leaking(
    endpoint, channel, patch_target, body, status,
):
    from omni_ai_controller.vision import OpenAIVisionClient
    from omni_ai_controller.bank_statement import OpenAIBankStatementClient
    store = SimpleNamespace(credentials=lambda: ("gpt-6.1-sol", "offline-test-secret"))
    controller = FakeController()
    controller.model_server.client = ModelServerClient(config())
    secret_body = BytesIO(b'{"error":"Bearer offline-test-secret private prompt"}')
    error = HTTPError("https://offline.invalid/secret", status, "private reason",
                      {"Retry-After": "17", "Authorization": "private token"}, secret_body)
    with patch("test_service.FakeVisionClient", return_value=OpenAIVisionClient(store)), client(
        controller=controller, statement_client=OpenAIBankStatementClient(store),
    ) as tc:
        with patch(f"omni_ai_controller.{patch_target}.urlopen", side_effect=error) as opener:
            result = tc.post(endpoint, headers={"X-Vision-Token": "internal-vision-token"}, json=body)
        assert result.status_code == status
        assert result.headers["Retry-After"] == "17"
        assert result.json()["code"] == ("model_upstream_rate_limited" if status == 429 else "model_upstream_unavailable")
        assert secret_body.closed and opener.call_count == 1
        assert not any(value in result.text for value in ("offline-test-secret", "private prompt", "private reason", "private token"))
        assert tc.app.state.model_queues.status()["channels"][channel]["running"] == 0


@pytest.mark.parametrize("value,expected", [(None, "1"), ("17", "17"), ("99999", "3600"),
    ("0", "1"), ("Bearer secret\r\nX-Secret: secret", "1"), ("x" * 129, "1")])
def test_retry_after_only_forwards_bounded_seconds(value, expected):
    assert _retry_after(value) == expected


def test_retry_after_http_date_becomes_safe_seconds():
    with patch("omni_ai_controller.model_transport.time", return_value=1000):
        assert _retry_after(format_datetime(datetime.fromtimestamp(1017, timezone.utc), usegmt=True)) == "17"


@pytest.mark.parametrize("error", [HTTPError("http://offline.invalid", 400, "secret", {}, BytesIO(b"secret-api-key")),
                                   URLError("secret-api-key")])
def test_local_model_errors_never_expose_upstream_body_or_reason(error):
    with patch("omni_ai_controller.client.urlopen", side_effect=error):
        with pytest.raises(ServerRequestError) as failure:
            ModelServerClient(config()).chat([], enable_thinking=False)
    assert "secret" not in str(failure.value)


def test_agentic_http_cancellation_does_not_detach_worker_cleanup():
    entered, cancelled, release, exited = Event(), Event(), Event(), Event()
    def analyze(*args, **kwargs):
        with CURRENT_QUEUES.get().local.slot():
            entered.set()
            assert REQUEST_CANCELLED.get().wait(2)
            cancelled.set()
            assert release.wait(3)
        exited.set()
        return {"status": "cancelled"}
    with client() as tc, patch("omni_ai_controller.service.analyze_agentic_audit", side_effect=analyze):
        async def run():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=tc.app), base_url="http://offline") as ac:
                task = asyncio.create_task(ac.post("/internal/audit/agentic", json={"audit_id": "offline-cancel"},
                                                   headers={"X-Vision-Token": "internal-vision-token"}))
                try:
                    assert await asyncio.to_thread(entered.wait, 2)
                    task.cancel()
                    assert await asyncio.to_thread(cancelled.wait, 2)
                    assert tc.app.state.model_queues.local.counts()["running"] == 1
                    assert not task.done()
                    release.set()
                    with pytest.raises(asyncio.CancelledError):
                        await asyncio.wait_for(task, 1)
                    assert exited.is_set() and tc.app.state.model_queues.local.counts()["running"] == 0
                finally:
                    release.set()
                    await asyncio.gather(task, return_exceptions=True)
        asyncio.run(run())


def test_named_manual_text_and_misnamed_supplement_are_not_silently_lost():
    auth = {"X-Vision-Token": "internal-vision-token"}
    with client() as tc:
        body = {"filename": "manual.txt", "source_kind": "text",
                "user_supplement": {"manual_receipt": "paid 12 SEK"}}
        assert tc.post("/internal/vision/receipts", json=body, headers=auth).status_code == 200
        body["supplement"] = body.pop("user_supplement")
        result = tc.post("/internal/vision/receipts", json=body, headers=auth)
        assert result.status_code == 422 and "user_supplement" in result.text


def test_previous_projection_rejects_deep_json_and_drops_wrong_container_types():
    previous = {"usage": {}}
    child = previous["usage"]
    for _ in range(33):
        child["nested"] = {}
        child = child["nested"]
    with pytest.raises(VisionRequestError):
        normalize_user_supplement({"previous_quantization": previous})
    value = normalize_user_supplement({"previous_quantization": {
        "financial_facts": "unexpected private payload", "receipts": [{"classification": "private payload"}]}})
    assert value["previous_quantization"] == {"financial_facts": None, "receipts": [{"classification": None}]}


def test_admin_queue_status_checks_account_version_and_rejects_bearer_token():
    from test_service import FakeAdminAccountStore
    store = FakeAdminAccountStore()
    with client(store=store) as tc:
        assert tc.get("/api/model-queue", headers=headers()).status_code == 200
        store.accounts["super"]["auth_version"] = 2
        assert tc.get("/api/model-queue", headers=headers()).status_code == 401
        tc.cookies.clear()
        assert tc.get("/api/model-queue", headers={**headers(), "Authorization": "Bearer admin-token"}).status_code == 401


@pytest.mark.parametrize("endpoint,channel,body", [
    ("/internal/audit/agentic", "local", {"audit_id": "offline-full", "context": AUDIT_CONTEXT}),
    ("/chat", "local", {"messages": [{"role": "user", "content": "private prompt"}]}),
    ("/mutsu/messages", "local", {"content": "private prompt"}),
    ("/internal/vision/receipts", "external", {"source_kind": "text", "document_text": "private prompt"}),
    ("/internal/vision/bank-statements", "external", {
        "filename": "bank.csv", "source_kind": "spreadsheet", "document_text": "private prompt"}),
])
def test_local_chat_agentic_and_external_vision_cannot_bypass_full_queue(endpoint, channel, body):
    from omni_ai_controller.vision import OpenAIVisionClient
    from omni_ai_controller.bank_statement import OpenAIBankStatementClient
    store = SimpleNamespace(credentials=lambda: ("gpt-6.1-sol", "offline-test-secret"))
    controller = FakeController()
    controller.model_server.client = ModelServerClient(config())
    with patch("test_service.FakeVisionClient", return_value=OpenAIVisionClient(store)), client(
        controller=controller, statement_client=OpenAIBankStatementClient(store),
    ) as tc:
        queue = getattr(tc.app.state.model_queues, channel)
        queue.settings = QueueSettings(max_pending=0)
        with queue.slot(), patch("omni_ai_controller.client.urlopen") as local, patch(
            "omni_ai_controller.vision.urlopen",
        ) as vision, patch("omni_ai_controller.bank_statement.urlopen") as bank:
            result = tc.post(endpoint, headers={**headers(), "X-Vision-Token": "internal-vision-token"}, json=body)
        assert result.status_code == 429 and result.json()["code"] == "model_queue_full"
        assert result.headers["Retry-After"] == "1"
        assert "private prompt" not in result.text and "offline-test-secret" not in result.text
        local.assert_not_called()
        vision.assert_not_called()
        bank.assert_not_called()


def test_async_preparation_can_call_internal_local_api_through_same_middleware():
    controller = FakeController()
    controller.model_server.client = ModelServerClient(config())
    with client(controller=controller) as tc:
        @tc.app.post("/prepare-local-test")
        async def prepare():
            # No route-wide local lock: preparation itself uses no inference slot.
            await run_model_call(lambda: None)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=tc.app), base_url="http://offline") as inner:
                response = await inner.post("/internal/support/chat", json={
                    "messages": [{"role": "user", "content": "prepared"}]},
                    headers={"X-Vision-Token": "internal-vision-token"})
            return response.json()
        async def run():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=tc.app), base_url="http://offline") as ac:
                result = await asyncio.wait_for(ac.post("/prepare-local-test", headers=headers()), 1)
                assert result.status_code == 200 and result.json()["content"] == "offline"
                assert tc.app.state.model_queues.local.counts()["running"] == 0
        def opener(*args, **kwargs):
            assert tc.app.state.model_queues.local.counts()["running"] == 1
            return BytesIO(b'{"choices":[{"message":{"content":"offline"}}]}')
        with patch("omni_ai_controller.client.urlopen", side_effect=opener):
            asyncio.run(run())


def test_agentic_evidence_preparation_inherits_queue_and_cancellation_context():
    from omni_ai_controller.agentic_audit import analyze_agentic_audit
    from omni_ai_controller.audit_notebooks import AuditNotebooks
    queues, cancel = ModelQueues(), Event()
    queue_token = CURRENT_QUEUES.set(queues)
    cancel_token = REQUEST_CANCELLED.set(cancel)
    original = AuditNotebooks.prepare
    prepared = []
    def prepare(self, *args, **kwargs):
        assert CURRENT_QUEUES.get() is queues and REQUEST_CANCELLED.get() is cancel
        prepared.append(True)
        return original(self, *args, **kwargs)
    try:
        with patch.object(AuditNotebooks, "prepare", prepare):
            result = analyze_agentic_audit(FakeController().model_server.client, audit_id="offline-prep-context",
                context={"transactions": [{"id": "tx-1"}], "receipts": [{"id": "receipt-1"}],
                         "deterministic_candidates": [{"transaction_id": "tx-1", "receipt_upload_id": "receipt-1",
                                                       "allocation_role": "direct_expense"}]})
        assert prepared and result["process_mode"] == "agentic"
    finally:
        REQUEST_CANCELLED.reset(cancel_token)
        CURRENT_QUEUES.reset(queue_token)