"""Only mock transports or loopback fake HTTP peers: never contact a model."""
import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from threading import Event, Thread
from time import monotonic
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest

from omni_ai_controller.client import ModelServerClient
from omni_ai_controller.model_queue import CURRENT_QUEUES, REQUEST_CANCELLED, ModelQueue, ModelQueues, QueueSettings
from omni_ai_controller.request_errors import (
    ModelCallCancelled, ModelCallTimeout, ModelQueueError, ModelQueueFull, ModelQueueTimeout,
)
from test_client import config


def until(predicate, timeout=3):
    deadline = monotonic() + timeout
    while not predicate():
        assert monotonic() < deadline, "barrier timed out"
        Event().wait(0.002)


def submit(executor, function, *args, **kwargs):
    return executor.submit(copy_context().run, function, *args, **kwargs)


@pytest.mark.parametrize("capacity", [0, 5, 100])
def test_conservative_capacity_bounds(capacity):
    with pytest.raises(ValueError):
        QueueSettings(capacity=capacity)


@pytest.mark.parametrize("values", [{"max_pending": -1}, {"max_pending": 257},
                                      {"wait_timeout": 0}, {"wait_timeout": float("inf")},
                                      {"wait_timeout": float("nan")}])
def test_queue_config_is_finite_and_bounded(values):
    with pytest.raises(ValueError):
        QueueSettings(**values)


def test_fifo_full_cancellation_and_counts():
    queue = ModelQueue(QueueSettings(max_pending=2))
    order = []
    cancel = Event()
    def work(index, cancelled=None):
        with queue.slot(cancelled=cancelled):
            order.append(index)
    with ThreadPoolExecutor(3) as executor:
        with queue.slot():
            first = executor.submit(work, 1, cancel.is_set)
            until(lambda: queue.counts()["queued"] == 1)
            second = executor.submit(work, 2)
            until(lambda: queue.counts()["queued"] == 2)
            assert queue.counts() == {"queued": 2, "running": 1, "capacity": 1, "max_pending": 2}
            with pytest.raises(ModelQueueFull):
                with queue.slot():
                    pytest.fail("full queue admitted")
            cancel.set()
            with pytest.raises(ModelCallCancelled):
                first.result(2)
            third = executor.submit(work, 3)
            until(lambda: queue.counts()["queued"] == 2)
        second.result(2)
        third.result(2)
    assert order == [2, 3]
    assert queue.counts()["queued"] == queue.counts()["running"] == 0


def test_wait_timeout_shutdown_and_zero_pending():
    queue = ModelQueue(QueueSettings(wait_timeout=0.05))
    with queue.slot():
        with pytest.raises(ModelQueueTimeout, match="等待超时"):
            with queue.slot():
                pytest.fail("timeout admitted")
        assert queue.counts()["queued"] == 0
        with ThreadPoolExecutor(1) as executor:
            pending = executor.submit(lambda: queue.slot().__enter__())
            until(lambda: queue.counts()["queued"] == 1)
            queue.close()
            with pytest.raises(ModelQueueError, match="已关闭"):
                pending.result(2)
    assert queue.counts()["running"] == 0
    no_wait = ModelQueue(QueueSettings(max_pending=0))
    with no_wait.slot():
        with pytest.raises(ModelQueueFull):
            with no_wait.slot():
                pass


def test_capacity_four_is_an_actual_concurrency_bound():
    queue = ModelQueue(QueueSettings(capacity=4))
    release = Event()
    def run():
        with queue.slot():
            assert release.wait(3)
    with ThreadPoolExecutor(5) as executor:
        jobs = [executor.submit(run) for _ in range(5)]
        until(lambda: queue.counts()["running"] == 4 and queue.counts()["queued"] == 1)
        release.set()
        for job in jobs:
            job.result(3)
    assert queue.counts()["running"] == 0


def test_two_clients_share_fifo_stream_holds_slot_and_external_is_independent():
    queues = ModelQueues()
    token = CURRENT_QUEUES.set(queues)
    entered, release = Event(), Event()
    order = []
    class Stream(BytesIO):
        def readline(self, *args):
            entered.set()
            assert release.wait(3)
            return super().readline(*args)
    stream = Stream(b'data: {"choices":[{"delta":{"content":"{}"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n')
    class Response(BytesIO):
        pass
    def opener(request, **kwargs):
        payload = json.loads(request.data)
        order.append(payload.get("stream"))
        return stream if payload.get("stream") else Response(b'{"choices":[{"message":{"content":"{}"}}]}')
    try:
        with patch("omni_ai_controller.client.urlopen", side_effect=opener), ThreadPoolExecutor(2) as executor:
            first = submit(executor, ModelServerClient(config()).chat_json,
                           [], schema_name="test", schema={}, stream=True)
            assert entered.wait(2)
            second = submit(executor, ModelServerClient(config()).chat, [], enable_thinking=False)
            until(lambda: queues.local.counts()["queued"] == 1)
            with queues.external.slot():
                assert queues.external.counts()["running"] == 1
                assert queues.local.counts()["running"] == 1
            assert order == [True]
            release.set()
            first.result(3)
            second.result(3)
        assert order == [True, False]
        assert stream.closed
        assert queues.local.counts()["running"] == 0
    finally:
        release.set()
        CURRENT_QUEUES.reset(token)


@pytest.fixture
def stalled_peer():
    entered, finish = Event(), Event()
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            if self.path.endswith("/stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.write(b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n')
                self.wfile.flush()
            entered.set()
            finish.wait(5)
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", entered
    finally:
        finish.set()
        server.shutdown()
        server.server_close()
        thread.join(2)


@pytest.mark.parametrize("stream", [False, True])
def test_cancellation_interrupts_stalled_headers_or_stream_and_releases_slot(stalled_peer, stream):
    from urllib.request import Request
    from omni_ai_controller.model_transport import model_response
    url, entered = stalled_peer
    queues = ModelQueues()
    cancelled = Event()
    token = CURRENT_QUEUES.set(queues)
    def request():
        with model_response("local", Request(url + ("/stream" if stream else "/headers"), data=b"{}"),
                            timeout=5, cancelled=cancelled.is_set) as response:
            response.read()
    try:
        with ThreadPoolExecutor(1) as executor:
            pending = submit(executor, request)
            assert entered.wait(2)
            cancelled.set()
            with pytest.raises(ModelCallCancelled):
                pending.result(1)
        assert queues.local.counts()["running"] == 0
        with queues.local.slot():
            pass
    finally:
        CURRENT_QUEUES.reset(token)


def test_total_deadline_interrupts_silent_upstream(stalled_peer):
    from urllib.request import Request
    from omni_ai_controller.model_transport import model_response
    url, _ = stalled_peer
    queues = ModelQueues()
    token = CURRENT_QUEUES.set(queues)
    try:
        with pytest.raises(ModelCallTimeout):
            with model_response("external", Request(url, data=b"{}"), timeout=0.1) as response:
                response.read()
        assert queues.external.counts()["running"] == 0
    finally:
        CURRENT_QUEUES.reset(token)


def test_async_client_does_not_block_event_loop_and_cancellation_cleans_up(stalled_peer):
    url, entered = stalled_peer
    async def run():
        queues = ModelQueues()
        token = CURRENT_QUEUES.set(queues)
        try:
            client = ModelServerClient(replace(config(), base_url=url))
            task = asyncio.create_task(client.async_chat([], enable_thinking=False))
            assert await asyncio.to_thread(entered.wait, 2)
            tick = Event()
            asyncio.get_running_loop().call_soon(tick.set)
            assert await asyncio.to_thread(tick.wait, 0.2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 1)
            assert queues.local.counts()["running"] == 0
        finally:
            CURRENT_QUEUES.reset(token)
    asyncio.run(run())


def test_http_cross_requests_local_fifo_status_and_full_error_are_shared():
    from test_service import client, FakeController
    from test_client import FakeResponse
    entered, release = Event(), Event()
    order = []
    class SlowResponse(FakeResponse):
        def read(self):
            entered.set()
            assert release.wait(3)
            return super().read()
    def opener(request, **kwargs):
        order.append(json.loads(request.data)["messages"][-1]["content"])
        if len(order) == 1:
            return SlowResponse({"choices": [{"message": {"content": "done"}}]})
        return FakeResponse({"choices": [{"message": {"content": json.dumps({
            "document_type": "expense_voucher", "is_certain": True, "confidence": 0.95,
            "reason": "fake classification", "evidence": [],
        })}}]})
    controller = FakeController()
    controller.model_server.client = ModelServerClient(config())
    with client(controller=controller) as tc:
        app = tc.app
        app.state.model_queues.local.settings = QueueSettings(max_pending=1)
        async def run():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="https://testserver") as ac:
                auth = {"X-Vision-Token": "internal-vision-token"}
                first = asyncio.create_task(ac.post("/internal/support/chat", headers=auth,
                                                   json={"messages": [{"role": "user", "content": "first"}]}))
                assert await asyncio.to_thread(entered.wait, 2)
                second = asyncio.create_task(ac.post("/internal/quantization/classify", headers=auth, json={"text": "second"}))
                await asyncio.to_thread(until, lambda: app.state.model_queues.local.counts()["queued"] == 1)
                counts = await ac.get("/internal/model-queue/status", headers=auth)
                assert counts.status_code == 200
                assert counts.json()["channels"]["local"]["running"] == 1
                assert counts.json()["channels"]["local"]["queued"] == 1
                full = await ac.post("/internal/support/chat", headers=auth,
                                     json={"messages": [{"role": "user", "content": "overflow-secret"}]} )
                assert full.status_code == 429
                assert full.json()["code"] == "model_queue_full"
                assert "overflow-secret" not in full.text + counts.text
                release.set()
                assert (await first).status_code == 200
                assert (await second).status_code == 200
                assert app.state.model_queues.local.counts()["running"] == 0
        try:
            with patch("omni_ai_controller.client.urlopen", side_effect=opener):
                asyncio.run(run())
        finally:
            release.set()
    assert len(order) == 2 and order[0] == "first"


def test_asgi_disconnect_cancels_waiting_request_without_transport_call():
    from omni_ai_controller.model_queue_middleware import ModelQueueMiddleware
    queues = ModelQueues()
    token = CURRENT_QUEUES.set(queues)
    async def run():
        incoming = asyncio.Queue()
        await incoming.put({"type": "http.request", "body": b"", "more_body": False})
        exited = Event()
        def call():
            try:
                with queues.local.slot(cancelled=REQUEST_CANCELLED.get().is_set):
                    pytest.fail("disconnected waiter admitted")
            finally:
                exited.set()
        async def endpoint(scope, receive, send):
            await receive()
            with pytest.raises(ModelCallCancelled):
                await asyncio.to_thread(call)
        async def send(message):
            pass
        task = asyncio.create_task(ModelQueueMiddleware(endpoint, queues=queues)(
            {"type": "http", "path": "/chat"}, incoming.get, send))
        await asyncio.to_thread(until, lambda: queues.local.counts()["queued"] == 1)
        await incoming.put({"type": "http.disconnect"})
        await asyncio.wait_for(task, 1)
        assert exited.is_set() and queues.local.counts()["queued"] == 0
    try:
        with queues.local.slot():
            asyncio.run(run())
    finally:
        CURRENT_QUEUES.reset(token)


def test_vision_and_bank_share_external_fifo_without_blocking_local():
    from omni_ai_controller.vision import OpenAIVisionClient
    from omni_ai_controller.bank_statement import OpenAIBankStatementClient
    from test_vision import FakeResponse
    queues = ModelQueues()
    token = CURRENT_QUEUES.set(queues)
    entered, release = Event(), Event()
    order = []
    store = SimpleNamespace(credentials=lambda: ("gpt-6.1-sol", "offline-test-key"))
    class Slow(FakeResponse):
        def read(self):
            entered.set()
            assert release.wait(3)
            return super().read()
    def open_vision(request, **kwargs):
        order.append("vision")
        return Slow({"choices": [{"message": {"content": json.dumps({"receipts": [], "page_reviews": []})}}]})
    def open_bank(request, **kwargs):
        order.append("bank")
        return FakeResponse({"choices": [{"message": {"content": json.dumps({"transactions": []})}}]})
    try:
        with patch("omni_ai_controller.vision.urlopen", side_effect=open_vision), patch(
            "omni_ai_controller.bank_statement.urlopen", side_effect=open_bank,
        ), ThreadPoolExecutor(2) as executor:
            first = submit(executor, OpenAIVisionClient(store).recognize, b"fake", filename="f.jpg", content_type="image/jpeg")
            assert entered.wait(2)
            second = submit(executor, OpenAIBankStatementClient(store).recognize, pages=[], document_text="fake rows",
                            source_kind="spreadsheet", filename="fake.csv")
            until(lambda: queues.external.counts()["queued"] == 1)
            with patch("omni_ai_controller.client.urlopen", return_value=BytesIO(b'{"choices":[{"message":{"content":"local"}}]}')):
                assert ModelServerClient(config()).chat([], enable_thinking=False).content == "local"
            assert order == ["vision"]
            release.set()
            first.result(3)
            second.result(3)
        assert order == ["vision", "bank"]
        assert queues.external.counts()["running"] == 0
    finally:
        release.set()
        CURRENT_QUEUES.reset(token)


def test_http_wait_timeout_and_shutdown_are_explicit_and_do_not_send_payload():
    from test_service import client, FakeController
    controller = FakeController()
    controller.model_server.client = ModelServerClient(config())
    with client(controller=controller) as tc:
        queue = tc.app.state.model_queues.local
        queue.settings = QueueSettings(wait_timeout=0.05)
        auth = {"X-Vision-Token": "internal-vision-token"}
        body = {"messages": [{"role": "user", "content": "never submitted"}]}
        with patch("omni_ai_controller.client.urlopen") as opener:
            with queue.slot():
                result = tc.post("/internal/support/chat", headers=auth, json=body)
            assert result.status_code == 504
            assert result.json()["code"] == "model_queue_wait_timeout"
            queue.close()
            result = tc.post("/internal/support/chat", headers=auth, json=body)
            assert result.status_code == 503
            assert result.json()["code"] == "model_queue_unavailable"
            opener.assert_not_called()


def test_settings_environment_channels_are_independent(monkeypatch):
    from omni_ai_controller.service import ServiceSettings
    monkeypatch.setenv("OMNI_MODEL_QUEUE_LOCAL_CAPACITY", "4")
    monkeypatch.setenv("OMNI_MODEL_QUEUE_EXTERNAL_MAX_PENDING", "8")
    monkeypatch.setenv("OMNI_MODEL_QUEUE_EXTERNAL_WAIT_TIMEOUT_SECONDS", "15")
    settings = ServiceSettings.from_environment()
    assert settings.local_queue.capacity == 4
    assert settings.external_queue == QueueSettings(max_pending=8, wait_timeout=15)
    monkeypatch.setenv("OMNI_MODEL_QUEUE_EXTERNAL_CAPACITY", "5")
    with pytest.raises(ValueError, match="between 1 and 4"):
        ServiceSettings.from_environment()


def test_receipt_body_budget_is_enforced_before_json_validation(monkeypatch):
    from omni_ai_controller import model_queue_middleware
    from test_service import client
    monkeypatch.setattr(model_queue_middleware, "MAX_RECEIPT_REQUEST_BYTES", 1024)
    with client() as tc:
        result = tc.post("/internal/vision/receipts", headers={"X-Vision-Token": "internal-vision-token"},
                         json={"user_supplement": {"previous_quantization": {"text": "x" * 2000}}})
        assert result.status_code == 413


def test_asgi_disconnect_interrupts_a_running_upstream(stalled_peer):
    from omni_ai_controller.model_queue_middleware import ModelQueueMiddleware
    url, entered = stalled_peer
    queues = ModelQueues()
    async def run():
        incoming = asyncio.Queue()
        await incoming.put({"type": "http.request", "body": b"", "more_body": False})
        client = ModelServerClient(replace(config(), base_url=url))
        async def endpoint(scope, receive, send):
            await receive()
            with pytest.raises(ModelCallCancelled):
                await client.async_chat([], enable_thinking=False)
        async def send(message):
            pass
        task = asyncio.create_task(ModelQueueMiddleware(endpoint, queues=queues)(
            {"type": "http", "path": "/chat"}, incoming.get, send))
        assert await asyncio.to_thread(entered.wait, 2)
        await incoming.put({"type": "http.disconnect"})
        await asyncio.wait_for(task, 1)
        assert queues.local.counts()["running"] == 0
    asyncio.run(run())


def test_health_and_control_are_not_gated_behind_inference():
    queues = ModelQueues()
    token = CURRENT_QUEUES.set(queues)
    try:
        with queues.local.slot(), patch("omni_ai_controller.client.urlopen", return_value=BytesIO(b'{}')):
            assert ModelServerClient(config()).status() == {}
            assert queues.local.counts()["queued"] == 0
    finally:
        CURRENT_QUEUES.reset(token)


def test_real_agentic_client_uses_one_gate_per_step_and_yields_to_other_requests():
    from omni_ai_controller.agentic_audit import analyze_agentic_audit
    from test_agentic_audit import _plan, _worker, _final
    queues = ModelQueues()
    token = CURRENT_QUEUES.set(queues)
    entered, release = Event(), Event()
    order = []
    class PlanStream(BytesIO):
        def readline(self, *args):
            entered.set()
            assert release.wait(3)
            return super().readline(*args)
    def opener(request, **kwargs):
        payload = json.loads(request.data)
        assert queues.local.counts()["running"] == 1
        if not payload["stream"]:
            order.append("chat")
            return BytesIO(b'{"choices":[{"message":{"content":"chat"}}]}')
        name = payload["response_format"]["json_schema"]["name"]
        order.append(name)
        content = {"li_shifu_audit_plan": _plan(), "ma_shifu_evidence_review": _worker(),
                   "li_shifu_final_assessment": _final()}[name]
        wire = b"data: " + json.dumps({"id": "fake-step", "choices": [{"delta": {"content": json.dumps(content)},
                                                                            "finish_reason": "stop"}]}).encode() + b"\n\ndata: [DONE]\n\n"
        return PlanStream(wire) if name == "li_shifu_audit_plan" else BytesIO(wire)
    try:
        with patch("omni_ai_controller.client.urlopen", side_effect=opener), ThreadPoolExecutor(2) as executor:
            audit = submit(executor, analyze_agentic_audit, ModelServerClient(config()), audit_id="fifo-audit",
                           context={"transactions": [{"id": "tx-1"}], "receipts": [{"id": "receipt-1"}],
                                    "deterministic_candidates": [{"transaction_id": "tx-1", "receipt_upload_id": "receipt-1",
                                                                  "allocation_role": "direct_expense"}]})
            assert entered.wait(2)
            chat = submit(executor, ModelServerClient(config()).chat, [], enable_thinking=False)
            until(lambda: queues.local.counts()["queued"] == 1)
            release.set()
            assert chat.result(3).content == "chat"
            assert audit.result(3)["process_mode"] == "agentic"
        assert order == ["li_shifu_audit_plan", "chat", "ma_shifu_evidence_review", "li_shifu_final_assessment"]
        assert queues.local.counts()["running"] == queues.local.counts()["queued"] == 0
    finally:
        release.set()
        CURRENT_QUEUES.reset(token)


def test_status_remains_readable_when_fastapi_sync_worker_pool_is_saturated():
    import anyio
    from test_service import client
    with client() as tc:
        async def run():
            limiter = anyio.to_thread.current_default_thread_limiter()
            original = limiter.total_tokens
            limiter.total_tokens = 1
            entered, release = Event(), Event()
            def block():
                entered.set()
                assert release.wait(3)
            worker = asyncio.create_task(anyio.to_thread.run_sync(block))
            try:
                assert await asyncio.to_thread(entered.wait, 2)
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=tc.app), base_url="https://testserver") as ac:
                    response = await asyncio.wait_for(ac.get("/internal/model-queue/status",
                        headers={"X-Vision-Token": "internal-vision-token"}), 0.5)
                    assert response.status_code == 200
                    admin = await asyncio.wait_for(ac.get("/api/model-queue", cookies=tc.cookies,
                        headers={"X-Forwarded-For": "192.168.192.10"}), 0.5)
                    assert admin.status_code == 200
            finally:
                release.set()
                await worker
                limiter.total_tokens = original
        asyncio.run(run())