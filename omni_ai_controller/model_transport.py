"""One gate per upstream exchange, held through body/stream close.

Synchronous consumers run in FastAPI/asyncio worker threads. A watchdog shuts
down the actual socket on cancellation/deadline, including while awaiting headers.
No detached inference thread is allowed to outlive its admission slot.
"""
from __future__ import annotations

import http.client
import math
import socket
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from email.utils import parsedate_to_datetime
from time import monotonic, time
from typing import Callable, Iterator, Any
from urllib.error import HTTPError, URLError
from urllib.request import HTTPHandler, HTTPSHandler, Request, build_opener

from .model_queue import CURRENT_QUEUES, cancellation_check
from .request_errors import ModelCallCancelled, ModelCallTimeout, ModelQueueError, ModelTransportError, ModelUpstreamError


def _retry_after(value: str | None) -> str:
    """Forward only bounded delta seconds, not arbitrary provider header text."""
    if not value or len(value) > 128:
        return "1"
    try:
        seconds = int(value) if value.strip().isascii() and value.strip().isdigit() else math.ceil(
            parsedate_to_datetime(value).timestamp() - time())
        return str(max(1, min(seconds, 3600)))
    except (ValueError, TypeError, OverflowError):
        return "1"


class _Sockets:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.sockets: list[socket.socket] = []
        self.aborted = False

    def track(self, sock: socket.socket | None) -> None:
        if sock is None:
            return
        with self.lock:
            self.sockets.append(sock)
            if self.aborted:
                self._shutdown(sock)

    @staticmethod
    def _shutdown(sock: socket.socket) -> None:
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass

    def abort(self) -> None:
        with self.lock:
            self.aborted = True
            for sock in self.sockets:
                self._shutdown(sock)


_ACTIVE_SOCKETS: ContextVar[_Sockets | None] = ContextVar("model_transport_sockets", default=None)


def urlopen(request: Request, *, timeout: float) -> Any:
    """urllib-compatible, injectable opener with tracked HTTP(S) sockets."""
    sockets = _ACTIVE_SOCKETS.get()

    class HTTPConnection(http.client.HTTPConnection):
        def connect(self) -> None:
            super().connect()
            if sockets:
                sockets.track(self.sock)

    class HTTPSConnection(http.client.HTTPSConnection):
        def connect(self) -> None:
            # Track before TLS handshake, which itself may wait on network I/O.
            http.client.HTTPConnection.connect(self)
            if sockets:
                sockets.track(self.sock)
            self.sock = self._context.wrap_socket(
                self.sock, server_hostname=self._tunnel_host or self.host,
                do_handshake_on_connect=False,
            )
            if sockets:
                sockets.track(self.sock)
            self.sock.do_handshake()

    class HTTP(HTTPHandler):
        def http_open(self, req: Request) -> Any:
            return self.do_open(HTTPConnection, req)

    class HTTPS(HTTPSHandler):
        def https_open(self, req: Request) -> Any:
            return self.do_open(HTTPSConnection, req, context=self._context)

    return build_opener(HTTP(), HTTPS()).open(request, timeout=timeout)


@contextmanager
def model_response(channel: str, request: Request, *, timeout: float,
                   opener: Callable[..., Any] = urlopen,
                   cancelled: Callable[[], bool] | None = None) -> Iterator[Any]:
    queue = getattr(CURRENT_QUEUES.get(), channel)
    check = cancellation_check(cancelled)
    deadline = monotonic() + timeout
    with queue.slot(cancelled=check, timeout=timeout):
        sockets = _Sockets()
        token = _ACTIVE_SOCKETS.set(sockets)
        done = threading.Event()
        failure: list[ModelQueueError] = []

        def watch() -> None:
            while not done.wait(0.05):
                if queue.stopping.is_set():
                    failure.append(ModelQueueError("模型服务正在关闭；请求未完成，请重试"))
                elif check():
                    failure.append(ModelCallCancelled("模型请求已取消；已关闭上游连接"))
                elif monotonic() >= deadline:
                    failure.append(ModelCallTimeout("模型请求达到总时间预算；已关闭上游连接"))
                else:
                    continue
                sockets.abort()
                return

        watcher = threading.Thread(target=watch, name="model-transport-watch", daemon=True)
        watcher.start()
        try:
            if check():
                raise ModelCallCancelled("模型请求已取消")
            if monotonic() >= deadline:
                raise ModelCallTimeout("模型请求达到总时间预算；未提交上游请求")
            # The watchdog includes queue wait in the total request budget.
            with opener(request, timeout=timeout) as response:
                sock = getattr(getattr(getattr(response, "fp", None), "raw", None), "_sock", None)
                sockets.track(sock)
                yield response
            if failure:
                raise failure[0]
            if check():
                raise ModelCallCancelled("模型请求已取消；不应用响应")
            if queue.stopping.is_set():
                raise ModelQueueError("模型服务正在关闭；请求未完成，请重试")
            if monotonic() >= deadline:
                raise ModelCallTimeout("模型请求达到总时间预算")
        except Exception as exc:
            if isinstance(exc, HTTPError):
                # urllib raises before entering its response context. Close the
                # error response before releasing admission; never read its body.
                exc.close()
            if failure:
                raise failure[0] from None
            if isinstance(exc, HTTPError) and exc.code in {429, 503}:
                raise ModelUpstreamError(exc.code, _retry_after(exc.headers.get("Retry-After")
                                         if exc.headers else None)) from None
            if isinstance(exc, TimeoutError) or (isinstance(exc, URLError) and isinstance(exc.reason, TimeoutError)):
                raise ModelCallTimeout("模型请求超时；未应用部分输出") from exc
            # Preserve HTTP/URL errors for the existing provider-specific handlers.
            if not isinstance(exc, URLError) and isinstance(exc, (OSError, http.client.HTTPException)):
                raise ModelTransportError("模型上游连接中断；未应用部分输出") from exc
            raise
        finally:
            done.set()
            watcher.join()
            _ACTIVE_SOCKETS.reset(token)