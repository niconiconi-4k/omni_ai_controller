"""Process-local FIFO admission. Tickets contain no prompts or user identifiers."""
from __future__ import annotations

import asyncio
import math
import os
from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from threading import Condition, Event
from time import monotonic
from typing import Callable, Iterator

from .request_errors import ModelCallCancelled, ModelQueueError, ModelQueueFull, ModelQueueTimeout


@dataclass(frozen=True)
class QueueSettings:
    capacity: int = 1
    max_pending: int = 32
    wait_timeout: float = 300

    def __post_init__(self) -> None:
        if not 1 <= self.capacity <= 4:
            raise ValueError("Model queue capacity must be between 1 and 4")
        if not 0 <= self.max_pending <= 256:
            raise ValueError("Model queue max_pending must be between 0 and 256")
        if not math.isfinite(self.wait_timeout) or not 0 < self.wait_timeout <= 3600:
            raise ValueError("Model queue wait_timeout must be > 0 and <= 3600 seconds")

    @classmethod
    def from_environment(cls, channel: str) -> QueueSettings:
        prefix = f"OMNI_MODEL_QUEUE_{channel.upper()}_"
        return cls(int(os.getenv(prefix + "CAPACITY", "1")),
                   int(os.getenv(prefix + "MAX_PENDING", "32")),
                   float(os.getenv(prefix + "WAIT_TIMEOUT_SECONDS", "300")))


class ModelQueue:
    def __init__(self, settings: QueueSettings = QueueSettings()) -> None:
        self.settings = settings
        self._condition = Condition()
        self._pending: deque[object] = deque()
        self._running = 0
        self._closed = False
        self.stopping = Event()

    def counts(self) -> dict[str, int]:
        with self._condition:
            return {"queued": len(self._pending), "running": self._running,
                    "capacity": self.settings.capacity, "max_pending": self.settings.max_pending}

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self.stopping.set()
            self._condition.notify_all()

    @contextmanager
    def slot(self, *, cancelled: Callable[[], bool] | None = None,
             timeout: float | None = None) -> Iterator[None]:
        deadline = monotonic() + min(self.settings.wait_timeout, timeout or self.settings.wait_timeout)
        ticket = object()
        admitted = False
        with self._condition:
            try:
                if self._closed:
                    raise ModelQueueError("模型等待队列已关闭；请求未执行，请重试")
                if cancelled and cancelled():
                    raise ModelCallCancelled("模型请求已取消")
                if self._pending or self._running >= self.settings.capacity:
                    if len(self._pending) >= self.settings.max_pending:
                        raise ModelQueueFull("模型等待队列已满；请求未执行，请稍后重试")
                    self._pending.append(ticket)
                    while True:
                        if self._closed:
                            raise ModelQueueError("模型等待队列已关闭；请求未执行，请重试")
                        if cancelled and cancelled():
                            raise ModelCallCancelled("等待模型时请求已取消")
                        remaining = deadline - monotonic()
                        if remaining <= 0:
                            raise ModelQueueTimeout("模型 FIFO 队列等待超时；请求未执行")
                        if self._pending[0] is ticket and self._running < self.settings.capacity:
                            self._pending.popleft()
                            break
                        self._condition.wait(min(remaining, 0.05))
                self._running += 1
                admitted = True
            finally:
                if not admitted and ticket in self._pending:
                    self._pending.remove(ticket)
                self._condition.notify_all()
        try:
            yield
        finally:
            with self._condition:
                self._running -= 1
                self._condition.notify_all()


class ModelQueues:
    def __init__(self, local: QueueSettings = QueueSettings(),
                 external: QueueSettings = QueueSettings()) -> None:
        self.local = ModelQueue(local)
        self.external = ModelQueue(external)

    def status(self) -> dict[str, dict[str, dict[str, int]]]:
        return {"channels": {"external": self.external.counts(), "local": self.local.counts()}}

    def close(self) -> None:
        self.local.close()
        self.external.close()


DEFAULT_QUEUES = ModelQueues(QueueSettings.from_environment("local"), QueueSettings.from_environment("external"))
CURRENT_QUEUES: ContextVar[ModelQueues] = ContextVar("model_queues", default=DEFAULT_QUEUES)
REQUEST_CANCELLED: ContextVar[Event | None] = ContextVar("model_request_cancelled", default=None)


def cancellation_check(explicit: Callable[[], bool] | None = None) -> Callable[[], bool]:
    event = REQUEST_CANCELLED.get()
    return lambda: bool((event and event.is_set()) or (explicit and explicit()))


async def run_model_call(function, *args, **kwargs):
    """Offload synchronous transport; cancelling the coroutine aborts upstream I/O."""
    event = REQUEST_CANCELLED.get() or Event()
    token = REQUEST_CANCELLED.set(event)
    task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        event.set()
        # A second cancellation must not detach inference during cleanup.
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if not task.cancelled():
            task.exception()  # consume the terminal exception after cancellation
        raise
    finally:
        REQUEST_CANCELLED.reset(token)