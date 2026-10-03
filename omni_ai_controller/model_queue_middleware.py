"""ASGI cancellation context shared by all routes and their worker threads."""
import asyncio
from threading import Event

from starlette.responses import JSONResponse

from .model_queue import CURRENT_QUEUES, REQUEST_CANCELLED, ModelQueues

# Existing receipt images have a combined decoded 32 MiB budget. Include base64,
# text, JSON metadata and bounded supplements without buffering unlimited bodies.
MAX_RECEIPT_REQUEST_BYTES = 46 * 1024 * 1024


class ModelQueueMiddleware:
    def __init__(self, app, *, queues: ModelQueues):
        self.app = app
        self.queues = queues

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        cancelled = Event()
        queue_token = CURRENT_QUEUES.set(self.queues)
        cancel_token = REQUEST_CANCELLED.set(cancelled)
        messages = asyncio.Queue(maxsize=1)
        oversized = asyncio.Event()

        async def pump():
            size = 0
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    cancelled.set()
                    await messages.put(message)
                    return
                size += len(message.get("body", b""))
                if scope["path"] == "/internal/vision/receipts" and size > MAX_RECEIPT_REQUEST_BYTES:
                    oversized.set()
                    cancelled.set()
                    return
                await messages.put(message)

        reader = asyncio.create_task(pump())
        worker = asyncio.create_task(self.app(scope, messages.get, send))
        limit = asyncio.create_task(oversized.wait())
        try:
            done, _ = await asyncio.wait({worker, limit}, return_when=asyncio.FIRST_COMPLETED)
            if limit in done and oversized.is_set():
                worker.cancel()
                await asyncio.gather(worker, return_exceptions=True)
                await JSONResponse({"detail": "Vision request body exceeds the bounded limit"},
                                   status_code=413)(scope, receive, send)
            else:
                await worker
        finally:
            cancelled.set()
            for task in (reader, worker, limit):
                if not task.done():
                    task.cancel()
            await asyncio.gather(reader, worker, limit, return_exceptions=True)
            CURRENT_QUEUES.reset(queue_token)
            REQUEST_CANCELLED.reset(cancel_token)