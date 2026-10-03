class ServerRequestError(RuntimeError):
    """Raised when the model server cannot satisfy a request."""


class ServerRequestTimeout(ServerRequestError):
    """A bounded model computation exceeded its time budget."""


class ServerRequestCancelled(ServerRequestError):
    """The parent request stopped; do not keep consuming model capacity."""


class ModelQueueError(ServerRequestError):
    code = "model_queue_unavailable"
    status_code = 503


class ModelQueueFull(ModelQueueError):
    code = "model_queue_full"
    status_code = 429


class ModelQueueTimeout(ModelQueueError, ServerRequestTimeout):
    code = "model_queue_wait_timeout"
    status_code = 504


class ModelCallTimeout(ModelQueueError, ServerRequestTimeout):
    code = "model_request_timeout"
    status_code = 504


class ModelCallCancelled(ModelQueueError, ServerRequestCancelled):
    code = "model_request_cancelled"
    status_code = 499


class ModelTransportError(ModelQueueError):
    code = "model_transport_error"
    status_code = 502


class ModelUpstreamError(ModelQueueError):
    """Safe, retryable upstream failure; never carries provider bodies/headers."""

    def __init__(self, status_code: int, retry_after: str = "1") -> None:
        super().__init__("模型上游达到速率或额度限制；请稍后重试" if status_code == 429
                         else "模型上游暂不可用；请稍后重试")
        self.status_code = status_code
        self.code = "model_upstream_rate_limited" if status_code == 429 else "model_upstream_unavailable"
        self.retry_after = retry_after