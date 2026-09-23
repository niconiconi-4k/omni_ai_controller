from __future__ import annotations

import hmac
import ipaddress
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .admin import AdminCommandError, AdminController
from .admin_session import issue_admin_session, validate_admin_csrf, validate_admin_session
from .client import ServerRequestError
from .config import ConfigurationError
from .conversation_store import (
    ConversationNotFoundError,
    ConversationStore,
    ConversationStoreError,
)
from .hardware import hardware_status
from .metric_store import MetricCollector, MetricName, MetricRange, MetricStore, MetricStoreError

LOGGER = logging.getLogger("omni_ai_controller.service")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
ADMIN_SESSION_COOKIE = "omni_admin_session"
ADMIN_CSRF_COOKIE = "omni_admin_csrf"


@dataclass(frozen=True)
class ServiceSettings:
    model_dir: Path
    admin_token: str
    allowed_networks: tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...]
    allowed_containers: tuple[str, ...]
    admin_session_ttl_hours: int = 12
    database_host: str = "127.0.0.1"
    database_port: int = 15432
    database_name: str = "omni_ai"
    database_user: str = "omni_ai"
    database_password: str = ""
    metric_collection_enabled: bool = True

    @classmethod
    def from_environment(cls) -> "ServiceSettings":
        network_values = os.getenv("OMNI_ALLOWED_NETWORKS", "192.168.192.0/24")
        container_values = os.getenv(
            "OMNI_ALLOWED_CONTAINERS",
            "omni-ai-model,omni-ai-receipt-ocr,omni-ai-main-service,omni-ai-database",
        )
        try:
            networks = tuple(
                ipaddress.ip_network(item.strip(), strict=False)
                for item in network_values.split(",")
                if item.strip()
            )
        except ValueError as exc:
            raise RuntimeError(f"OMNI_ALLOWED_NETWORKS 配置无效：{exc}") from exc
        containers = tuple(
            item.strip() for item in container_values.split(",") if item.strip()
        )
        if not networks:
            raise RuntimeError("OMNI_ALLOWED_NETWORKS 不能为空")
        if not containers:
            raise RuntimeError("OMNI_ALLOWED_CONTAINERS 不能为空")
        session_ttl_hours = int(os.getenv("OMNI_ADMIN_SESSION_TTL_HOURS", "12"))
        if session_ttl_hours < 1 or session_ttl_hours > 168:
            raise RuntimeError("OMNI_ADMIN_SESSION_TTL_HOURS 必须介于 1 和 168 之间")
        return cls(
            model_dir=Path(os.getenv("OMNI_MODEL_DIR", "/opt/ai_server/omni_ai_model")),
            admin_token=os.getenv("OMNI_ADMIN_TOKEN", ""),
            allowed_networks=networks,
            allowed_containers=containers,
            admin_session_ttl_hours=session_ttl_hours,
            database_host=os.getenv("OMNI_CONVERSATION_DATABASE_HOST", "127.0.0.1"),
            database_port=int(os.getenv("OMNI_CONVERSATION_DATABASE_PORT", "15432")),
            database_name=os.getenv("DATABASE_NAME", "omni_ai"),
            database_user=os.getenv("DATABASE_USER", "omni_ai"),
            database_password=os.getenv("DATABASE_PASSWORD", ""),
            metric_collection_enabled=os.getenv(
                "OMNI_METRIC_COLLECTION_ENABLED", "true"
            ).lower()
            not in {"0", "false", "no"},
        )


class AdminLoginRequest(BaseModel):
    token: str = Field(min_length=1, max_length=1024)


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=16_000)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=32)
    enable_thinking: bool = True


class CreateConversationRequest(BaseModel):
    title: str = Field(default="新对话", min_length=1, max_length=160)
    enable_thinking: bool = True


class RenameConversationRequest(BaseModel):
    title: str = Field(min_length=1, max_length=160)


class ConversationChatRequest(BaseModel):
    content: str = Field(min_length=1, max_length=16_000)
    enable_thinking: bool = True


def _client_ip(request: Request) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    forwarded = request.headers.get("x-forwarded-for", "").split(",", 1)[0].strip()
    candidate = forwarded or (request.client.host if request.client else "")
    try:
        return ipaddress.ip_address(candidate)
    except ValueError:
        return None


def create_app(
    settings: ServiceSettings | None = None,
    controller: AdminController | None = None,
    conversation_store: ConversationStore | None = None,
    metric_store: MetricStore | None = None,
) -> FastAPI:
    active_settings = settings or ServiceSettings.from_environment()
    active_controller = controller or AdminController(
        active_settings.model_dir,
        active_settings.allowed_containers,
    )
    active_conversation_store = conversation_store or ConversationStore(
        host=active_settings.database_host,
        port=active_settings.database_port,
        database=active_settings.database_name,
        user=active_settings.database_user,
        password=active_settings.database_password,
    )
    active_metric_store = metric_store or MetricStore(
        host=active_settings.database_host,
        port=active_settings.database_port,
        database=active_settings.database_name,
        user=active_settings.database_user,
        password=active_settings.database_password,
    )
    metric_collector = MetricCollector(active_metric_store, hardware_status)

    @asynccontextmanager
    async def lifespan(_application: FastAPI):  # type: ignore[no-untyped-def]
        if active_settings.metric_collection_enabled:
            metric_collector.start()
        try:
            yield
        finally:
            if active_settings.metric_collection_enabled:
                metric_collector.stop()

    application = FastAPI(
        title="Omni AI Host Controller",
        version="1.0.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )

    @application.middleware("http")
    async def restrict_network(request: Request, call_next):  # type: ignore[no-untyped-def]
        if request.url.path == "/health/live":
            return await call_next(request)
        address = _client_ip(request)
        if address is None or not any(address in network for network in active_settings.allowed_networks):
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={"detail": "Client IP is not allowed"},
            )
        return await call_next(request)

    def require_admin(
        request: Request,
        authorization: Annotated[str | None, Header()] = None,
    ) -> None:
        if not active_settings.admin_token:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Admin token is not configured")
        supplied = authorization.removeprefix("Bearer ") if authorization else ""
        if supplied and hmac.compare_digest(supplied, active_settings.admin_token):
            return
        csrf_hash = validate_admin_session(
            active_settings.admin_token,
            request.cookies.get(ADMIN_SESSION_COOKIE, ""),
        )
        if csrf_hash is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid admin token")
        if request.method not in {"GET", "HEAD", "OPTIONS"} and not validate_admin_csrf(
            csrf_hash,
            request.cookies.get(ADMIN_CSRF_COOKIE, ""),
            request.headers.get("x-csrf-token", ""),
        ):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="CSRF validation failed")

    admin = Depends(require_admin)

    @application.get("/health/live")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.post("/auth/login")
    def admin_login(payload: AdminLoginRequest, response: Response, request: Request) -> dict[str, str]:
        if not active_settings.admin_token:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Admin token is not configured")
        if not hmac.compare_digest(payload.token, active_settings.admin_token):
            LOGGER.warning("admin login failed client=%s", _client_ip(request))
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid admin token")
        session_token, csrf_token = issue_admin_session(
            active_settings.admin_token,
            active_settings.admin_session_ttl_hours,
        )
        max_age = active_settings.admin_session_ttl_hours * 3600
        response.set_cookie(
            ADMIN_SESSION_COOKIE,
            session_token,
            max_age=max_age,
            secure=True,
            httponly=True,
            samesite="strict",
            path="/",
        )
        response.set_cookie(
            ADMIN_CSRF_COOKIE,
            csrf_token,
            max_age=max_age,
            secure=True,
            httponly=False,
            samesite="strict",
            path="/",
        )
        LOGGER.info("admin login succeeded client=%s", _client_ip(request))
        return {"csrf_token": csrf_token}

    @application.get("/auth/check", status_code=status.HTTP_204_NO_CONTENT, dependencies=[admin])
    def admin_check() -> Response:
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @application.get("/auth/session", dependencies=[admin])
    def admin_session(request: Request) -> dict[str, str]:
        return {"csrf_token": request.cookies.get(ADMIN_CSRF_COOKIE, "")}

    @application.post("/auth/logout", status_code=status.HTTP_204_NO_CONTENT, dependencies=[admin])
    def admin_logout() -> Response:
        response = Response(status_code=status.HTTP_204_NO_CONTENT)
        response.delete_cookie(ADMIN_SESSION_COOKIE, path="/", secure=True, httponly=True, samesite="strict")
        response.delete_cookie(ADMIN_CSRF_COOKIE, path="/", secure=True, httponly=False, samesite="strict")
        return response

    @application.get("/overview", dependencies=[admin])
    def overview() -> dict[str, object]:
        return active_controller.overview()

    @application.get("/metrics/history", dependencies=[admin])
    def metric_history(
        metric: MetricName,
        range_name: Annotated[MetricRange, Query(alias="range")] = "5m",
    ) -> dict[str, object]:
        try:
            return active_metric_store.history(metric, range_name)
        except MetricStoreError as exc:
            LOGGER.exception("hardware metric history query failed")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc

    @application.post("/model/{action}", dependencies=[admin])
    def model_action(action: Literal["start", "stop", "restart"], request: Request) -> dict[str, object]:
        LOGGER.warning("model action=%s client=%s", action, _client_ip(request))
        try:
            return active_controller.model_action(action)
        except (AdminCommandError, ConfigurationError, ServerRequestError) as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    @application.post("/containers/{name}/{action}", dependencies=[admin])
    def container_action(
        name: str,
        action: Literal["start", "stop", "restart"],
        request: Request,
    ) -> dict[str, str]:
        LOGGER.warning("container action=%s name=%s client=%s", action, name, _client_ip(request))
        try:
            return active_controller.container_action(name, action)
        except AdminCommandError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    @application.get("/containers/{name}/logs", dependencies=[admin])
    def container_logs(
        name: str,
        tail: Annotated[int, Query(ge=1, le=500)] = 120,
    ) -> dict[str, str]:
        try:
            return active_controller.container_logs(name, tail)
        except AdminCommandError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    def model_name() -> str | None:
        config = getattr(active_controller.model_server, "config", None)
        value = getattr(config, "model_name", None)
        return str(value) if value else None

    def conversation_http_error(exc: ConversationStoreError) -> HTTPException:
        if isinstance(exc, ConversationNotFoundError):
            return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
        LOGGER.exception("conversation store operation failed", exc_info=exc)
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        )

    @application.get("/conversations", dependencies=[admin])
    def list_conversations() -> dict[str, object]:
        try:
            return {"items": active_conversation_store.list_conversations()}
        except ConversationStoreError as exc:
            raise conversation_http_error(exc) from exc

    @application.post("/conversations", status_code=status.HTTP_201_CREATED, dependencies=[admin])
    def create_conversation(payload: CreateConversationRequest) -> dict[str, object]:
        title = payload.title.strip()
        if not title:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="对话标题不能为空")
        try:
            conversation = active_conversation_store.create_conversation(
                title=title,
                model_name=model_name(),
                enable_thinking=payload.enable_thinking,
            )
            return {"conversation": conversation}
        except ConversationStoreError as exc:
            raise conversation_http_error(exc) from exc

    @application.get("/conversations/{conversation_id}", dependencies=[admin])
    def get_conversation(conversation_id: str) -> dict[str, object]:
        try:
            return {"conversation": active_conversation_store.get_conversation(conversation_id)}
        except ConversationStoreError as exc:
            raise conversation_http_error(exc) from exc

    @application.patch("/conversations/{conversation_id}", dependencies=[admin])
    def rename_conversation(
        conversation_id: str,
        payload: RenameConversationRequest,
    ) -> dict[str, object]:
        title = payload.title.strip()
        if not title:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="对话标题不能为空")
        try:
            conversation = active_conversation_store.rename_conversation(conversation_id, title)
            return {"conversation": conversation}
        except ConversationStoreError as exc:
            raise conversation_http_error(exc) from exc

    @application.delete(
        "/conversations/{conversation_id}",
        status_code=status.HTTP_204_NO_CONTENT,
        dependencies=[admin],
    )
    def delete_conversation(conversation_id: str) -> Response:
        try:
            active_conversation_store.delete_conversation(conversation_id)
            return Response(status_code=status.HTTP_204_NO_CONTENT)
        except ConversationStoreError as exc:
            raise conversation_http_error(exc) from exc

    @application.post("/conversations/{conversation_id}/chat", dependencies=[admin])
    def chat_in_conversation(
        conversation_id: str,
        payload: ConversationChatRequest,
    ) -> dict[str, object]:
        content = payload.content.strip()
        if not content:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="消息不能为空")
        active_controller.model_server.refresh()
        active_model_name = model_name()
        try:
            conversation, user_message, context = active_conversation_store.start_turn(
                conversation_id,
                content=content,
                model_name=active_model_name,
                enable_thinking=payload.enable_thinking,
            )
        except ConversationStoreError as exc:
            raise conversation_http_error(exc) from exc
        try:
            result = active_controller.model_server.client.chat(
                context,
                enable_thinking=payload.enable_thinking,
            )
        except (ConfigurationError, ServerRequestError) as exc:
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
        try:
            conversation, assistant_message = active_conversation_store.finish_turn(
                conversation_id,
                content=result.content,
                reasoning_content=result.reasoning_content,
                model_name=active_model_name,
            )
        except ConversationStoreError as exc:
            raise conversation_http_error(exc) from exc
        return {
            "conversation": conversation,
            "user_message": user_message,
            "assistant_message": assistant_message,
        }

    @application.post("/chat", dependencies=[admin])
    def chat(payload: ChatRequest) -> dict[str, str]:
        active_controller.model_server.refresh()
        try:
            result = active_controller.model_server.client.chat(
                [message.model_dump() for message in payload.messages],
                enable_thinking=payload.enable_thinking,
            )
        except (ConfigurationError, ServerRequestError) as exc:
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
        return {
            "content": result.content,
            "reasoning_content": result.reasoning_content,
        }

    return application


app = create_app()


def main() -> None:
    socket_path = os.getenv("OMNI_CONTROLLER_SOCKET", "/run/omni-ai-controller/controller.sock")
    uvicorn.run(
        "omni_ai_controller.service:app",
        uds=socket_path,
        proxy_headers=True,
        forwarded_allow_ips="*",
        log_level="info",
    )


if __name__ == "__main__":
    main()
