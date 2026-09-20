from __future__ import annotations

import hmac
import ipaddress
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .admin import AdminCommandError, AdminController
from .client import ServerRequestError
from .config import ConfigurationError

LOGGER = logging.getLogger("omni_ai_controller.service")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@dataclass(frozen=True)
class ServiceSettings:
    model_dir: Path
    admin_token: str
    allowed_networks: tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...]
    allowed_containers: tuple[str, ...]

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
        return cls(
            model_dir=Path(os.getenv("OMNI_MODEL_DIR", "/opt/ai_server/omni_ai_model")),
            admin_token=os.getenv("OMNI_ADMIN_TOKEN", ""),
            allowed_networks=networks,
            allowed_containers=containers,
        )


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=16_000)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=32)
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
) -> FastAPI:
    active_settings = settings or ServiceSettings.from_environment()
    active_controller = controller or AdminController(
        active_settings.model_dir,
        active_settings.allowed_containers,
    )
    application = FastAPI(
        title="Omni AI Host Controller",
        version="1.0.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
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

    def require_admin(authorization: Annotated[str | None, Header()] = None) -> None:
        if not active_settings.admin_token:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Admin token is not configured")
        supplied = authorization.removeprefix("Bearer ") if authorization else ""
        if not hmac.compare_digest(supplied, active_settings.admin_token):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid admin token")

    admin = Depends(require_admin)

    @application.get("/health/live")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.get("/overview", dependencies=[admin])
    def overview() -> dict[str, object]:
        return active_controller.overview()

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
