from __future__ import annotations

import json
import logging
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .admin_account_store import PERMISSIONS
from .mutsu_control_store import MutsuControlConflict, MutsuControlError, MutsuControlStore

LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIGURATION = {"display_name": "陆奥", "avatar_icon": "陆", "persona": ""}


class ControlInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class SettingsUpdate(ControlInput):
    revision: int = Field(ge=1)
    display_name: str = Field(default="陆奥", min_length=1, max_length=40)
    avatar_icon: str = Field(default="陆", min_length=1, max_length=16)
    persona: str = Field(default="", max_length=4000)


class CapabilityUpdate(ControlInput):
    name: str = Field(min_length=1, max_length=160)
    kind: Literal["api", "tool"]
    description: str = Field(default="", max_length=1000)
    logical_target: str = Field(min_length=1, max_length=128, pattern=r"^[a-zA-Z][a-zA-Z0-9_.:-]*$")
    required_permission: str | None = None
    enabled: bool = False
    parameters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("required_permission")
    @classmethod
    def valid_permission(cls, value: str | None) -> str | None:
        if value and value not in PERMISSIONS:
            raise ValueError("未知管理员权限")
        return value or None

    @field_validator("parameters")
    @classmethod
    def bounded_parameters(cls, value: dict[str, Any]) -> dict[str, Any]:
        if len(json.dumps(value, ensure_ascii=False, allow_nan=False)) > 8000:
            raise ValueError("参数定义超过安全预算")
        return value


class SkillUpdate(ControlInput):
    title: str = Field(min_length=1, max_length=160)
    guidance: str = Field(min_length=1, max_length=2000)
    status: Literal["draft", "active", "disabled"] = "draft"
    required_permission: str | None = None

    @field_validator("required_permission")
    @classmethod
    def valid_permission(cls, value: str | None) -> str | None:
        return CapabilityUpdate.valid_permission(value)


def runtime_configuration(store: MutsuControlStore) -> dict[str, Any]:
    try:
        return {**DEFAULT_CONFIGURATION, **store.settings()["configuration"]}
    except MutsuControlError:
        # Missing migration must not disable the existing administrator assistant.
        return dict(DEFAULT_CONFIGURATION)


def appearance(store: MutsuControlStore) -> dict[str, str]:
    configuration = runtime_configuration(store)
    return {key: str(configuration[key]) for key in ("display_name", "avatar_icon")}


def runtime_skills(store: MutsuControlStore, permissions: list[str]) -> list[dict[str, str]]:
    try:
        skills = store.skills(active_only=True)
    except MutsuControlError:
        return []
    selected: list[dict[str, str]] = []
    characters = 0
    for skill in skills:
        if skill.get("required_permission") and skill["required_permission"] not in permissions:
            continue
        guidance = str(skill.get("guidance") or "")
        if len(selected) >= 8 or characters + len(guidance) > 4000:
            break
        selected.append({"key": str(skill["key"]), "title": str(skill["title"]), "guidance": guidance})
        characters += len(guidance)
    return selected


def record_behavior(store: MutsuControlStore, actor: str, event: str, status: str, details: dict[str, Any]) -> None:
    try:
        store.record(actor, event, status, details)
    except MutsuControlError:
        LOGGER.warning("Mutsu behavior storage unavailable; event=%s", event)


def control_router(store: MutsuControlStore, require_super: Any) -> APIRouter:
    router = APIRouter(prefix="/mutsu/control")
    key_type = Path(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_-]*$")

    def unavailable(exc: MutsuControlError) -> HTTPException:
        return HTTPException(status_code=409 if isinstance(exc, MutsuControlConflict) else 503, detail=str(exc))

    @router.get("/settings")
    def settings(account: dict = Depends(require_super)) -> dict:
        try:
            data = store.settings()
            data["configuration"] = {**DEFAULT_CONFIGURATION, **data["configuration"]}
            data["runtime_policy"] = {"tools_executable": False, "automatic_learning": False, "permissions_enforced": True}
            return data
        except MutsuControlError as exc:
            raise unavailable(exc) from exc

    @router.put("/settings")
    def save_settings(payload: SettingsUpdate, account: dict = Depends(require_super)) -> dict:
        try:
            return store.save_settings(str(account["id"]), payload.model_dump(exclude={"revision"}), payload.revision)
        except MutsuControlError as exc:
            raise unavailable(exc) from exc

    @router.get("/capabilities")
    def capabilities(account: dict = Depends(require_super)) -> dict:
        try:
            return {"items": store.capabilities(), "execution_enabled": False}
        except MutsuControlError as exc:
            raise unavailable(exc) from exc

    @router.put("/capabilities/{key}")
    def save_capability(payload: CapabilityUpdate, key: str = key_type, account: dict = Depends(require_super)) -> dict:
        try:
            store.save_capability(str(account["id"]), key, payload.model_dump())
            return {"saved": True, "runtime_available": False}
        except MutsuControlError as exc:
            raise unavailable(exc) from exc

    @router.get("/skills")
    def skills(account: dict = Depends(require_super)) -> dict:
        try:
            return {"items": store.skills(), "automatic_learning": False}
        except MutsuControlError as exc:
            raise unavailable(exc) from exc

    @router.put("/skills/{key}")
    def save_skill(payload: SkillUpdate, key: str = key_type, account: dict = Depends(require_super)) -> dict:
        try:
            store.save_skill(str(account["id"]), key, payload.model_dump())
            return {"saved": True}
        except MutsuControlError as exc:
            raise unavailable(exc) from exc

    @router.get("/events")
    def events(limit: int = Query(default=50, ge=1, le=100), before: int | None = Query(default=None, ge=1), account: dict = Depends(require_super)) -> dict:
        try:
            items = store.events(limit, before)
            return {"items": items, "next_before": items[-1]["id"] if len(items) == limit else None}
        except MutsuControlError as exc:
            raise unavailable(exc) from exc

    return router