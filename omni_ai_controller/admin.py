from __future__ import annotations

import json
import subprocess
import threading
from pathlib import Path
from typing import Any

from .hardware import hardware_status
from .runtime import LocalModelServer


class AdminCommandError(RuntimeError):
    """Raised when an allowlisted administrative command fails."""


class AdminController:
    def __init__(self, model_dir: Path | str, allowed_containers: tuple[str, ...]) -> None:
        self.model_server = LocalModelServer(model_dir)
        self.allowed_containers = allowed_containers
        self._action_lock = threading.Lock()

    def overview(self) -> dict[str, Any]:
        return {
            "hardware": hardware_status(),
            "containers": self.container_status(),
            "model": self.model_status(),
        }

    def model_status(self) -> dict[str, Any]:
        compose_status, api_status, api_error = self.model_server.status()
        return {
            "compose": compose_status,
            "api": api_status,
            "error": api_error,
        }

    def model_action(self, action: str) -> dict[str, Any]:
        actions = {
            "start": self.model_server.start,
            "stop": self.model_server.stop,
            "restart": self.model_server.restart,
        }
        operation = actions.get(action)
        if operation is None:
            raise AdminCommandError("不支持的模型操作")
        if not self._action_lock.acquire(blocking=False):
            raise AdminCommandError("另一个控制操作正在执行")
        try:
            return operation()
        finally:
            self._action_lock.release()

    def container_action(self, name: str, action: str) -> dict[str, str]:
        self._require_container(name)
        if action not in {"start", "stop", "restart"}:
            raise AdminCommandError("不支持的容器操作")
        if not self._action_lock.acquire(blocking=False):
            raise AdminCommandError("另一个控制操作正在执行")
        try:
            self._docker([action, name], timeout=90)
        finally:
            self._action_lock.release()
        return {"container": name, "action": action, "status": "accepted"}

    def container_logs(self, name: str, tail: int) -> dict[str, str]:
        self._require_container(name)
        safe_tail = max(1, min(tail, 500))
        output = self._docker(["logs", "--tail", str(safe_tail), name], timeout=15)
        return {"container": name, "logs": output[-100_000:]}

    def container_status(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for name in self.allowed_containers:
            try:
                raw = self._docker(["inspect", name], timeout=8)
                item = json.loads(raw)[0]
                state = item.get("State") or {}
                health = state.get("Health") or {}
                records.append(
                    {
                        "name": name,
                        "image": (item.get("Config") or {}).get("Image"),
                        "status": state.get("Status", "unknown"),
                        "running": bool(state.get("Running")),
                        "health": health.get("Status"),
                        "started_at": state.get("StartedAt"),
                        "restart_count": item.get("RestartCount", 0),
                    }
                )
            except (AdminCommandError, IndexError, KeyError, json.JSONDecodeError):
                records.append(
                    {
                        "name": name,
                        "image": None,
                        "status": "absent",
                        "running": False,
                        "health": None,
                        "started_at": None,
                        "restart_count": 0,
                    }
                )
        return records

    def _require_container(self, name: str) -> None:
        if name not in self.allowed_containers:
            raise AdminCommandError("容器不在允许控制的白名单中")

    @staticmethod
    def _docker(args: list[str], *, timeout: int) -> str:
        try:
            result = subprocess.run(
                ["docker", *args],
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise AdminCommandError(f"Docker 命令无法完成：{exc}") from exc
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise AdminCommandError(detail or "Docker 命令执行失败")
        return result.stdout.strip()
