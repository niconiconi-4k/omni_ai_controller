from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from .client import ModelServerClient, ServerRequestError
from .config import ServerConfig


class RuntimeCommandError(RuntimeError):
    """Raised when a local Docker or deployment command fails."""


class LocalModelServer:
    def __init__(self, model_dir: Path | str) -> None:
        self.config = ServerConfig.from_model_dir(model_dir)
        self.client = ModelServerClient(self.config)

    def refresh(self) -> None:
        self.config = ServerConfig.from_model_dir(self.config.model_dir)
        self.client = ModelServerClient(self.config)

    def start(self) -> dict[str, Any]:
        if not (self.config.model_dir / ".env").is_file() or not self.client.live():
            self._run(["bash", "scripts/deploy.sh"], capture=False)
            self.refresh()
        return self.client.start_model()

    def stop(self) -> dict[str, Any]:
        self.refresh()
        return self.client.stop_model()

    def restart(self) -> dict[str, Any]:
        self.refresh()
        return self.client.restart_model()

    def status(self) -> tuple[str, dict[str, Any] | None, str | None]:
        if not (self.config.model_dir / ".env").is_file():
            return "尚未部署模型容器。", None, "模型控制 API 尚未初始化。"
        compose_status = self._compose(["ps"])
        self.refresh()
        try:
            api_status = self.client.status()
            return compose_status, api_status, None
        except ServerRequestError as exc:
            return compose_status, None, str(exc)

    def stop_container(self) -> None:
        self._compose(["stop", "model"], capture=False)

    def open_shell(self) -> None:
        self._compose(["exec", "model", "bash"], capture=False)

    def _compose(self, args: list[str], *, capture: bool = True) -> str:
        command = [*self._docker_command(), "compose", "-f", "compose.yaml", *args]
        return self._run(command, capture=capture)

    def _docker_command(self) -> list[str]:
        direct = subprocess.run(
            ["docker", "info"],
            cwd=self.config.model_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return ["docker"] if direct.returncode == 0 else ["sudo", "docker"]

    def _run(self, command: list[str], *, capture: bool) -> str:
        try:
            result = subprocess.run(
                command,
                cwd=self.config.model_dir,
                text=True,
                stdout=subprocess.PIPE if capture else None,
                stderr=subprocess.STDOUT if capture else None,
                check=False,
            )
        except OSError as exc:
            raise RuntimeCommandError(f"无法执行 {' '.join(command)}：{exc}") from exc
        if result.returncode != 0:
            output = (result.stdout or "").strip()
            raise RuntimeCommandError(
                f"命令执行失败（退出码 {result.returncode}）：{' '.join(command)}"
                + (f"\n{output}" if output else "")
            )
        return (result.stdout or "").strip()
