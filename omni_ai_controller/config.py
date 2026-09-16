from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


class ConfigurationError(ValueError):
    """Raised when the selected model server directory is invalid."""


def read_env(path: Path) -> dict[str, str]:
    """Read the simple KEY=VALUE format used by Docker Compose env files."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        values[key] = value
    return values


def validate_model_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    required = (
        resolved / "compose.yaml",
        resolved / "scripts" / "deploy.sh",
    )
    missing = [str(item.relative_to(resolved)) for item in required if not item.is_file()]
    if missing:
        raise ConfigurationError(
            f"不是有效的 omni_ai_model 目录，缺少：{', '.join(missing)}"
        )
    return resolved


@dataclass(frozen=True)
class ServerConfig:
    model_dir: Path
    base_url: str
    api_key: str
    control_token: str
    model_name: str

    @classmethod
    def from_model_dir(cls, model_dir: Path | str) -> "ServerConfig":
        resolved = validate_model_dir(Path(model_dir))
        values = read_env(resolved / ".env")
        port = values.get("API_PORT", "8000")
        return cls(
            model_dir=resolved,
            base_url=f"http://127.0.0.1:{port}",
            api_key=values.get("OPENAI_API_KEY", ""),
            control_token=values.get("CONTROL_TOKEN", ""),
            model_name=values.get("SERVED_MODEL_NAME", "qwen3.6-27b-instruct"),
        )

    def require_credentials(self) -> None:
        if not self.api_key or not self.control_token:
            raise ConfigurationError(
                "模型目录中没有可用的 .env 或访问密钥，请先启动容器以生成配置。"
            )


def controller_config_path() -> Path:
    root = Path(os.getenv("XDG_CONFIG_HOME", Path.home() / ".config"))
    return root / "omni-ai-controller" / "config.json"


def load_saved_model_dir(path: Path | None = None) -> Path | None:
    config_path = path or controller_config_path()
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
        value = data.get("model_dir")
        return Path(value).expanduser() if isinstance(value, str) and value else None
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None


def save_model_dir(model_dir: Path, path: Path | None = None) -> None:
    config_path = path or controller_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps({"model_dir": str(model_dir)}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
