from pathlib import Path

import pytest

from omni_ai_controller.config import (
    ConfigurationError,
    ServerConfig,
    load_saved_model_dir,
    read_env,
    save_model_dir,
    validate_model_dir,
)


def make_model_dir(tmp_path: Path) -> Path:
    model_dir = tmp_path / "omni_ai_model"
    (model_dir / "scripts").mkdir(parents=True)
    (model_dir / "compose.yaml").write_text("services: {}\n", encoding="utf-8")
    (model_dir / "scripts" / "deploy.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    return model_dir


def test_read_env_supports_comments_export_and_quotes(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        '# comment\nexport API_PORT="8123"\nOPENAI_API_KEY=secret\nEMPTY=\n',
        encoding="utf-8",
    )

    assert read_env(env_file) == {
        "API_PORT": "8123",
        "OPENAI_API_KEY": "secret",
        "EMPTY": "",
    }


def test_server_config_reads_model_environment(tmp_path: Path) -> None:
    model_dir = make_model_dir(tmp_path)
    (model_dir / ".env").write_text(
        "API_PORT=9000\nOPENAI_API_KEY=openai\nCONTROL_TOKEN=control\n"
        "SERVED_MODEL_NAME=test-model\n",
        encoding="utf-8",
    )

    config = ServerConfig.from_model_dir(model_dir)

    assert config.base_url == "http://127.0.0.1:9000"
    assert config.api_key == "openai"
    assert config.control_token == "control"
    assert config.model_name == "test-model"


def test_validate_model_dir_reports_missing_files(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="compose.yaml"):
        validate_model_dir(tmp_path)


def test_saved_model_directory_round_trip(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    model_dir = tmp_path / "model"

    save_model_dir(model_dir, config_path)

    assert load_saved_model_dir(config_path) == model_dir
