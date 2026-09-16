from pathlib import Path

from omni_ai_controller.runtime import LocalModelServer


def make_model_dir(tmp_path: Path) -> Path:
    model_dir = tmp_path / "omni_ai_model"
    (model_dir / "scripts").mkdir(parents=True)
    (model_dir / "compose.yaml").write_text("services: {}\n", encoding="utf-8")
    (model_dir / "scripts" / "deploy.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    return model_dir


def test_status_before_first_deployment_does_not_call_docker(tmp_path: Path) -> None:
    server = LocalModelServer(make_model_dir(tmp_path))

    compose_status, api_status, api_error = server.status()

    assert compose_status == "尚未部署模型容器。"
    assert api_status is None
    assert api_error == "模型控制 API 尚未初始化。"