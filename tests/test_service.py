import ipaddress
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from omni_ai_controller.service import ServiceSettings, create_app


class FakeController:
    def __init__(self) -> None:
        self.model_server = SimpleNamespace(
            refresh=lambda: None,
            client=SimpleNamespace(
                chat=lambda messages, enable_thinking: SimpleNamespace(
                    content="你好",
                    reasoning_content="",
                )
            ),
        )

    def overview(self) -> dict[str, object]:
        return {"hardware": {}, "containers": [], "model": {}}

    def model_action(self, action: str) -> dict[str, str]:
        return {"action": action}

    def container_action(self, name: str, action: str) -> dict[str, str]:
        return {"container": name, "action": action}

    def container_logs(self, name: str, tail: int) -> dict[str, str]:
        return {"container": name, "logs": f"last {tail}"}


def client() -> TestClient:
    settings = ServiceSettings(
        model_dir=Path("/tmp/model"),
        admin_token="secret-token",
        allowed_networks=(ipaddress.ip_network("192.168.192.0/24"),),
        allowed_containers=("omni-ai-model",),
    )
    return TestClient(create_app(settings, FakeController()))


def headers(token: str = "secret-token", ip: str = "192.168.192.10") -> dict[str, str]:
    return {"Authorization": f"Bearer {token}", "X-Forwarded-For": ip}


def test_overview_requires_network_and_token() -> None:
    with client() as test_client:
        assert test_client.get("/overview", headers=headers()).status_code == 200
        assert test_client.get("/overview", headers=headers(token="wrong")).status_code == 401
        assert test_client.get("/overview", headers=headers(ip="192.168.50.10")).status_code == 403


def test_control_and_chat_routes() -> None:
    with client() as test_client:
        action = test_client.post("/model/start", headers=headers())
        assert action.status_code == 200
        assert action.json() == {"action": "start"}

        response = test_client.post(
            "/chat",
            headers=headers(),
            json={
                "messages": [{"role": "user", "content": "你好"}],
                "enable_thinking": False,
            },
        )
        assert response.status_code == 200
        assert response.json()["content"] == "你好"
