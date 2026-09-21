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
    return TestClient(
        create_app(settings, FakeController()),
        base_url="https://testserver",
    )


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


def test_browser_admin_session_requires_key_and_csrf() -> None:
    network_headers = {"X-Forwarded-For": "192.168.192.10"}
    with client() as test_client:
        rejected = test_client.post(
            "/auth/login",
            headers=network_headers,
            json={"token": "wrong"},
        )
        assert rejected.status_code == 401

        login = test_client.post(
            "/auth/login",
            headers=network_headers,
            json={"token": "secret-token"},
        )
        assert login.status_code == 200
        csrf_token = login.json()["csrf_token"]
        assert "omni_admin_session=" in login.headers["set-cookie"]
        assert "HttpOnly" in login.headers["set-cookie"]
        assert "Secure" in login.headers["set-cookie"]
        assert "secret-token" not in login.headers["set-cookie"]

        assert test_client.get("/auth/check", headers=network_headers).status_code == 204
        assert test_client.get("/overview", headers=network_headers).status_code == 200
        assert test_client.post("/model/start", headers=network_headers).status_code == 403

        action_headers = {
            **network_headers,
            "X-CSRF-Token": csrf_token,
        }
        action = test_client.post("/model/start", headers=action_headers)
        assert action.status_code == 200

        logout = test_client.post("/auth/logout", headers=action_headers)
        assert logout.status_code == 204
        assert test_client.get("/auth/check", headers=network_headers).status_code == 401
