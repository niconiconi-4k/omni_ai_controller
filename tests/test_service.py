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


class FakeConversationStore:
    def __init__(self) -> None:
        self.items: dict[str, dict[str, object]] = {}
        self.messages: dict[str, list[dict[str, object]]] = {}
        self.counter = 0

    def list_conversations(self) -> list[dict[str, object]]:
        return list(reversed(self.items.values()))

    def create_conversation(
        self,
        *,
        title: str,
        model_name: str | None,
        enable_thinking: bool,
    ) -> dict[str, object]:
        self.counter += 1
        conversation_id = f"conversation-{self.counter}"
        conversation = {
            "id": conversation_id,
            "title": title,
            "model_name": model_name,
            "enable_thinking": enable_thinking,
            "message_count": 0,
            "last_message": "",
        }
        self.items[conversation_id] = conversation
        self.messages[conversation_id] = []
        return conversation

    def get_conversation(self, conversation_id: str) -> dict[str, object]:
        return {**self.items[conversation_id], "messages": self.messages[conversation_id]}

    def rename_conversation(self, conversation_id: str, title: str) -> dict[str, object]:
        self.items[conversation_id]["title"] = title
        return self.items[conversation_id]

    def delete_conversation(self, conversation_id: str) -> None:
        del self.items[conversation_id]
        del self.messages[conversation_id]

    def start_turn(
        self,
        conversation_id: str,
        *,
        content: str,
        model_name: str | None,
        enable_thinking: bool,
    ) -> tuple[dict[str, object], dict[str, object], list[dict[str, str]]]:
        message = {"id": "user-1", "role": "user", "content": content, "reasoning_content": ""}
        self.messages[conversation_id].append(message)
        if self.items[conversation_id]["title"] == "新对话":
            self.items[conversation_id]["title"] = content
        self.items[conversation_id]["message_count"] = len(self.messages[conversation_id])
        return self.items[conversation_id], message, [
            {"role": item["role"], "content": item["content"]}  # type: ignore[dict-item]
            for item in self.messages[conversation_id]
        ]

    def finish_turn(
        self,
        conversation_id: str,
        *,
        content: str,
        reasoning_content: str,
        model_name: str | None,
    ) -> tuple[dict[str, object], dict[str, object]]:
        message = {
            "id": "assistant-1",
            "role": "assistant",
            "content": content,
            "reasoning_content": reasoning_content,
        }
        self.messages[conversation_id].append(message)
        self.items[conversation_id]["message_count"] = len(self.messages[conversation_id])
        self.items[conversation_id]["last_message"] = content
        return self.items[conversation_id], message


class FakeMetricStore:
    def history(self, metric: str, range_name: str) -> dict[str, object]:
        return {
            "metric": metric,
            "range": range_name,
            "resolution_seconds": 60,
            "resolution_label": "1 分钟",
            "from": "2026-01-01T00:00:00Z",
            "to": "2026-01-02T00:00:00Z",
            "series": [
                {
                    "key": "cpu_percent",
                    "label": "CPU 使用率",
                    "unit": "%",
                    "color": "#5ee7d0",
                }
            ],
            "points": [
                {
                    "timestamp": "2026-01-01T12:00:00Z",
                    "values": {"cpu_percent": 25.0},
                }
            ],
            "summaries": {
                "cpu_percent": {
                    "latest": 25.0,
                    "average": 25.0,
                    "minimum": 25.0,
                    "maximum": 25.0,
                }
            },
            "capabilities": {"host_power": False},
        }


def client() -> TestClient:
    settings = ServiceSettings(
        model_dir=Path("/tmp/model"),
        admin_token="secret-token",
        allowed_networks=(ipaddress.ip_network("192.168.192.0/24"),),
        allowed_containers=("omni-ai-model",),
        metric_collection_enabled=False,
    )
    return TestClient(
        create_app(
            settings,
            FakeController(),
            FakeConversationStore(),
            FakeMetricStore(),
        ),  # type: ignore[arg-type]
        base_url="https://testserver",
    )


def headers(token: str = "secret-token", ip: str = "192.168.192.10") -> dict[str, str]:
    return {"Authorization": f"Bearer {token}", "X-Forwarded-For": ip}


def test_overview_requires_network_and_token() -> None:
    with client() as test_client:
        assert test_client.get("/overview", headers=headers()).status_code == 200
        assert test_client.get("/overview", headers=headers(token="wrong")).status_code == 401
        assert test_client.get("/overview", headers=headers(ip="192.168.50.10")).status_code == 403


def test_metric_history_is_authenticated_and_validated() -> None:
    with client() as test_client:
        assert test_client.get(
            "/metrics/history?metric=cpu&range=1d",
            headers=headers(token="wrong"),
        ).status_code == 401

        response = test_client.get(
            "/metrics/history?metric=cpu&range=1d",
            headers=headers(),
        )
        assert response.status_code == 200
        assert response.json()["metric"] == "cpu"
        assert response.json()["range"] == "1d"
        assert response.json()["points"][0]["values"]["cpu_percent"] == 25.0

        assert test_client.get(
            "/metrics/history?metric=cpu&range=forever",
            headers=headers(),
        ).status_code == 422


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


def test_persisted_conversation_lifecycle() -> None:
    with client() as test_client:
        created = test_client.post(
            "/conversations",
            headers=headers(),
            json={"title": "新对话", "enable_thinking": False},
        )
        assert created.status_code == 201
        conversation_id = created.json()["conversation"]["id"]

        response = test_client.post(
            f"/conversations/{conversation_id}/chat",
            headers=headers(),
            json={"content": "检查模型状态", "enable_thinking": False},
        )
        assert response.status_code == 200
        assert response.json()["conversation"]["title"] == "检查模型状态"
        assert response.json()["user_message"]["role"] == "user"
        assert response.json()["assistant_message"]["content"] == "你好"

        detail = test_client.get(f"/conversations/{conversation_id}", headers=headers())
        assert [item["role"] for item in detail.json()["conversation"]["messages"]] == [
            "user",
            "assistant",
        ]

        renamed = test_client.patch(
            f"/conversations/{conversation_id}",
            headers=headers(),
            json={"title": "运行诊断"},
        )
        assert renamed.status_code == 200
        assert renamed.json()["conversation"]["title"] == "运行诊断"

        deleted = test_client.delete(f"/conversations/{conversation_id}", headers=headers())
        assert deleted.status_code == 204
        assert test_client.get("/conversations", headers=headers()).json()["items"] == []


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
