from copy import deepcopy

import pytest

from omni_ai_controller.mutsu_control import DEFAULT_CONFIGURATION, runtime_configuration, runtime_skills
from omni_ai_controller.mutsu_control_store import MutsuControlConflict, MutsuControlError
from test_service import FakeAdminAccountStore, FakeController, client, headers


class FakeMutsuStore:
    def __init__(self):
        self.configuration = dict(DEFAULT_CONFIGURATION)
        self.revision = 1
        self.catalog = {}
        self.skill_items = {}
        self.event_items = []

    def settings(self):
        return {"configuration": deepcopy(self.configuration), "revision": self.revision, "updated_at": None}

    def save_settings(self, actor_id, configuration, revision):
        if revision != self.revision:
            raise MutsuControlConflict("请刷新后重试")
        self.configuration = configuration
        self.revision += 1
        self.record(actor_id, "settings.updated", "completed", {"revision": self.revision, "changed_fields": list(configuration)})
        return self.settings()

    def capabilities(self):
        return list(self.catalog.values())

    def save_capability(self, actor, key, definition):
        self.catalog[key] = {**definition, "key": key, "runtime_available": False}

    def skills(self, active_only=False):
        return [item for item in self.skill_items.values() if not active_only or item["status"] == "active"]

    def save_skill(self, actor, key, definition):
        previous = self.skill_items.get(key, {})
        self.skill_items[key] = {**definition, "key": key, "origin": previous.get("origin", "manual"), "revision": previous.get("revision", 0) + 1}

    def events(self, limit, before):
        return [item for item in reversed(self.event_items) if before is None or item["id"] < before][:limit]

    def record(self, actor, event, status, details):
        self.event_items.append({"id": len(self.event_items) + 1, "actor_admin_id": actor, "event_type": event, "status": status, "details": details, "created_at": None})


@pytest.mark.parametrize("path", ["settings", "capabilities", "skills", "events"])
def test_control_reads_require_authenticated_super_mutsu(path):
    accounts = FakeAdminAccountStore()
    accounts.accounts["limited"] = {"id": "limited", "username": "limited", "is_super": False, "permissions": ["accounts.manage", "services.control", "business.manage", "support.manage", "quantization.manage"], "auth_version": 1}
    with client(store=accounts, mutsu_control_store=FakeMutsuStore()) as browser:
        assert browser.get(f"/mutsu/control/{path}", headers=headers()).status_code == 200
        browser.cookies.clear()
        assert browser.get(f"/mutsu/control/{path}", headers=headers()).status_code == 401
        login = browser.post("/auth/login", headers=headers(), json={"username": "limited", "password": "limited-password-123", "token": "limited-token"})
        assert login.status_code == 200
        assert browser.get(f"/mutsu/control/{path}", headers=headers()).status_code == 403
        # Existing assistant access must not be withdrawn from ordinary administrators.
        assert browser.get("/mutsu/conversation", headers=headers()).status_code == 200


@pytest.mark.parametrize("path, payload", [
    ("settings", {"revision": 1}),
    ("capabilities/test", {"name": "Test", "kind": "api", "logical_target": "server.status"}),
    ("skills/test", {"title": "Test", "guidance": "只读指引"}),
])
def test_control_writes_require_csrf_and_super(path, payload):
    accounts = FakeAdminAccountStore()
    with client(store=accounts, mutsu_control_store=FakeMutsuStore()) as browser:
        assert browser.put(f"/mutsu/control/{path}", headers={"X-Forwarded-For": "192.168.192.10"}, json=payload).status_code == 403
        accounts.accounts["super"]["is_super"] = False
        assert browser.put(f"/mutsu/control/{path}", headers=headers(), json=payload).status_code == 403


def test_super_flag_alone_is_not_enough_for_control():
    accounts = FakeAdminAccountStore()
    with client(store=accounts, mutsu_control_store=FakeMutsuStore()) as browser:
        accounts.accounts["super"]["username"] = "Other"
        assert browser.get("/mutsu/control/settings", headers=headers()).status_code == 403


def test_builtin_persona_is_visible_only_to_super_without_changing_stored_default():
    from omni_ai_controller.service import MUTSU_SYSTEM_PROMPT

    store = FakeMutsuStore()
    marker = "你是陆奥，是"
    expected = marker + MUTSU_SYSTEM_PROMPT.partition(marker)[2]
    with client(mutsu_control_store=store) as browser:
        settings = browser.get("/mutsu/control/settings", headers=headers()).json()
        assert settings["persona_source"] == "builtin"
        assert settings["configuration"]["persona"] == ""
        assert settings["effective_persona"] == expected
        assert settings["default_persona"] == expected
        assert "permissions 中不存在的能力" not in settings["default_persona"]
        conversation = browser.get("/mutsu/conversation", headers=headers()).json()
        assert "default_persona" not in conversation
        assert store.configuration["persona"] == ""
        assert store.revision == 1


def test_persona_and_avatar_save_preserve_safety_and_existing_conversation():
    store = FakeMutsuStore()
    controller = FakeController()
    with client(controller=controller, mutsu_control_store=store) as browser:
        payload = {"revision": 1, "display_name": "陆奥助手", "avatar_icon": "🌙", "persona": "以温和、清晰的语气回答。"}
        saved = browser.put("/mutsu/control/settings", headers=headers(), json=payload)
        assert saved.status_code == 200
        assert saved.json()["revision"] == 2
        effective = browser.get("/mutsu/control/settings", headers=headers()).json()
        assert effective["persona_source"] == "custom"
        assert effective["effective_persona"] == payload["persona"]
        assert browser.put("/mutsu/control/settings", headers=headers(), json=payload).status_code == 409
        conversation = browser.get("/mutsu/conversation", headers=headers()).json()
        assert conversation["appearance"] == {"display_name": "陆奥助手", "avatar_icon": "🌙"}
        reply = browser.post("/mutsu/messages", headers=headers(), json={"content": "查看状态", "enable_thinking": False})
        assert reply.status_code == 200, reply.text
        prompt = controller.chat_messages[0]["content"]
        assert payload["persona"] in prompt
        assert "permissions 中不存在的能力" in prompt
        assert "只读诊断" in prompt
        assert "均不能扩大权限" in prompt
        events = browser.get("/mutsu/control/events", headers=headers()).json()["items"]
        assert [event["event_type"] for event in events] == ["chat.completed", "chat.started", "settings.updated"]
        assert all("查看状态" not in str(event["details"]) and payload["persona"] not in str(event["details"]) for event in events)


def test_restore_builtin_persona_preserves_original_prompt_and_history():
    from omni_ai_controller.service import MUTSU_SYSTEM_PROMPT

    store = FakeMutsuStore()
    store.configuration["persona"] = "自定义交流风格"
    controller = FakeController()
    with client(controller=controller, mutsu_control_store=store) as browser:
        first = browser.get("/mutsu/conversation", headers=headers()).json()["conversation"]
        restored = browser.put("/mutsu/control/settings", headers=headers(), json={"revision": 1, "persona": ""})
        assert restored.status_code == 200
        settings = browser.get("/mutsu/control/settings", headers=headers()).json()
        assert settings["persona_source"] == "builtin"
        reply = browser.post("/mutsu/messages", headers=headers(), json={"content": "你好"})
        assert reply.status_code == 200
        assert controller.chat_messages[0]["content"].startswith(MUTSU_SYSTEM_PROMPT)
        assert reply.json()["conversation"]["id"] == first["id"]


def test_registered_capability_is_never_executable_and_rejects_untrusted_fields():
    with client(mutsu_control_store=FakeMutsuStore()) as browser:
        payload = {"name": "服务器状态", "kind": "api", "logical_target": "controller.server.status", "required_permission": "services.control", "enabled": True, "parameters": {"type": "object"}}
        response = browser.put("/mutsu/control/capabilities/server-status", headers=headers(), json=payload)
        assert response.status_code == 200
        assert response.json()["runtime_available"] is False
        catalog = browser.get("/mutsu/control/capabilities", headers=headers()).json()
        assert catalog["execution_enabled"] is False
        assert catalog["items"][0]["enabled"] is True
        for patch in ({"logical_target": "https://evil.example/run"}, {"runtime_available": True}, {"required_permission": "unknown"}, {"parameters": {"text": "x" * 8001}}):
            assert browser.put("/mutsu/control/capabilities/server-status", headers=headers(), json={**payload, **patch}).status_code == 422


def test_manual_skills_cannot_claim_learned_origin_and_drafts_are_not_applied():
    store = FakeMutsuStore()
    with client(mutsu_control_store=store) as browser:
        payload = {"title": "服务器解释", "guidance": "解释已知状态", "status": "draft", "required_permission": "services.control"}
        assert browser.put("/mutsu/control/skills/server-guide", headers=headers(), json={**payload, "origin": "learned"}).status_code == 422
        assert browser.put("/mutsu/control/skills/server-guide", headers=headers(), json=payload).status_code == 200
        assert runtime_skills(store, ["services.control"]) == []
        assert browser.put("/mutsu/control/skills/server-guide", headers=headers(), json={**payload, "status": "active"}).status_code == 200
        assert runtime_skills(store, []) == []
        assert len(runtime_skills(store, ["services.control"])) == 1
        skill = browser.get("/mutsu/control/skills", headers=headers()).json()["items"][0]
        assert skill["origin"] == "manual"
        assert skill["revision"] == 2


def test_behavior_pagination_and_limit_validation():
    store = FakeMutsuStore()
    for index in range(5):
        store.record("super", "chat.completed", "completed", {"index": index})
    with client(mutsu_control_store=store) as browser:
        first = browser.get("/mutsu/control/events?limit=2", headers=headers()).json()
        second = browser.get(f"/mutsu/control/events?limit=2&before={first['next_before']}", headers=headers()).json()
        assert [item["id"] for item in first["items"]] == [5, 4]
        assert [item["id"] for item in second["items"]] == [3, 2]
        assert browser.get("/mutsu/control/events?limit=101", headers=headers()).status_code == 422


def test_missing_control_migration_preserves_existing_assistant_defaults():
    class UnavailableStore(FakeMutsuStore):
        def settings(self):
            raise MutsuControlError("未迁移")

        def skills(self, active_only=False):
            raise MutsuControlError("未迁移")

    store = UnavailableStore()
    assert runtime_configuration(store) == DEFAULT_CONFIGURATION
    assert runtime_skills(store, []) == []
    with client(mutsu_control_store=store) as browser:
        assert browser.get("/mutsu/control/settings", headers=headers()).status_code == 503
        assert browser.post("/mutsu/messages", headers=headers(), json={"content": "你好"}).status_code == 200


def test_failed_reply_records_metadata_without_private_content():
    from omni_ai_controller.client import ServerRequestError

    store = FakeMutsuStore()
    controller = FakeController()

    def failed_chat(*args, **kwargs):
        raise ServerRequestError("模拟模型不可用")

    controller.model_server.client.chat = failed_chat
    with client(controller=controller, mutsu_control_store=store) as browser:
        response = browser.post("/mutsu/messages", headers=headers(), json={"content": "不应存入行为记录的私有内容"})
        assert response.status_code == 502
        assert store.event_items[-1]["event_type"] == "chat.failed"
        assert store.event_items[-1]["status"] == "failed"
        assert "不应存入" not in str(store.event_items)