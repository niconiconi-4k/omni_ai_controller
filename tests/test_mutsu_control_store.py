from unittest.mock import MagicMock

import psycopg
import pytest

from omni_ai_controller.mutsu_control_store import MutsuControlConflict, MutsuControlError, MutsuControlStore


def mocked_store():
    store = MutsuControlStore(host="localhost", port=15432, database="test", user="test", password="test-only")
    connection = MagicMock()
    cursor = connection.__enter__.return_value.cursor.return_value.__enter__.return_value
    store._connect = MagicMock(return_value=connection)
    return store, connection, cursor


def test_configuration_revision_update_and_event_use_same_transaction():
    store, connection, cursor = mocked_store()
    config = {"display_name": "陆奥", "avatar_icon": "🌙", "persona": "不应出现在行为记录中的人格文本"}
    cursor.fetchone.return_value = {"configuration": config, "revision": 2, "updated_at": None}
    assert store.save_settings("actor", config, 1)["revision"] == 2
    update, event = cursor.execute.call_args_list
    assert "AND revision = %s" in update.args[0]
    assert update.args[1][0].obj == config
    assert update.args[1][2] == 1
    assert "mutsu_behavior_events" in event.args[0]
    assert "不应出现在" not in str(event.args[1][3].obj)
    assert connection.__exit__.call_args.args[0] is None


def test_configuration_conflict_rolls_back_and_does_not_append_success_event():
    store, connection, cursor = mocked_store()
    cursor.fetchone.return_value = None
    with pytest.raises(MutsuControlConflict):
        store.save_settings("actor", {}, 1)
    assert cursor.execute.call_count == 1
    assert connection.__exit__.call_args.args[0] is MutsuControlConflict


def test_capability_catalog_limit_is_enforced_inside_lock():
    store, _, cursor = mocked_store()
    cursor.fetchone.return_value = {"count": 100}
    with pytest.raises(MutsuControlConflict):
        store.save_capability("actor", "new-tool", {"enabled": True})
    assert "FOR UPDATE" in cursor.execute.call_args_list[0].args[0]
    assert all("INSERT INTO" not in call.args[0] for call in cursor.execute.call_args_list)


def test_skill_update_preserves_origin_and_only_increments_revision():
    store, _, cursor = mocked_store()
    cursor.fetchone.return_value = {"count": 0}
    store.save_skill("actor", "skill", {"title": "指引", "guidance": "只读", "status": "active", "required_permission": None})
    insert = cursor.execute.call_args_list[2].args[0]
    assert "revision = mutsu_skills.revision + 1" in insert
    assert "origin =" not in insert
    assert "mutsu_behavior_events" in cursor.execute.call_args_list[-1].args[0]


def test_behavior_query_is_keyset_paginated_not_message_content():
    store, _, cursor = mocked_store()
    cursor.fetchall.return_value = []
    assert store.events(50, 100) == []
    query, params = cursor.execute.call_args.args
    assert "event.id < %s" in query
    assert "ai_messages" not in query
    assert params == (100, 100, 50)


def test_store_translates_database_errors_without_exposing_credentials():
    store, _, _ = mocked_store()
    store._connect.side_effect = psycopg.OperationalError("sensitive database detail")
    with pytest.raises(MutsuControlError) as failure:
        store.settings()
    assert "sensitive database detail" not in str(failure.value)