from __future__ import annotations

from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from .conversation_store import ConversationStore, ConversationStoreError


class MutsuControlError(RuntimeError):
    """Mutsu control data is unavailable."""


class MutsuControlConflict(MutsuControlError):
    """Configuration was edited concurrently."""


class MutsuControlStore(ConversationStore):
    """Use the existing Controller DB connection, not a separate credential file."""

    @staticmethod
    def _event(cursor: Any, actor_id: str, event: str, status: str, details: dict[str, Any]) -> None:
        cursor.execute(
            """INSERT INTO mutsu_behavior_events(actor_admin_id, event_type, status, details)
               VALUES (%s, %s, %s, %s)""",
            (actor_id, event, status, Jsonb(details)),
        )

    def settings(self) -> dict[str, Any]:
        try:
            with self._connect() as connection, connection.cursor(row_factory=dict_row) as cursor:
                cursor.execute("SELECT configuration, revision, updated_at FROM mutsu_settings WHERE singleton = TRUE")
                row = cursor.fetchone()
                if row is None:
                    raise MutsuControlError("陆奥控制数据尚未初始化")
                return dict(row)
        except (psycopg.Error, ConversationStoreError) as exc:
            raise MutsuControlError("陆奥控制数据不可用，请确认数据库迁移已完成") from exc

    def save_settings(self, actor_id: str, configuration: dict[str, Any], revision: int) -> dict[str, Any]:
        try:
            with self._connect() as connection, connection.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    """UPDATE mutsu_settings SET configuration = %s, revision = revision + 1,
                           updated_by = %s, updated_at = NOW()
                       WHERE singleton = TRUE AND revision = %s
                       RETURNING configuration, revision, updated_at""",
                    (Jsonb(configuration), actor_id, revision),
                )
                row = cursor.fetchone()
                if row is None:
                    raise MutsuControlConflict("设置已被其他窗口修改，请刷新后重试")
                self._event(cursor, actor_id, "settings.updated", "completed", {
                    "revision": row["revision"], "changed_fields": list(configuration),
                })
                return dict(row)
        except (psycopg.Error, ConversationStoreError) as exc:
            raise MutsuControlError("无法保存陆奥设置") from exc

    def capabilities(self) -> list[dict[str, Any]]:
        try:
            with self._connect() as connection, connection.cursor(row_factory=dict_row) as cursor:
                cursor.execute("SELECT capability_key, definition, updated_at FROM mutsu_capabilities ORDER BY capability_key LIMIT 100")
                return [{**row["definition"], "key": row["capability_key"], "updated_at": row["updated_at"], "runtime_available": False} for row in cursor.fetchall()]
        except (psycopg.Error, ConversationStoreError) as exc:
            raise MutsuControlError("无法读取陆奥能力登记") from exc

    def save_capability(self, actor_id: str, key: str, definition: dict[str, Any]) -> None:
        try:
            with self._connect() as connection, connection.cursor(row_factory=dict_row) as cursor:
                # Serialize registration to enforce a finite catalog, including concurrent requests.
                cursor.execute("SELECT singleton FROM mutsu_settings WHERE singleton = TRUE FOR UPDATE")
                cursor.execute("SELECT COUNT(*) AS count FROM mutsu_capabilities WHERE capability_key <> %s", (key,))
                if cursor.fetchone()["count"] >= 100:
                    raise MutsuControlConflict("能力登记最多 100 项")
                cursor.execute(
                    """INSERT INTO mutsu_capabilities(capability_key, definition, updated_by)
                       VALUES (%s, %s, %s) ON CONFLICT (capability_key) DO UPDATE
                       SET definition = EXCLUDED.definition, updated_by = EXCLUDED.updated_by, updated_at = NOW()""",
                    (key, Jsonb(definition), actor_id),
                )
                self._event(cursor, actor_id, "capability.updated", "completed", {"key": key, "enabled": definition["enabled"], "runtime_available": False})
        except (psycopg.Error, ConversationStoreError) as exc:
            raise MutsuControlError("无法保存陆奥能力登记") from exc

    def skills(self, *, active_only: bool = False) -> list[dict[str, Any]]:
        try:
            with self._connect() as connection, connection.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    """SELECT skill_key AS key, title, guidance, status, origin,
                              required_permission, revision, updated_at
                       FROM mutsu_skills WHERE (%s = FALSE OR status = 'active')
                       ORDER BY skill_key LIMIT 100""", (active_only,),
                )
                return [dict(row) for row in cursor.fetchall()]
        except (psycopg.Error, ConversationStoreError) as exc:
            raise MutsuControlError("无法读取陆奥技能") from exc

    def save_skill(self, actor_id: str, key: str, definition: dict[str, Any]) -> None:
        try:
            with self._connect() as connection, connection.cursor(row_factory=dict_row) as cursor:
                cursor.execute("SELECT singleton FROM mutsu_settings WHERE singleton = TRUE FOR UPDATE")
                cursor.execute("SELECT COUNT(*) AS count FROM mutsu_skills WHERE skill_key <> %s", (key,))
                if cursor.fetchone()["count"] >= 100:
                    raise MutsuControlConflict("技能库最多 100 项")
                cursor.execute(
                    """INSERT INTO mutsu_skills(skill_key, title, guidance, status, required_permission, updated_by)
                       VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (skill_key) DO UPDATE
                       SET title = EXCLUDED.title, guidance = EXCLUDED.guidance,
                           status = EXCLUDED.status, required_permission = EXCLUDED.required_permission,
                           revision = mutsu_skills.revision + 1, updated_by = EXCLUDED.updated_by, updated_at = NOW()""",
                    (key, definition["title"], definition["guidance"], definition["status"], definition["required_permission"], actor_id),
                )
                self._event(cursor, actor_id, "skill.updated", "completed", {"key": key, "status": definition["status"]})
        except (psycopg.Error, ConversationStoreError) as exc:
            raise MutsuControlError("无法保存陆奥技能") from exc

    def events(self, limit: int, before: int | None) -> list[dict[str, Any]]:
        try:
            with self._connect() as connection, connection.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                          """SELECT event.id, event.actor_admin_id::text, account.username AS actor_name,
                                        event.event_type, event.status, event.details, event.created_at
                              FROM mutsu_behavior_events event
                              LEFT JOIN controller_admin_accounts account ON account.id = event.actor_admin_id
                              WHERE (%s::bigint IS NULL OR event.id < %s)
                              ORDER BY event.id DESC LIMIT %s""", (before, before, limit),
                )
                return [dict(row) for row in cursor.fetchall()]
        except (psycopg.Error, ConversationStoreError) as exc:
            raise MutsuControlError("无法读取陆奥行为记录") from exc

    def record(self, actor_id: str, event: str, status: str, details: dict[str, Any]) -> None:
        try:
            with self._connect() as connection, connection.cursor(row_factory=dict_row) as cursor:
                self._event(cursor, actor_id, event, status, details)
        except (psycopg.Error, ConversationStoreError) as exc:
            raise MutsuControlError("无法记录陆奥行为") from exc