from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


class ConversationStoreError(RuntimeError):
    """Raised when persisted model conversations cannot be accessed."""


class ConversationNotFoundError(ConversationStoreError):
    """Raised when a dashboard conversation does not exist or was deleted."""


class ConversationBusyError(ConversationStoreError):
    """Raised when an assistant response is already in flight."""


class ConversationStore:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        database: str,
        user: str,
        password: str,
    ) -> None:
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password

    def _connect(self) -> psycopg.Connection[Any]:
        if not self.password:
            raise ConversationStoreError("对话数据库凭据尚未配置")
        try:
            return psycopg.connect(
                host=self.host,
                port=self.port,
                dbname=self.database,
                user=self.user,
                password=self.password,
                connect_timeout=5,
            )
        except psycopg.Error as exc:
            raise ConversationStoreError("无法连接对话数据库") from exc

    @staticmethod
    def _normalize_id(conversation_id: str) -> UUID:
        try:
            return UUID(conversation_id)
        except (TypeError, ValueError) as exc:
            raise ConversationNotFoundError("对话不存在") from exc

    @staticmethod
    def _title_from_message(content: str) -> str:
        compact = " ".join(content.split())
        if len(compact) <= 42:
            return compact
        return f"{compact[:42].rstrip()}…"

    @staticmethod
    def _conversation_query() -> str:
        return """
            SELECT conversation.id::text AS id,
                   conversation.scope_type,
                     conversation.owner_account_id::text AS owner_account_id,
                   conversation.title,
                   conversation.model_name,
                   conversation.enable_thinking,
                   conversation.created_at,
                   conversation.updated_at,
                   COUNT(message.id)::integer AS message_count,
                   COALESCE((
                       SELECT latest.content
                       FROM ai_messages latest
                       WHERE latest.conversation_id = conversation.id
                       ORDER BY latest.sequence_number DESC
                       LIMIT 1
                   ), '') AS last_message
            FROM ai_conversations conversation
            LEFT JOIN ai_messages message ON message.conversation_id = conversation.id
            WHERE conversation.id = %s
              AND conversation.scope_type = 'dashboard'
                            AND conversation.conversation_kind = 'general'
              AND conversation.deleted_at IS NULL
            GROUP BY conversation.id
        """

    def _get_conversation(
        self,
        cursor: psycopg.Cursor[Any],
        conversation_id: UUID,
    ) -> dict[str, Any]:
        cursor.execute(self._conversation_query(), (conversation_id,))
        row = cursor.fetchone()
        if row is None:
            raise ConversationNotFoundError("对话不存在")
        return dict(row)

    def list_conversations(self) -> list[dict[str, Any]]:
        try:
            with self._connect() as connection, connection.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    """
                    SELECT conversation.id::text AS id,
                           conversation.scope_type,
                           conversation.owner_account_id::text AS owner_account_id,
                           conversation.title,
                           conversation.model_name,
                           conversation.enable_thinking,
                           conversation.created_at,
                           conversation.updated_at,
                           COUNT(message.id)::integer AS message_count,
                           COALESCE((
                               SELECT latest.content
                               FROM ai_messages latest
                               WHERE latest.conversation_id = conversation.id
                               ORDER BY latest.sequence_number DESC
                               LIMIT 1
                           ), '') AS last_message
                                        FROM ai_conversations conversation
                                        LEFT JOIN ai_messages message ON message.conversation_id = conversation.id
                                        WHERE conversation.scope_type = 'dashboard'
                                            AND conversation.conversation_kind = 'general'
                      AND conversation.deleted_at IS NULL
                    GROUP BY conversation.id
                    ORDER BY conversation.updated_at DESC, conversation.created_at DESC
                    """
                )
                return [dict(row) for row in cursor.fetchall()]
        except ConversationStoreError:
            raise
        except psycopg.Error as exc:
            raise ConversationStoreError("无法读取对话列表") from exc

    def create_conversation(
        self,
        *,
        title: str,
        model_name: str | None,
        enable_thinking: bool,
    ) -> dict[str, Any]:
        try:
            with self._connect() as connection, connection.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    """
                    INSERT INTO ai_conversations (
                        scope_type, conversation_kind, title, model_name,
                        enable_thinking
                    )
                    VALUES ('dashboard', 'general', %s, %s, %s)
                    RETURNING id
                    """,
                    (title, model_name, enable_thinking),
                )
                return self._get_conversation(cursor, cursor.fetchone()["id"])
        except psycopg.Error as exc:
            raise ConversationStoreError("无法创建对话") from exc

    def get_conversation(self, conversation_id: str) -> dict[str, Any]:
        normalized_id = self._normalize_id(conversation_id)
        try:
            with self._connect() as connection, connection.cursor(row_factory=dict_row) as cursor:
                conversation = self._get_conversation(cursor, normalized_id)
                cursor.execute(
                    """
                    SELECT id::text AS id, sequence_number, role, content,
                           reasoning_content, model_name, metadata, created_at
                    FROM ai_messages
                    WHERE conversation_id = %s
                    ORDER BY sequence_number
                    LIMIT 500
                    """,
                    (normalized_id,),
                )
                conversation["messages"] = [dict(row) for row in cursor.fetchall()]
                return conversation
        except ConversationNotFoundError:
            raise
        except psycopg.Error as exc:
            raise ConversationStoreError("无法读取对话") from exc

    def rename_conversation(self, conversation_id: str, title: str) -> dict[str, Any]:
        normalized_id = self._normalize_id(conversation_id)
        try:
            with self._connect() as connection, connection.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    """
                    UPDATE ai_conversations
                    SET title = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                      AND scope_type = 'dashboard'
                                            AND conversation_kind = 'general'
                      AND deleted_at IS NULL
                    RETURNING id
                    """,
                    (title, normalized_id),
                )
                if cursor.fetchone() is None:
                    raise ConversationNotFoundError("对话不存在")
                return self._get_conversation(cursor, normalized_id)
        except ConversationNotFoundError:
            raise
        except psycopg.Error as exc:
            raise ConversationStoreError("无法重命名对话") from exc

    def delete_conversation(self, conversation_id: str) -> None:
        normalized_id = self._normalize_id(conversation_id)
        try:
            with self._connect() as connection, connection.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE ai_conversations
                    SET deleted_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                      AND scope_type = 'dashboard'
                                            AND conversation_kind = 'general'
                      AND deleted_at IS NULL
                    """,
                    (normalized_id,),
                )
                if cursor.rowcount != 1:
                    raise ConversationNotFoundError("对话不存在")
        except ConversationNotFoundError:
            raise
        except psycopg.Error as exc:
            raise ConversationStoreError("无法删除对话") from exc

    def start_turn(
        self,
        conversation_id: str,
        *,
        content: str,
        model_name: str | None,
        enable_thinking: bool,
    ) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, str]]]:
        normalized_id = self._normalize_id(conversation_id)
        try:
            with self._connect() as connection, connection.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    """
                    SELECT id, title
                    FROM ai_conversations
                    WHERE id = %s
                      AND scope_type = 'dashboard'
                                            AND conversation_kind = 'general'
                      AND deleted_at IS NULL
                    FOR UPDATE
                    """,
                    (normalized_id,),
                )
                current = cursor.fetchone()
                if current is None:
                    raise ConversationNotFoundError("对话不存在")
                cursor.execute(
                    """
                    SELECT COALESCE(MAX(sequence_number), 0) + 1 AS next_sequence
                    FROM ai_messages
                    WHERE conversation_id = %s
                    """,
                    (normalized_id,),
                )
                sequence_number = cursor.fetchone()["next_sequence"]
                cursor.execute(
                    """
                    INSERT INTO ai_messages (
                        conversation_id, sequence_number, role, content, model_name,
                        metadata
                    )
                    VALUES (%s, %s, 'user', %s, %s, %s::jsonb)
                    RETURNING id::text AS id, sequence_number, role, content,
                              reasoning_content, model_name, metadata, created_at
                    """,
                    (
                        normalized_id,
                        sequence_number,
                        content,
                        model_name,
                        '{"enable_thinking": true}' if enable_thinking else '{"enable_thinking": false}',
                    ),
                )
                user_message = dict(cursor.fetchone())
                title = (
                    self._title_from_message(content)
                    if sequence_number == 1 and current["title"] == "新对话"
                    else current["title"]
                )
                cursor.execute(
                    """
                    UPDATE ai_conversations
                    SET title = %s, model_name = %s, enable_thinking = %s,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                    """,
                    (title, model_name, enable_thinking, normalized_id),
                )
                cursor.execute(
                    """
                    SELECT role, content
                    FROM (
                        SELECT sequence_number, role, content
                        FROM ai_messages
                        WHERE conversation_id = %s
                        ORDER BY sequence_number DESC
                        LIMIT 32
                    ) recent
                    ORDER BY sequence_number
                    """,
                    (normalized_id,),
                )
                context = [
                    {"role": row["role"], "content": row["content"]}
                    for row in cursor.fetchall()
                ]
                conversation = self._get_conversation(cursor, normalized_id)
                return conversation, user_message, context
        except ConversationNotFoundError:
            raise
        except psycopg.Error as exc:
            raise ConversationStoreError("无法保存用户消息") from exc

    def finish_turn(
        self,
        conversation_id: str,
        *,
        content: str,
        reasoning_content: str,
        model_name: str | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        normalized_id = self._normalize_id(conversation_id)
        stored_content = content or "[模型未返回文本]"
        try:
            with self._connect() as connection, connection.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    """
                    SELECT id
                    FROM ai_conversations
                    WHERE id = %s
                      AND scope_type = 'dashboard'
                                            AND conversation_kind = 'general'
                      AND deleted_at IS NULL
                    FOR UPDATE
                    """,
                    (normalized_id,),
                )
                if cursor.fetchone() is None:
                    raise ConversationNotFoundError("对话不存在")
                cursor.execute(
                    """
                    SELECT COALESCE(MAX(sequence_number), 0) + 1 AS next_sequence
                    FROM ai_messages
                    WHERE conversation_id = %s
                    """,
                    (normalized_id,),
                )
                sequence_number = cursor.fetchone()["next_sequence"]
                cursor.execute(
                    """
                    INSERT INTO ai_messages (
                        conversation_id, sequence_number, role, content,
                        reasoning_content, model_name
                    )
                    VALUES (%s, %s, 'assistant', %s, %s, %s)
                    RETURNING id::text AS id, sequence_number, role, content,
                              reasoning_content, model_name, metadata, created_at
                    """,
                    (
                        normalized_id,
                        sequence_number,
                        stored_content,
                        reasoning_content,
                        model_name,
                    ),
                )
                assistant_message = dict(cursor.fetchone())
                cursor.execute(
                    """
                    UPDATE ai_conversations
                    SET model_name = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                    """,
                    (model_name, normalized_id),
                )
                return (
                    self._get_conversation(cursor, normalized_id),
                    assistant_message,
                )
        except ConversationNotFoundError:
            raise
        except psycopg.Error as exc:
            raise ConversationStoreError("无法保存模型回复") from exc

    @staticmethod
    def _ensure_mutsu_conversation(
        cursor: psycopg.Cursor[Any], owner_admin_id: str
    ) -> None:
        cursor.execute(
            """INSERT INTO ai_conversations (
                   scope_type, owner_admin_id, conversation_kind, title,
                   model_name, enable_thinking, support_mode
               ) VALUES ('dashboard', %s, 'mutsu', '陆奥', NULL, TRUE, 'ai')
               ON CONFLICT (owner_admin_id)
                   WHERE scope_type = 'dashboard'
                     AND conversation_kind = 'mutsu'
                     AND deleted_at IS NULL
               DO NOTHING""",
            (owner_admin_id,),
        )

    @staticmethod
    def _mutsu_row(
        cursor: psycopg.Cursor[Any],
        owner_admin_id: str,
        *,
        for_update: bool = False,
    ) -> dict[str, Any]:
        cursor.execute(
            f"""SELECT id::text AS id, title, model_name, enable_thinking,
                       support_run_id, created_at, updated_at
                FROM ai_conversations
                WHERE scope_type = 'dashboard'
                  AND conversation_kind = 'mutsu'
                  AND owner_admin_id = %s
                  AND deleted_at IS NULL
                {"FOR UPDATE" if for_update else ""}""",
            (owner_admin_id,),
        )
        row = cursor.fetchone()
        if row is None:
            raise ConversationNotFoundError("陆奥会话不存在")
        return dict(row)

    @staticmethod
    def _mutsu_messages(
        cursor: psycopg.Cursor[Any], conversation_id: str
    ) -> list[dict[str, Any]]:
        cursor.execute(
            """SELECT id::text AS id, sequence_number, role, content,
                      reasoning_content, model_name, metadata, created_at
               FROM ai_messages
               WHERE conversation_id = %s
               ORDER BY sequence_number
               LIMIT 1000""",
            (conversation_id,),
        )
        return [dict(row) for row in cursor.fetchall()]

    @classmethod
    def _mutsu_detail(
        cls, cursor: psycopg.Cursor[Any], conversation: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            **conversation,
            "support_run_id": None,
            "ai_responding": conversation.get("support_run_id") is not None,
            "messages": cls._mutsu_messages(cursor, str(conversation["id"])),
        }

    def get_mutsu_conversation(self, owner_admin_id: str) -> dict[str, Any]:
        try:
            with self._connect() as connection, connection.cursor(row_factory=dict_row) as cursor:
                self._ensure_mutsu_conversation(cursor, owner_admin_id)
                return self._mutsu_detail(
                    cursor, self._mutsu_row(cursor, owner_admin_id)
                )
        except ConversationNotFoundError:
            raise
        except psycopg.Error as exc:
            raise ConversationStoreError("无法读取陆奥会话") from exc

    def start_mutsu_turn(
        self,
        owner_admin_id: str,
        *,
        content: str,
        model_name: str | None,
        enable_thinking: bool,
    ) -> dict[str, Any]:
        run_id = str(uuid4())
        try:
            with self._connect() as connection, connection.cursor(row_factory=dict_row) as cursor:
                self._ensure_mutsu_conversation(cursor, owner_admin_id)
                conversation = self._mutsu_row(
                    cursor, owner_admin_id, for_update=True
                )
                if conversation.get("support_run_id") is not None:
                    raise ConversationBusyError("陆奥正在处理你的上一条消息")
                cursor.execute(
                    """SELECT COALESCE(MAX(sequence_number), 0) + 1 AS sequence
                       FROM ai_messages WHERE conversation_id = %s""",
                    (conversation["id"],),
                )
                sequence = int(cursor.fetchone()["sequence"])
                cursor.execute(
                    """INSERT INTO ai_messages (
                           conversation_id, sequence_number, role, content,
                           model_name, metadata
                       ) VALUES (%s, %s, 'user', %s, %s, %s)
                       RETURNING id::text AS id, sequence_number, role, content,
                                 reasoning_content, model_name, metadata, created_at""",
                    (
                        conversation["id"], sequence, content, model_name,
                        Jsonb({"source": "administrator", "agent": "mutsu"}),
                    ),
                )
                user_message = dict(cursor.fetchone())
                cursor.execute(
                    """UPDATE ai_conversations
                       SET model_name = %s, enable_thinking = %s,
                           support_run_id = %s, updated_at = CURRENT_TIMESTAMP
                       WHERE id = %s""",
                    (model_name, enable_thinking, run_id, conversation["id"]),
                )
                cursor.execute(
                    """SELECT role, content
                       FROM ai_messages
                       WHERE conversation_id = %s
                         AND role IN ('user', 'assistant')
                       ORDER BY sequence_number DESC
                       LIMIT 32""",
                    (conversation["id"],),
                )
                context = [dict(row) for row in reversed(cursor.fetchall())]
                return {
                    "conversation": self._mutsu_detail(
                        cursor, self._mutsu_row(cursor, owner_admin_id)
                    ),
                    "user_message": user_message,
                    "run_id": run_id,
                    "context": context,
                }
        except ConversationStoreError:
            raise
        except psycopg.Error as exc:
            raise ConversationStoreError("无法保存陆奥消息") from exc

    def finish_mutsu_turn(
        self,
        owner_admin_id: str,
        run_id: str,
        *,
        content: str,
        reasoning_content: str,
        model_name: str | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        stored_content = content or "[陆奥未返回文本]"
        try:
            with self._connect() as connection, connection.cursor(row_factory=dict_row) as cursor:
                conversation = self._mutsu_row(
                    cursor, owner_admin_id, for_update=True
                )
                if str(conversation.get("support_run_id") or "") != run_id:
                    raise ConversationStoreError("陆奥会话状态已发生变化")
                cursor.execute(
                    """SELECT COALESCE(MAX(sequence_number), 0) + 1 AS sequence
                       FROM ai_messages WHERE conversation_id = %s""",
                    (conversation["id"],),
                )
                sequence = int(cursor.fetchone()["sequence"])
                cursor.execute(
                    """INSERT INTO ai_messages (
                           conversation_id, sequence_number, role, content,
                           reasoning_content, model_name, metadata
                       ) VALUES (%s, %s, 'assistant', %s, %s, %s, %s)
                       RETURNING id::text AS id, sequence_number, role, content,
                                 reasoning_content, model_name, metadata, created_at""",
                    (
                        conversation["id"], sequence, stored_content,
                        reasoning_content, model_name,
                        Jsonb({"source": "mutsu", "agent": "mutsu"}),
                    ),
                )
                assistant_message = dict(cursor.fetchone())
                cursor.execute(
                    """UPDATE ai_conversations
                       SET model_name = %s, support_run_id = NULL,
                           updated_at = CURRENT_TIMESTAMP
                       WHERE id = %s""",
                    (model_name, conversation["id"]),
                )
                return (
                    self._mutsu_detail(
                        cursor, self._mutsu_row(cursor, owner_admin_id)
                    ),
                    assistant_message,
                )
        except ConversationStoreError:
            raise
        except psycopg.Error as exc:
            raise ConversationStoreError("无法保存陆奥回复") from exc

    def fail_mutsu_turn(self, owner_admin_id: str, run_id: str) -> None:
        try:
            with self._connect() as connection, connection.cursor() as cursor:
                cursor.execute(
                    """UPDATE ai_conversations
                       SET support_run_id = NULL, updated_at = CURRENT_TIMESTAMP
                       WHERE scope_type = 'dashboard'
                         AND conversation_kind = 'mutsu'
                         AND owner_admin_id = %s
                         AND support_run_id = %s""",
                    (owner_admin_id, run_id),
                )
        except psycopg.Error as exc:
            raise ConversationStoreError("无法恢复陆奥会话") from exc
