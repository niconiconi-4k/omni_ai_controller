from __future__ import annotations

import hashlib
import hmac
import re
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

import psycopg
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError
from psycopg.rows import dict_row

PERMISSIONS = (
    "accounts.manage",
    "services.control",
    "business.manage",
    "support.manage",
    "quantization.manage",
)
USERNAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9._-]{2,63}$")


class AdminAccountError(RuntimeError):
    pass


class AdminAccountConflict(AdminAccountError):
    pass


class AdminAccountStore:
    def __init__(
        self, *, host: str, port: int, database: str, user: str, password: str, secret: str
    ) -> None:
        self.host, self.port, self.database = host, port, database
        self.user, self.password, self.secret = user, password, secret
        self.hasher = PasswordHasher(time_cost=3, memory_cost=65536, parallelism=2)
        self.dummy_hash = self.hasher.hash(self._pepper_password("unknown-admin-password"))

    def _connect(self) -> psycopg.Connection[Any]:
        if not self.password or not self.secret:
            raise AdminAccountError("Administrator account storage is not configured")
        try:
            return psycopg.connect(
                host=self.host, port=self.port, dbname=self.database,
                user=self.user, password=self.password, connect_timeout=5,
            )
        except psycopg.Error as exc:
            raise AdminAccountError("Administrator account storage is unavailable") from exc

    def _token_hash(self, token: str) -> str:
        return hmac.new(self.secret.encode(), token.encode(), hashlib.sha256).hexdigest()

    def _password_hash(self, password: str) -> str:
        if not 12 <= len(password) <= 1024:
            raise ValueError("Password must contain 12 to 1024 characters")
        return self.hasher.hash(self._pepper_password(password))

    def _pepper_password(self, password: str) -> str:
        return hmac.new(self.secret.encode(), password.encode(), hashlib.sha256).hexdigest()

    def _verify_password(self, encoded: str, password: str) -> bool:
        try:
            return self.hasher.verify(encoded, self._pepper_password(password))
        except (InvalidHashError, VerificationError, VerifyMismatchError):
            return False

    @staticmethod
    def public(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": str(row["id"]), "username": row["username"],
            "permissions": list(PERMISSIONS) if row["is_super"] else list(row["permissions"]),
            "is_super": row["is_super"],
            "created_at": row.get("created_at"),
        }

    def bootstrap_super(self, password: str) -> str | None:
        """Create the immutable Mutsu account once; return its one-time token."""
        encoded = self._password_hash(password)
        token = secrets.token_urlsafe(48)
        try:
            with self._connect() as connection, connection.cursor(row_factory=dict_row) as cursor:
                cursor.execute("SELECT id FROM controller_admin_accounts WHERE is_super = TRUE")
                if cursor.fetchone():
                    return None
                cursor.execute(
                    """INSERT INTO controller_admin_accounts
                       (username, password_hash, token_hash, permissions, is_super)
                       VALUES ('Mutsu', %s, %s, '{}', TRUE)""",
                    (encoded, self._token_hash(token)),
                )
            return token
        except psycopg.Error as exc:
            raise AdminAccountError("Unable to bootstrap the super administrator") from exc

    def authenticate(self, username: str, password: str, token: str) -> dict[str, Any] | None:
        try:
            with self._connect() as connection, connection.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    "SELECT * FROM controller_admin_accounts WHERE lower(username) = lower(%s)",
                    (username,),
                )
                row = cursor.fetchone()
                if not row:
                    self._verify_password(self.dummy_hash, password)
                    return None
                if row["locked_until"] and row["locked_until"] > datetime.now(timezone.utc):
                    return None
                valid = self._verify_password(row["password_hash"], password)
                valid = valid & hmac.compare_digest(row["token_hash"], self._token_hash(token))
                if not valid:
                    if not row["is_super"]:
                        cursor.execute(
                            """UPDATE controller_admin_accounts SET
                               failed_attempts = failed_attempts + 1,
                               locked_until = CASE WHEN failed_attempts + 1 >= 5
                                 THEN CURRENT_TIMESTAMP + INTERVAL '15 minutes'
                                 ELSE NULL END
                               WHERE id = %s""",
                            (row["id"],),
                        )
                    return None
                if not row["is_super"]:
                    cursor.execute(
                        """UPDATE controller_admin_accounts SET failed_attempts = 0,
                           locked_until = NULL WHERE id = %s""",
                        (row["id"],),
                    )
                return self.public(row) | {"auth_version": row["auth_version"]}
        except psycopg.Error as exc:
            raise AdminAccountError("Unable to authenticate administrator") from exc

    def session_account(self, account_id: str, version: int) -> dict[str, Any] | None:
        try:
            with self._connect() as connection, connection.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    "SELECT * FROM controller_admin_accounts WHERE id = %s AND auth_version = %s",
                    (account_id, version),
                )
                row = cursor.fetchone()
                return self.public(row) if row else None
        except (ValueError, psycopg.Error) as exc:
            raise AdminAccountError("Unable to validate administrator session") from exc

    def list_accounts(self) -> list[dict[str, Any]]:
        try:
            with self._connect() as connection, connection.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    "SELECT * FROM controller_admin_accounts WHERE is_super = FALSE ORDER BY username"
                )
                return [self.public(row) for row in cursor.fetchall()]
        except psycopg.Error as exc:
            raise AdminAccountError("Unable to list administrators") from exc

    def create(self, username: str, password: str, permissions: list[str]) -> dict[str, Any]:
        username = username.strip()
        if not USERNAME_RE.fullmatch(username) or username.lower() == "mutsu":
            raise ValueError("Invalid administrator username")
        self.validate_permissions(permissions)
        encoded = self._password_hash(password)
        token = secrets.token_urlsafe(48)
        try:
            with self._connect() as connection, connection.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    """INSERT INTO controller_admin_accounts
                       (username, password_hash, token_hash, permissions)
                       VALUES (%s, %s, %s, %s) RETURNING *""",
                    (username, encoded, self._token_hash(token), permissions),
                )
                return {"account": self.public(dict(cursor.fetchone())), "token": token}
        except psycopg.errors.UniqueViolation as exc:
            raise AdminAccountConflict("Username already exists") from exc
        except psycopg.Error as exc:
            raise AdminAccountError("Unable to create administrator") from exc

    @staticmethod
    def validate_permissions(permissions: list[str]) -> None:
        if len(permissions) != len(set(permissions)) or set(permissions) - set(PERMISSIONS):
            raise ValueError("Invalid administrator permissions")

    def update_permissions(self, account_id: str, permissions: list[str]) -> dict[str, Any]:
        self.validate_permissions(permissions)
        try:
            with self._connect() as connection, connection.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    """UPDATE controller_admin_accounts SET permissions = %s,
                       auth_version = auth_version + 1, updated_at = CURRENT_TIMESTAMP
                       WHERE id = %s AND is_super = FALSE RETURNING *""",
                    (permissions, account_id),
                )
                row = cursor.fetchone()
                if not row:
                    raise AdminAccountConflict("Administrator not found or immutable")
                return self.public(dict(row))
        except AdminAccountConflict:
            raise
        except psycopg.Error as exc:
            raise AdminAccountError("Unable to update administrator") from exc

    def rotate_token(self, account_id: str) -> dict[str, str]:
        token = secrets.token_urlsafe(48)
        try:
            with self._connect() as connection, connection.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    """UPDATE controller_admin_accounts SET token_hash = %s,
                       auth_version = auth_version + 1, updated_at = CURRENT_TIMESTAMP
                       WHERE id = %s AND is_super = FALSE RETURNING id""",
                    (self._token_hash(token), account_id),
                )
                if not cursor.fetchone():
                    raise AdminAccountConflict("Administrator not found or immutable")
            return {"token": token}
        except AdminAccountConflict:
            raise
        except psycopg.Error as exc:
            raise AdminAccountError("Unable to rotate administrator token") from exc

    def update_password(self, account_id: str, password: str) -> None:
        encoded = self._password_hash(password)
        try:
            with self._connect() as connection, connection.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    """UPDATE controller_admin_accounts SET password_hash = %s,
                       auth_version = auth_version + 1, updated_at = CURRENT_TIMESTAMP
                       WHERE id = %s AND is_super = FALSE RETURNING id""",
                    (encoded, account_id),
                )
                if not cursor.fetchone():
                    raise AdminAccountConflict("Administrator not found or immutable")
        except AdminAccountConflict:
            raise
        except psycopg.Error as exc:
            raise AdminAccountError("Unable to reset administrator password") from exc

    def delete(self, account_id: str) -> None:
        try:
            with self._connect() as connection, connection.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    "DELETE FROM controller_admin_accounts WHERE id = %s AND is_super = FALSE RETURNING id",
                    (account_id,),
                )
                if not cursor.fetchone():
                    raise AdminAccountConflict("Administrator not found or immutable")
        except AdminAccountConflict:
            raise
        except psycopg.Error as exc:
            raise AdminAccountError("Unable to delete administrator") from exc
