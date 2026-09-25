from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time


def _encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def issue_admin_session(secret: str, ttl_hours: int, account_id: str, version: int) -> tuple[str, str]:
    csrf_token = secrets.token_urlsafe(32)
    payload = {
        "expires_at": int(time.time()) + ttl_hours * 3600,
        "csrf_hash": hashlib.sha256(csrf_token.encode("utf-8")).hexdigest(),
        "account_id": account_id,
        "version": version,
        "nonce": secrets.token_urlsafe(16),
    }
    body = _encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signature = _encode(
        hmac.new(secret.encode("utf-8"), body.encode("ascii"), hashlib.sha256).digest()
    )
    return f"{body}.{signature}", csrf_token


def validate_admin_session(secret: str, token: str) -> dict[str, str | int] | None:
    if not secret or not token:
        return None
    try:
        body, supplied_signature = token.split(".", 1)
        expected_signature = _encode(
            hmac.new(
                secret.encode("utf-8"), body.encode("ascii"), hashlib.sha256
            ).digest()
        )
        if not hmac.compare_digest(supplied_signature, expected_signature):
            return None
        payload = json.loads(_decode(body))
        if int(payload["expires_at"]) <= int(time.time()):
            return None
        csrf_hash = str(payload["csrf_hash"])
        account_id = str(payload["account_id"])
        version = int(payload["version"])
        if len(csrf_hash) != 64 or not account_id or version < 1:
            return None
        return {"csrf_hash": csrf_hash, "account_id": account_id, "version": version}
    except (ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def validate_admin_csrf(
    expected_hash: str, cookie_token: str, header_token: str
) -> bool:
    if not cookie_token or not header_token:
        return False
    if not hmac.compare_digest(cookie_token, header_token):
        return False
    actual_hash = hashlib.sha256(header_token.encode("utf-8")).hexdigest()
    return hmac.compare_digest(actual_hash, expected_hash)