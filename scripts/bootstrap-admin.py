#!/usr/bin/env python3
"""One-time, interactive bootstrap of the immutable Mutsu administrator.

Run after migration 012 as root. The password is never passed on the command line,
written to a file, or recorded in the shell history. Keep the issued token private.
"""
from __future__ import annotations

import getpass
from pathlib import Path

from omni_ai_controller.admin_account_store import AdminAccountStore


def read_environment(path: Path) -> dict[str, str]:
    return dict(
        line.split("=", 1)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#") and "=" in line
    )


def main() -> None:
    controller = read_environment(Path("/etc/omni-ai-controller/service.env"))
    database = read_environment(Path("/var/lib/omni-ai/config/database.env"))
    store = AdminAccountStore(
        host=controller.get("OMNI_CONVERSATION_DATABASE_HOST", "127.0.0.1"),
        port=int(controller.get("OMNI_CONVERSATION_DATABASE_PORT", "15432")),
        database=database["DATABASE_NAME"],
        user=database["DATABASE_USER"],
        password=database["DATABASE_PASSWORD"],
        secret=controller["OMNI_ADMIN_TOKEN"],
    )
    password = getpass.getpass("Initial Mutsu password: ")
    token = store.bootstrap_super(password)
    if token is None:
        print("Mutsu already exists; the immutable token cannot be recovered or changed.")
    else:
        print(f"Mutsu one-time personal token: {token}")


if __name__ == "__main__":
    main()
