"""Read-only post-start verification; bytecode cache contents are not attested.

Usage: python scripts/verify-runtime-snapshot.py RELEASE_DIRECTORY
Never use this to authorize redeployment of an already deployed release.
"""
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import sys

SOURCES = {"omni_ai_main_service", "omni_ai_database", "omni_ai_server_interface", "omni_ai_controller"}
CACHE_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\.cpython-\d{2,3}\.pyc\Z")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def regular(path):
    require(stat.S_ISREG(path.lstat().st_mode), f"Not a regular file: {path}")
    return path.read_bytes()


def safe_path(root, relative):
    require(isinstance(relative, str), "Non-string manifest path")
    parts = PurePosixPath(relative).parts
    require(parts and not relative.startswith("/") and ".." not in parts
            and PurePosixPath(relative).as_posix() == relative, f"Unsafe path: {relative}")
    path = root
    for part in parts:
        path = path / part
        require(not path.is_symlink(), f"Symlink: {path}")
    return path


def inventory(tree):
    require(stat.S_ISDIR(tree.lstat().st_mode), f"Not a real directory: {tree}")
    files = set()
    for path in tree.iterdir():
        mode = path.lstat().st_mode
        if stat.S_ISDIR(mode):
            files.update(inventory(path))
        else:
            require(stat.S_ISREG(mode), f"Symlink or special file: {path}")
            files.add(path)
    return files


def verify(root):
    root = Path(root).resolve(strict=True)
    manifest_bytes = regular(root / "snapshot-sha256.json")
    manifest = json.loads(manifest_bytes)
    prepared = json.loads(regular(root / "prepared.json"))
    require(hashlib.sha256(manifest_bytes).hexdigest() == prepared["snapshot_manifest_sha256"],
            "Frozen manifest differs from prepared binding")
    require(set(manifest["sources"]) == SOURCES, "Unexpected source repositories")
    for relative, digest in prepared.get("release_helpers", {}).items():
        require(hashlib.sha256(regular(safe_path(root, relative))).hexdigest() == digest,
                f"Frozen helper changed: {relative}")
    for relative, digest in manifest["files"].items():
        path = safe_path(root, relative)
        require(PurePosixPath(relative).parts[0] in SOURCES, f"Unexpected source path: {relative}")
        require(hashlib.sha256(regular(path)).hexdigest() == digest, f"Source changed: {relative}")
    caches = []
    for name in sorted(SOURCES):
        actual = {path.relative_to(root).as_posix() for path in inventory(root / name)}
        expected = {p for p in manifest["files"] if p.startswith(name + "/")}
        require(expected, f"Empty source manifest: {name}")
        require(not expected - actual, f"Missing source files: {name}")
        for relative in sorted(actual - expected):
            path = root / relative
            cache_dir = root / "omni_ai_controller/omni_ai_controller/__pycache__"
            require(name == "omni_ai_controller" and path.parent == cache_dir
                    and CACHE_NAME.fullmatch(path.name), f"Unexpected file: {relative}")
            module = path.name.split(".cpython-", 1)[0]
            source = (cache_dir.parent / (module + ".py")).relative_to(root).as_posix()
            require(source in expected, f"Bytecode without committed source: {relative}")
            caches.append(relative)
    return {"source_files": len(manifest["files"]), "runtime_cache_files": caches,
            "bytecode_contents_attested": False}


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: verify-runtime-snapshot.py RELEASE_DIRECTORY")
    try:
        result = verify(sys.argv[1])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit(f"Verification FAILED: {exc}") from exc
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()