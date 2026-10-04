import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/verify-runtime-snapshot.py"
spec = importlib.util.spec_from_file_location("runtime_snapshot_verifier", SCRIPT)
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


def bind(root, manifest):
    raw = json.dumps(manifest).encode()
    (root / "snapshot-sha256.json").write_bytes(raw)
    (root / "prepared.json").write_text(json.dumps({"snapshot_manifest_sha256": hashlib.sha256(raw).hexdigest()}))


@pytest.fixture
def release(tmp_path):
    manifest = {"sources": {name: "test-commit" for name in verifier.SOURCES}, "files": {}}
    for name in verifier.SOURCES:
        path = tmp_path / name / name / "__init__.py"
        path.parent.mkdir(parents=True)
        path.write_text("# frozen\n")
        manifest["files"][path.relative_to(tmp_path).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    bind(tmp_path, manifest)
    return tmp_path


def cache(root, filename="__init__.cpython-312.pyc"):
    path = root / "omni_ai_controller/omni_ai_controller/__pycache__" / filename
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(b"runtime cache content is explicitly not attested")
    return path


def test_pristine(release):
    assert verifier.verify(release)["source_files"] == 4


def test_runtime_cache_is_separate_from_source_attestation(release):
    path = cache(release)
    result = verifier.verify(release)
    assert result["runtime_cache_files"] == [path.relative_to(release).as_posix()]
    assert result["bytecode_contents_attested"] is False


@pytest.mark.parametrize("filename", ["evil.py", "evil.txt", "__init__.pyc", "missing.cpython-312.pyc", "__init__.cpython-312.pyc.bak", "__init__.cpython-312.opt-1.pyc"])
def test_reject_extra_files(release, filename):
    cache(release, filename)
    with pytest.raises(ValueError):
        verifier.verify(release)


@pytest.mark.parametrize("kind", ["changed", "missing", "source_symlink", "directory_symlink", "root_symlink", "cache_symlink", "fifo", "other_repository_cache", "nested_cache"])
def test_reject_source_and_inventory_anomalies(release, kind):
    source = release / "omni_ai_controller/omni_ai_controller/__init__.py"
    if kind == "changed":
        source.write_text("# altered\n")
    elif kind == "missing":
        source.unlink()
    elif kind == "source_symlink":
        source.unlink()
        source.symlink_to(release / "prepared.json")
    elif kind == "directory_symlink":
        (release / "omni_ai_controller/link").symlink_to(release, target_is_directory=True)
    elif kind == "root_symlink":
        tree = release / "omni_ai_database"
        tree.rename(release / "saved")
        tree.symlink_to(release / "saved", target_is_directory=True)
    elif kind == "cache_symlink":
        path = cache(release)
        path.unlink()
        path.symlink_to(source)
    elif kind == "fifo":
        import os
        os.mkfifo(source.parent / "unexpected")
    else:
        name = "omni_ai_main_service" if kind == "other_repository_cache" else "omni_ai_controller"
        path = release / name / "nested/__pycache__/__init__.cpython-312.pyc"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"extra")
    with pytest.raises((ValueError, FileNotFoundError)):
        verifier.verify(release)


@pytest.mark.parametrize("relative", ["../escape", "/tmp/escape", "omni_ai_controller/../escape", "omni_ai_controller//bad", "omni_ai_controller/./bad"])
def test_reject_noncanonical_manifest_paths(release, relative):
    manifest = json.loads((release / "snapshot-sha256.json").read_text())
    manifest["files"][relative] = "invalid"
    bind(release, manifest)
    with pytest.raises(ValueError):
        verifier.verify(release)


def test_reject_manifest_binding_change(release):
    (release / "snapshot-sha256.json").write_text("{}")
    with pytest.raises(ValueError, match="binding"):
        verifier.verify(release)


def test_helper_hashes(release):
    prepared = json.loads((release / "prepared.json").read_text())
    helper = release / "deploy.sh"
    helper.write_text("frozen")
    prepared["release_helpers"] = {"deploy.sh": hashlib.sha256(helper.read_bytes()).hexdigest()}
    (release / "prepared.json").write_text(json.dumps(prepared))
    verifier.verify(release)
    helper.write_text("changed")
    with pytest.raises(ValueError, match="helper"):
        verifier.verify(release)


def test_optimized_python_does_not_disable_checks(release):
    (release / "omni_ai_controller/omni_ai_controller/__init__.py").write_text("changed")
    result = subprocess.run([sys.executable, "-O", str(SCRIPT), str(release)], capture_output=True, text=True)
    assert result.returncode != 0
    assert "Source changed" in result.stderr