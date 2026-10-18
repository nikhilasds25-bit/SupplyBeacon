from pathlib import Path

import pytest

from deploy.audit_seed import audit
from deploy.start import configuration


@pytest.fixture(autouse=True)
def clean_deployment_environment(monkeypatch):
    for name in ("PORT", "RAILWAY_ENVIRONMENT_ID", "RAILWAY_VOLUME_MOUNT_PATH",
                 "SUPPLYBEACON_MASTER_WRITE_ENABLED", "SUPPLYBEACON_LLM_ENABLED",
                 "SUPPLYBEACON_ALLOWED_ORIGINS", "LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2"):
        # Record absent keys too, so configuration() cannot leak direct env
        # mutations into later tests through setdefault or assignment.
        import os
        monkeypatch.setenv(name, os.environ.get(name, ""))
        monkeypatch.delenv(name, raising=False)


def test_public_defaults_and_railway_port(monkeypatch):
    import os
    assert configuration() == 8000
    assert os.environ["SUPPLYBEACON_ALLOWED_ORIGINS"] == "https://supply-beacon.vercel.app"
    monkeypatch.setenv("PORT", "18273")
    assert configuration() == 18273


def test_entrypoint_executes_uvicorn_with_configured_port(monkeypatch):
    from deploy import start
    monkeypatch.setenv("PORT", "18273")
    calls = []
    monkeypatch.setattr(start.os, "execv", lambda executable, arguments: calls.append(arguments))
    start.main()
    assert calls[0][1:] == ["-m", "uvicorn", "src.api.app:app", "--host", "0.0.0.0",
                           "--port", "18273", "--workers", "1"]


@pytest.mark.parametrize("name", ["SUPPLYBEACON_MASTER_WRITE_ENABLED", "SUPPLYBEACON_LLM_ENABLED"])
def test_public_entrypoint_rejects_unsafe_flags(monkeypatch, name):
    monkeypatch.setenv(name, "true")
    with pytest.raises(RuntimeError, match="must be false"):
        configuration()


@pytest.mark.parametrize("value", ["0", "65536", "not-a-port"])
def test_invalid_port_fails_before_startup(monkeypatch, value):
    monkeypatch.setenv("PORT", value)
    with pytest.raises(RuntimeError, match="PORT"):
        configuration()


def test_railway_requires_volume_and_paths_inside_it(monkeypatch, tmp_path):
    monkeypatch.setenv("RAILWAY_ENVIRONMENT_ID", "configuration-test")
    with pytest.raises(RuntimeError, match="Attach a Railway volume"):
        configuration()
    monkeypatch.setenv("RAILWAY_VOLUME_MOUNT_PATH", str(tmp_path))
    for name, relative in (("SUPPLYBEACON_CURRENT_DATA_DIR", "current"),
                           ("SUPPLYBEACON_MASTER_DB_PATH", "master/master.db"),
                           ("SUPPLYBEACON_MASTER_GENERATED_DIR", "master/generated")):
        monkeypatch.setenv(name, str(tmp_path / relative))
    assert configuration() == 8000
    monkeypatch.setenv("SUPPLYBEACON_CURRENT_DATA_DIR", str(tmp_path.parent / "outside"))
    with pytest.raises(RuntimeError, match="inside the Railway volume"):
        configuration()


def test_reviewed_seed_checksums_and_allowlist():
    assert audit()["files"] == 9


def test_seed_audit_rejects_extra_files(tmp_path):
    import shutil
    root = Path(__file__).resolve().parents[1]
    shutil.copytree(root / "deploy/seed", tmp_path / "deploy/seed")
    shutil.copyfile(root / "deploy/seed-manifest.json", tmp_path / "deploy/seed-manifest.json")
    (tmp_path / "deploy/seed/current/unapproved.json").write_text("{}")
    with pytest.raises(ValueError, match="allowlist"):
        audit(tmp_path)
