import pytest

from src.dynamic_risk import current_state


def test_app_startup_seeds_current_data_and_preserves_restart_state(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from src.api import app as api_module

    current_dir = tmp_path / "persistent" / "current"
    database_path = tmp_path / "persistent" / "master" / "supplybeacon_master.db"
    generated_dir = tmp_path / "persistent" / "master" / "generated"
    monkeypatch.setattr(current_state, "CURRENT_DIR", current_dir)
    monkeypatch.setattr(api_module, "CURRENT_DIR", current_dir)
    monkeypatch.setenv("SUPPLYBEACON_MASTER_DB_PATH", str(database_path))
    monkeypatch.setenv("SUPPLYBEACON_MASTER_GENERATED_DIR", str(generated_dir))
    monkeypatch.setenv("SUPPLYBEACON_MASTER_WRITE_ENABLED", "false")
    monkeypatch.setenv("SUPPLYBEACON_LLM_ENABLED", "false")

    with TestClient(api_module.app) as client:
        assert {path.name for path in current_dir.iterdir()} == set(current_state.CURRENT_SEED_FILES)
        assert client.get("/health").status_code == 200
        weather = client.get("/current/weather")
        assert weather.status_code == 200
        assert weather.json()["cities"]
        status = client.get("/master/status")
        assert status.status_code == 200
        assert status.json()["database_initialized"] is True
        assert status.json()["write_enabled"] is False
        assert client.post("/master/products", json={"name": "Blocked"}).status_code == 403

    marker = current_dir / "restart-marker.test"
    marker.write_text("runtime state", encoding="utf-8")
    original_events = (current_dir / "current_events.json").read_bytes()

    with TestClient(api_module.app):
        assert marker.read_text(encoding="utf-8") == "runtime state"
        assert (current_dir / "current_events.json").read_bytes() == original_events
        assert database_path.is_file()
        assert generated_dir.is_dir()


def test_bootstrap_copies_seed_into_empty_persistent_directory(tmp_path, monkeypatch):
    seed_dir = tmp_path / "deploy" / "seed" / "current"
    seed_dir.mkdir(parents=True)
    for filename in current_state.CURRENT_SEED_FILES:
        (seed_dir / filename).write_text("approved seed", encoding="utf-8")
    (seed_dir / "unapproved.cache").write_text("not copied", encoding="utf-8")
    persistent_dir = tmp_path / "persistent" / "current"

    monkeypatch.setattr(current_state, "CURRENT_DIR", persistent_dir)
    monkeypatch.setattr(current_state, "CURRENT_SEED_DIR", seed_dir)

    assert current_state.bootstrap_current_data()
    assert {path.name for path in persistent_dir.iterdir()} == set(current_state.CURRENT_SEED_FILES)
    assert all(path.read_text(encoding="utf-8") == "approved seed" for path in persistent_dir.iterdir())


def test_bootstrap_preserves_nonempty_persistent_directory(tmp_path, monkeypatch):
    seed_dir = tmp_path / "deploy" / "seed" / "current"
    seed_dir.mkdir(parents=True)
    for filename in current_state.CURRENT_SEED_FILES:
        (seed_dir / filename).write_text("seed", encoding="utf-8")
    persistent_dir = tmp_path / "persistent" / "current"
    persistent_dir.mkdir(parents=True)
    (persistent_dir / "current_events.json").write_text("runtime", encoding="utf-8")

    monkeypatch.setattr(current_state, "CURRENT_DIR", persistent_dir)
    monkeypatch.setattr(current_state, "CURRENT_SEED_DIR", seed_dir)

    assert not current_state.bootstrap_current_data()
    assert (persistent_dir / "current_events.json").read_text(encoding="utf-8") == "runtime"


def test_bootstrap_rejects_a_seed_missing_required_files(tmp_path, monkeypatch):
    seed_dir = tmp_path / "deploy" / "seed" / "current"
    seed_dir.mkdir(parents=True)
    (seed_dir / "current_events.json").write_text("[]", encoding="utf-8")
    persistent_dir = tmp_path / "persistent" / "current"
    monkeypatch.setattr(current_state, "CURRENT_DIR", persistent_dir)
    monkeypatch.setattr(current_state, "CURRENT_SEED_DIR", seed_dir)

    with pytest.raises(RuntimeError, match="incomplete"):
        current_state.bootstrap_current_data()
