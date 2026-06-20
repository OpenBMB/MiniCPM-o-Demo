from pathlib import Path


def test_session_dir_maps_backend_session_id_directly(monkeypatch, tmp_path):
    import gateway

    class _Service:
        data_dir = "data"

    class _Config:
        service = _Service()
        data_dir = "data"

    backend_dir = tmp_path / "data" / "sessions" / "sess_abc"

    monkeypatch.setattr(gateway, "_BASE_DIR", str(tmp_path))
    monkeypatch.setattr(gateway, "get_config", lambda: _Config(), raising=False)

    assert Path(gateway._session_dir("sess_abc")) == backend_dir.resolve()
