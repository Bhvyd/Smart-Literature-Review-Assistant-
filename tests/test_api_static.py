
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from store.db import init_db


def _app(tmp_path, dist):
    db = tmp_path / "static.sqlite3"
    init_db(db)
    return create_app(
        Settings(db_path=db), warm_query_embedder=False, static_dir=dist
    )


def test_the_built_app_is_served_at_root(tmp_path):
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<!doctype html><title>litreview</title>", encoding="utf-8")

    with TestClient(_app(tmp_path, dist)) as api:
        response = api.get("/")

    assert response.status_code == 200
    assert "litreview" in response.text


def test_an_unknown_api_path_keeps_the_json_envelope(tmp_path):
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<!doctype html>", encoding="utf-8")

    with TestClient(_app(tmp_path, dist)) as api:
        response = api.get("/api/nope")

    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["error"]["code"] == "not_found"
    assert response.json()["error"]["detail"]["path"] == "nope"


def test_without_a_build_no_mount_is_added(tmp_path):
    dist = tmp_path / "missing"

    with TestClient(_app(tmp_path, dist)) as api:
        root = api.get("/")
        health = api.get("/api/health")

    assert root.status_code == 404
    assert health.status_code == 200
