
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

client = TestClient(create_app(Settings()))


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["version"]
    assert "embedding_model" in body


def test_health_reports_generation_availability_without_leaking_the_key():
    body = client.get("/api/health").json()
    assert isinstance(body["generation_available"], bool)
    assert "sk-" not in str(body)


def test_without_a_credential_generation_is_unavailable_and_says_why():
    body = client.get("/api/health").json()
    assert body["generation_available"] is False
    assert body["generation_reason"] == "no_api_key"
    assert body["generation_last_error"] is None


def test_a_credential_makes_the_answer_path_available_without_being_echoed():
    keyed = TestClient(create_app(Settings(llm_api_key="sk-not-a-real-key")))
    body = keyed.get("/api/health").json()
    assert body["generation_available"] is True
    assert body["generation_reason"] is None
    assert "sk-not-a-real-key" not in str(body)


def test_unknown_route_uses_error_envelope():
    r = client.get("/api/nope")
    assert r.status_code == 404
    assert "error" in r.json()
    assert r.json()["error"]["code"] == "not_found"


def test_openapi_schema_loads():
    assert client.get("/openapi.json").status_code == 200


def test_settings_load_from_env_file(tmp_path):
    env = tmp_path / ".env"
    env.write_text(
        "OPENAI_API_KEY=sk-test-value\nEMBEDDING_MODEL=tiny-test-model\nLLM_MODEL=some-model\n",
        encoding="utf-8",
    )
    s = Settings.from_env(env_path=env)
    assert s.llm_api_key == "sk-test-value"
    assert s.embedding_model == "tiny-test-model"
    assert s.llm_model == "some-model"


def test_missing_env_file_is_not_fatal(tmp_path):
    s = Settings.from_env(env_path=tmp_path / "absent.env")
    assert s.llm_api_key is None
    assert s.llm_model
    assert s.generation_available is False


def test_placeholder_key_is_treated_as_missing(tmp_path):
    env = tmp_path / ".env"
    env.write_text("OPENAI_API_KEY=replace-me\n", encoding="utf-8")
    assert Settings.from_env(env_path=env).llm_api_key is None


def test_comments_and_quotes_are_handled(tmp_path):
    env = tmp_path / ".env"
    env.write_text(
        "# a comment\n\nOPENAI_API_KEY=\"sk-quoted\"\nDB_PATH='./data/custom.sqlite3'\n",
        encoding="utf-8",
    )
    s = Settings.from_env(env_path=env)
    assert s.llm_api_key == "sk-quoted"
    assert s.db_path == Path("./data/custom.sqlite3")


def _delete_client(tmp_path):
    from store import repo
    from store.db import connect, init_db

    path = tmp_path / "del.sqlite3"
    init_db(path)
    app = create_app(Settings(db_path=path, data_dir=tmp_path), warm_query_embedder=False)
    conn = connect(path)
    return TestClient(app), conn, repo


def _ready_collection(conn, repo, query, mode="topic"):
    cid = repo.create_collection(conn, mode=mode, query=query, embedding_model="m")
    repo.set_collection_status(conn, cid, "ready")
    return cid


def test_delete_removes_the_collection_its_uploaded_pdfs_and_its_cached_index(tmp_path):
    from services import retrieval
    from services.upload import upload_dir

    client, conn, repo = _delete_client(tmp_path)
    keep = _ready_collection(conn, repo, "keep")
    drop = _ready_collection(conn, repo, "drop", mode="upload")
    for cid in (keep, drop):
        folder = upload_dir(tmp_path, cid)
        folder.mkdir(parents=True)
        (folder / ("a" * 64 + ".pdf")).write_bytes(b"%PDF-")
    key = (str(tmp_path / "del.sqlite3"), drop, None, 0, 0, 0)
    retrieval._cache[key] = (None, None)
    with client:
        response = client.delete(f"/api/collections/{drop}")
        listed = [row["id"] for row in client.get("/api/collections").json()["collections"]]
    assert response.status_code == 200
    body = response.json()
    assert body["collection_id"] == drop
    assert body["deleted"]["collections"] == 1 and body["deleted"]["uploaded_files"] == 1
    assert listed == [keep]
    assert not upload_dir(tmp_path, drop).exists() and upload_dir(tmp_path, keep).exists()
    assert key not in retrieval._cache


def test_delete_of_an_unknown_collection_is_404_in_the_error_envelope(tmp_path):
    client, _, _ = _delete_client(tmp_path)
    with client:
        response = client.delete("/api/collections/nope")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_delete_of_a_running_build_is_409_and_changes_nothing(tmp_path):
    client, conn, repo = _delete_client(tmp_path)
    cid = repo.create_collection(conn, mode="topic", query="busy", embedding_model="m")
    repo.create_job(conn, collection_id=cid, kind="build")
    with client:
        response = client.delete(f"/api/collections/{cid}")
        still = client.get(f"/api/collections/{cid}")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "build_in_progress"
    assert still.status_code == 200


def test_delete_removes_the_answers_and_the_cached_projection_but_not_another_collections(tmp_path):
    client, conn, repo = _delete_client(tmp_path)
    keep = _ready_collection(conn, repo, "keep")
    drop = _ready_collection(conn, repo, "drop")
    for cid in (keep, drop):
        conn.execute(
            "INSERT INTO answers (id, collection_id, question, created_at) VALUES (?, ?, 'q', 'now')",
            (f"a-{cid}", cid),
        )
        conn.execute(
            "INSERT INTO projections (collection_id, method, params_hash, payload, created_at) "
            "VALUES (?, 'umap', 'h', '{}', 'now')",
            (cid,),
        )
    conn.commit()
    with client:
        body = client.delete(f"/api/collections/{drop}").json()
    assert body["deleted"]["answers"] == 1 and body["deleted"]["projections"] == 1
    for table in ("answers", "projections"):
        remaining = [r[0] for r in conn.execute(f"SELECT collection_id FROM {table}")]
        assert remaining == [keep]
