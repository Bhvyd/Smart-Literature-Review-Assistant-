
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from store import repo
from store.db import connect, init_db


class RecordingStarter:
    def __init__(self):
        self.calls = []

    def __call__(self, settings, **kwargs):
        self.calls.append(kwargs)


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "collections_api.sqlite3"
    init_db(path)
    return path


def client(path, starter=None):
    app = create_app(Settings(db_path=path), warm_query_embedder=False)
    app.state.build_starter = starter or RecordingStarter()
    return TestClient(app)


def test_a_topic_post_returns_202_with_ids_and_a_queued_job(db):
    starter = RecordingStarter()
    with client(db, starter) as api:
        response = api.post(
            "/api/collections",
            json={"query": "retrieval augmented generation", "max_papers": 50},
        )

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "building"
    assert body["collection_id"] and body["job_id"]
    assert starter.calls == [
        {
            "collection_id": body["collection_id"],
            "job_id": body["job_id"],
            "query": "retrieval augmented generation",
            "max_papers": 50,
        }
    ]

    conn = connect(db)
    assert repo.get_collection(conn, body["collection_id"])["status"] == "building"
    job = repo.get_job(conn, body["job_id"])
    assert job["status"] == "queued"
    assert job["progress"] == 0.0
    conn.close()


def test_a_blank_query_is_422_and_starts_nothing(db):
    starter = RecordingStarter()
    with client(db, starter) as api:
        response = api.post("/api/collections", json={"query": "   "})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"
    assert starter.calls == []


def test_max_papers_outside_the_range_is_422(db):
    starter = RecordingStarter()
    with client(db, starter) as api:
        too_small = api.post("/api/collections", json={"query": "q", "max_papers": 5})
        too_big = api.post("/api/collections", json={"query": "q", "max_papers": 201})

    assert too_small.status_code == 422
    assert too_big.status_code == 422
    assert starter.calls == []


def test_upload_mode_is_rejected_with_the_supported_modes(db):
    with client(db) as api:
        response = api.post(
            "/api/collections", json={"query": "q", "mode": "upload"}
        )

    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    assert error["detail"]["supported"] == ["topic"]


def test_a_second_build_for_the_same_query_is_409_with_the_running_ids(db):
    starter = RecordingStarter()
    with client(db, starter) as api:
        first = api.post("/api/collections", json={"query": "same topic"}).json()
        second = api.post("/api/collections", json={"query": "same topic"})

    assert second.status_code == 409
    error = second.json()["error"]
    assert error["code"] == "build_in_progress"
    assert error["detail"]["collection_id"] == first["collection_id"]
    assert error["detail"]["job_id"] == first["job_id"]
    assert len(starter.calls) == 1


def test_collections_are_listed_newest_first(db):
    conn = connect(db)
    first = repo.create_collection(conn, mode="topic", query="older", embedding_model="m")
    second = repo.create_collection(conn, mode="topic", query="newer", embedding_model="m")
    conn.close()

    with client(db) as api:
        payload = api.get("/api/collections").json()

    assert [row["id"] for row in payload["collections"]] == [second, first]
    assert payload["collections"][0]["query"] == "newer"


def test_collection_detail_and_unknown_ids(db):
    conn = connect(db)
    cid = repo.create_collection(conn, mode="topic", query="q", embedding_model="m")
    conn.close()

    with client(db) as api:
        found = api.get(f"/api/collections/{cid}")
        missing = api.get("/api/collections/nope")

    assert found.status_code == 200
    assert found.json()["id"] == cid
    assert found.json()["mode"] == "topic"
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "not_found"


def test_job_status_and_unknown_job(db):
    conn = connect(db)
    cid = repo.create_collection(conn, mode="topic", query="q", embedding_model="m")
    job_id = repo.create_job(conn, collection_id=cid, kind="build")
    repo.update_job(conn, job_id, status="running", progress=0.5, message="ingested 50/100")
    conn.close()

    with client(db) as api:
        found = api.get(f"/api/jobs/{job_id}")
        missing = api.get("/api/jobs/nope")

    assert found.status_code == 200
    assert found.json()["progress"] == 0.5
    assert found.json()["message"] == "ingested 50/100"
    assert missing.status_code == 404
