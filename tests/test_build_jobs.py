
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from services import build_jobs
from services.collection import CollectionBuildError
from store.db import connect, init_db
from store import repo


@pytest.fixture
def conn(tmp_path):
    path = tmp_path / "build.sqlite3"
    init_db(path)
    return connect(path)


def test_prepare_creates_a_building_collection_and_a_queued_job(conn, tmp_path):
    collection_id, job_id = build_jobs.prepare_topic_build(
        conn, query="retrieval augmented generation", settings=_settings(tmp_path)
    )

    collection = repo.get_collection(conn, collection_id)
    job = repo.get_job(conn, job_id)
    assert collection["status"] == "building"
    assert collection["mode"] == "topic"
    assert job["status"] == "queued"
    assert job["collection_id"] == collection_id


def test_prepare_with_reuse_false_always_makes_a_new_collection(conn, tmp_path):
    first, _ = build_jobs.prepare_topic_build(
        conn, query="q", settings=_settings(tmp_path)
    )
    repo.set_collection_status(conn, first, "ready")

    second, _ = build_jobs.prepare_topic_build(
        conn, query="q", settings=_settings(tmp_path)
    )
    assert second != first


def _age(conn, job_id, seconds):
    from datetime import datetime, timedelta, timezone

    old = (datetime.now(timezone.utc) - timedelta(seconds=seconds)).isoformat(timespec="seconds")
    conn.execute("UPDATE jobs SET updated_at=? WHERE id=?", (old, job_id))
    conn.commit()


def test_a_build_whose_worker_went_quiet_is_failed(conn, tmp_path):
    collection_id, job_id = build_jobs.prepare_topic_build(
        conn, query="q", settings=_settings(tmp_path)
    )
    repo.update_job(conn, job_id, status="running", progress=0.4)
    _age(conn, job_id, repo.STALE_BUILD_SECONDS + 60)

    assert build_jobs.reconcile_interrupted_builds(conn) == (1, 1)
    collection = repo.get_collection(conn, collection_id)
    assert collection["status"] == "failed"
    assert "interrupted" in collection["error"]
    assert repo.get_job(conn, job_id)["status"] == "failed"

    assert build_jobs.reconcile_interrupted_builds(conn) == (0, 0)


def test_a_live_build_owned_by_another_process_is_left_alone(conn, tmp_path):
    collection_id, job_id = build_jobs.prepare_topic_build(
        conn, query="q", settings=_settings(tmp_path)
    )
    repo.update_job(conn, job_id, status="running", progress=0.96)

    assert build_jobs.reconcile_interrupted_builds(conn) == (0, 0)
    assert repo.get_collection(conn, collection_id)["status"] == "building"
    assert repo.get_job(conn, job_id)["status"] == "running"


def test_every_progress_update_refreshes_the_heartbeat(conn, tmp_path):
    _, job_id = build_jobs.prepare_topic_build(conn, query="q", settings=_settings(tmp_path))
    _age(conn, job_id, 10_000)
    repo.update_job(conn, job_id, status="running", progress=0.1)
    assert build_jobs.reconcile_interrupted_builds(conn) == (0, 0)


def test_a_finished_build_survives_the_reconcile(conn, tmp_path):
    collection_id, job_id = build_jobs.prepare_topic_build(
        conn, query="q", settings=_settings(tmp_path)
    )
    repo.set_collection_status(conn, collection_id, "ready")
    repo.update_job(conn, job_id, status="done", progress=1.0)

    assert build_jobs.reconcile_interrupted_builds(conn) == (0, 0)
    assert repo.get_collection(conn, collection_id)["status"] == "ready"
    assert repo.get_job(conn, job_id)["status"] == "done"


def test_app_startup_fails_orphaned_builds(tmp_path):
    path = tmp_path / "startup.sqlite3"
    init_db(path)
    conn = connect(path)
    collection_id, job_id = build_jobs.prepare_topic_build(
        conn, query="q", settings=_settings(tmp_path)
    )
    _age(conn, job_id, repo.STALE_BUILD_SECONDS + 60)
    conn.close()

    with TestClient(create_app(Settings(db_path=path), warm_query_embedder=False)):
        pass

    conn = connect(path)
    assert repo.get_collection(conn, collection_id)["status"] == "failed"
    assert repo.get_job(conn, job_id)["status"] == "failed"
    conn.close()


def test_prepare_with_reuse_true_rebuilds_the_ready_collection(conn, tmp_path):
    first, _ = build_jobs.prepare_topic_build(
        conn, query="q", settings=_settings(tmp_path)
    )
    repo.set_collection_status(conn, first, "ready")

    again, _ = build_jobs.prepare_topic_build(
        conn, query="q", settings=_settings(tmp_path), reuse=True
    )
    assert again == first
    assert repo.get_collection(conn, first)["status"] == "building"


def test_the_worker_calls_build_collection_with_the_prepared_ids(conn, tmp_path, monkeypatch):
    collection_id, job_id = build_jobs.prepare_topic_build(
        conn, query="q", settings=_settings(tmp_path)
    )
    calls = {}

    def fake_build(build_conn, **kwargs):
        calls.update(kwargs)
        calls["conn"] = build_conn
        calls["is_separate_connection"] = build_conn is not conn
        build_conn.execute("SELECT 1").fetchone()
        repo.update_job(build_conn, job_id, status="done", progress=1.0, message="ready")
        repo.set_collection_status(build_conn, collection_id, "ready")

    monkeypatch.setattr(build_jobs, "build_collection", fake_build)
    build_jobs._run(_settings(tmp_path), collection_id, job_id, "q", 42)

    assert calls["collection_id"] == collection_id
    assert calls["job_id"] == job_id
    assert calls["query"] == "q"
    assert calls["limit"] == 42
    assert calls["is_separate_connection"] is True
    assert repo.get_job(conn, job_id)["status"] == "done"


def test_a_failed_collection_build_keeps_the_persisted_reason(conn, tmp_path, monkeypatch):
    collection_id, job_id = build_jobs.prepare_topic_build(
        conn, query="q", settings=_settings(tmp_path)
    )

    def fake_build(build_conn, **kwargs):
        repo.set_collection_status(build_conn, collection_id, "failed", error="arxiv unreachable")
        repo.update_job(build_conn, job_id, status="failed", error="arxiv unreachable")
        raise CollectionBuildError("arxiv unreachable")

    monkeypatch.setattr(build_jobs, "build_collection", fake_build)
    build_jobs._run(_settings(tmp_path), collection_id, job_id, "q", 10)

    assert repo.get_collection(conn, collection_id)["status"] == "failed"
    assert repo.get_job(conn, job_id)["error"] == "arxiv unreachable"


def test_an_unexpected_crash_never_leaves_the_job_queued(conn, tmp_path, monkeypatch):
    collection_id, job_id = build_jobs.prepare_topic_build(
        conn, query="q", settings=_settings(tmp_path)
    )

    def fake_build(*args, **kwargs):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(build_jobs, "build_collection", fake_build)
    build_jobs._run(_settings(tmp_path), collection_id, job_id, "q", 10)

    assert repo.get_collection(conn, collection_id)["status"] == "failed"
    assert repo.get_job(conn, job_id)["status"] == "failed"
    assert "disk on fire" in repo.get_job(conn, job_id)["error"]


def _settings(tmp_path):
    from app.config import Settings

    return Settings(db_path=tmp_path / "build.sqlite3", models_dir=tmp_path / "models")


def test_a_crashed_build_stops_blocking_its_topic_without_a_restart(tmp_path):
    path = tmp_path / "lazy.sqlite3"
    init_db(path)
    conn = connect(path)
    _, job_id = build_jobs.prepare_topic_build(conn, query="stuck topic", settings=_settings(tmp_path))
    _age(conn, job_id, repo.STALE_BUILD_SECONDS + 60)

    app = create_app(Settings(db_path=path, data_dir=tmp_path), warm_query_embedder=False)
    app.state.build_starter = lambda *a, **k: None
    with TestClient(app) as client:
        response = client.post("/api/collections", json={"query": "stuck topic", "max_papers": 10})
    assert response.status_code == 202


def _wait_for(predicate, timeout=5.0):
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_a_job_waiting_for_the_build_slot_keeps_a_fresh_heartbeat_and_delete_refuses_it(
    tmp_path, monkeypatch
):
    import threading

    path = tmp_path / "wait.sqlite3"
    init_db(path)
    conn = connect(path)
    collection_id, job_id = build_jobs.prepare_topic_build(conn, query="queued", settings=_settings(tmp_path))
    _age(conn, job_id, repo.STALE_BUILD_SECONDS + 60)
    built = []
    monkeypatch.setattr(build_jobs, "build_collection", lambda *a, **k: built.append(1))
    monkeypatch.setattr(build_jobs, "WAIT_HEARTBEAT_SECONDS", 0.05)

    assert build_jobs._BUILD_SLOT.acquire(blocking=False)
    worker = threading.Thread(
        target=build_jobs._run,
        args=(Settings(db_path=path, models_dir=tmp_path / "m"), collection_id, job_id, "queued", 10),
    )
    try:
        worker.start()
        assert _wait_for(lambda: repo.get_job(conn, job_id)["message"] == "waiting for another build to finish")
        assert repo.fail_interrupted_builds(conn) == (0, 0)
        assert repo.get_job(conn, job_id)["status"] == "queued"
        app = create_app(Settings(db_path=path, data_dir=tmp_path), warm_query_embedder=False)
        with TestClient(app) as client:
            response = client.delete(f"/api/collections/{collection_id}")
        assert response.status_code == 409
        assert repo.get_collection(conn, collection_id)["status"] == "building"
    finally:
        build_jobs._BUILD_SLOT.release()
        worker.join(5)
    assert built == [1]


def test_a_job_reaped_while_waiting_does_not_build_into_failed_rows(tmp_path, monkeypatch):
    import threading

    path = tmp_path / "reaped.sqlite3"
    init_db(path)
    conn = connect(path)
    collection_id, job_id = build_jobs.prepare_topic_build(conn, query="q", settings=_settings(tmp_path))
    built = []
    monkeypatch.setattr(build_jobs, "build_collection", lambda *a, **k: built.append(1))
    monkeypatch.setattr(build_jobs, "WAIT_HEARTBEAT_SECONDS", 0.05)

    assert build_jobs._BUILD_SLOT.acquire(blocking=False)
    worker = threading.Thread(
        target=build_jobs._run,
        args=(Settings(db_path=path, models_dir=tmp_path / "m"), collection_id, job_id, "q", 10),
    )
    try:
        worker.start()
        assert _wait_for(lambda: repo.get_job(conn, job_id)["message"] == "waiting for another build to finish")
        repo.update_job(conn, job_id, status="failed", error="reaped")
    finally:
        build_jobs._BUILD_SLOT.release()
        worker.join(5)
    assert not worker.is_alive()
    assert built == []
    assert repo.get_job(conn, job_id)["status"] == "failed"


def test_a_job_deleted_while_waiting_does_not_build(tmp_path, monkeypatch):
    import threading

    path = tmp_path / "gone.sqlite3"
    init_db(path)
    conn = connect(path)
    collection_id, job_id = build_jobs.prepare_topic_build(conn, query="q", settings=_settings(tmp_path))
    built = []
    monkeypatch.setattr(build_jobs, "build_collection", lambda *a, **k: built.append(1))
    monkeypatch.setattr(build_jobs, "WAIT_HEARTBEAT_SECONDS", 0.05)

    assert build_jobs._BUILD_SLOT.acquire(blocking=False)
    worker = threading.Thread(
        target=build_jobs._run,
        args=(Settings(db_path=path, models_dir=tmp_path / "m"), collection_id, job_id, "q", 10),
    )
    try:
        worker.start()
        assert _wait_for(lambda: repo.get_job(conn, job_id)["message"] == "waiting for another build to finish")
        repo.set_collection_status(conn, collection_id, "failed")
        repo.delete_collection(conn, collection_id)
    finally:
        build_jobs._BUILD_SLOT.release()
        worker.join(5)
    assert built == []


def test_delete_heals_a_crashed_build_instead_of_refusing_it(tmp_path):
    path = tmp_path / "heal.sqlite3"
    init_db(path)
    conn = connect(path)
    collection_id, job_id = build_jobs.prepare_topic_build(conn, query="stuck", settings=_settings(tmp_path))
    _age(conn, job_id, repo.STALE_BUILD_SECONDS + 60)
    app = create_app(Settings(db_path=path, data_dir=tmp_path), warm_query_embedder=False)
    with TestClient(app) as client:
        _age(conn, job_id, repo.STALE_BUILD_SECONDS + 60)
        conn.execute("UPDATE collections SET status='building' WHERE id=?", (collection_id,))
        conn.execute("UPDATE jobs SET status='queued' WHERE id=?", (job_id,))
        conn.commit()
        response = client.delete(f"/api/collections/{collection_id}")
    assert response.status_code == 200
    assert repo.get_collection(conn, collection_id) is None


def _api(path, tmp_path):
    app = create_app(Settings(db_path=path, data_dir=tmp_path), warm_query_embedder=False)
    app.state.build_starter = lambda *a, **k: None
    return TestClient(app)


def test_a_building_collection_reports_its_live_job_progress_and_others_report_none(tmp_path):
    path = tmp_path / "progress.sqlite3"
    init_db(path)
    conn = connect(path)
    building, job_id = build_jobs.prepare_topic_build(conn, query="live", settings=_settings(tmp_path))
    repo.update_job(conn, job_id, status="running", progress=0.45, message="ingested 45/100")
    ready, _ = build_jobs.prepare_topic_build(conn, query="old", settings=_settings(tmp_path))
    repo.set_collection_status(conn, ready, "ready")

    with _api(path, tmp_path) as client:
        listed = {c["id"]: c for c in client.get("/api/collections").json()["collections"]}
        detail = client.get(f"/api/collections/{building}").json()

    assert listed[building]["job"] == {
        "id": job_id, "status": "running", "progress": 0.45, "message": "ingested 45/100",
    }
    assert listed[ready]["job"] is None
    assert detail["job"]["message"] == "ingested 45/100"


def test_cancel_unknown_collection_is_404(tmp_path):
    path = tmp_path / "c404.sqlite3"
    init_db(path)
    with _api(path, tmp_path) as client:
        response = client.post("/api/collections/nope/cancel")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_cancel_a_collection_that_is_not_building_is_409(tmp_path):
    path = tmp_path / "c409.sqlite3"
    init_db(path)
    conn = connect(path)
    cid, _ = build_jobs.prepare_topic_build(conn, query="q", settings=_settings(tmp_path))
    repo.set_collection_status(conn, cid, "ready")
    with _api(path, tmp_path) as client:
        response = client.post(f"/api/collections/{cid}/cancel")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "not_building"
    assert repo.get_collection(conn, cid)["status"] == "ready"


def test_cancel_returns_202_fails_job_and_collection_and_delete_then_works(tmp_path):
    path = tmp_path / "c202.sqlite3"
    init_db(path)
    conn = connect(path)
    cid, job_id = build_jobs.prepare_topic_build(conn, query="q", settings=_settings(tmp_path))
    repo.update_job(conn, job_id, status="running", progress=0.2, message="ingested 2/10")

    with _api(path, tmp_path) as client:
        response = client.post(f"/api/collections/{cid}/cancel")
        job = repo.get_job(conn, job_id)
        collection = repo.get_collection(conn, cid)
        again = client.post(f"/api/collections/{cid}/cancel")
        deleted = client.delete(f"/api/collections/{cid}")

    assert response.status_code == 202
    assert response.json() == {"collection_id": cid, "status": "cancelling"}
    assert (job["status"], job["error"]) == ("failed", "cancelled")
    assert (collection["status"], collection["error"], collection["job"]) == ("failed", "Cancelled.", None)
    assert again.status_code == 409
    assert deleted.status_code == 200
    assert repo.get_collection(conn, cid) is None


def test_a_worker_that_sees_the_cancellation_releases_the_slot_and_leaves_the_state(tmp_path, monkeypatch):
    from services.collection import BuildCancelled

    path = tmp_path / "slot.sqlite3"
    init_db(path)
    conn = connect(path)
    first, first_job = build_jobs.prepare_topic_build(conn, query="one", settings=_settings(tmp_path))
    second, second_job = build_jobs.prepare_topic_build(conn, query="two", settings=_settings(tmp_path))
    ran = []

    def fake_build(build_conn, **kwargs):
        if kwargs["job_id"] == first_job:
            repo.update_job(build_conn, first_job, status="running", message="ingested 2/5")
            repo.cancel_build(build_conn, first)
            raise BuildCancelled(first_job)
        ran.append(kwargs["job_id"])
        repo.update_job(build_conn, kwargs["job_id"], status="done", progress=1.0)
        repo.set_collection_status(build_conn, kwargs["collection_id"], "ready")

    monkeypatch.setattr(build_jobs, "build_collection", fake_build)
    settings = Settings(db_path=path, models_dir=tmp_path / "m")
    build_jobs._run(settings, first, first_job, "one", 5)
    build_jobs._run(settings, second, second_job, "two", 5)

    job = repo.get_job(conn, first_job)
    assert (job["status"], job["error"]) == ("failed", "cancelled")
    assert repo.get_collection(conn, first)["error"] == "Cancelled."
    assert ran == [second_job] and repo.get_collection(conn, second)["status"] == "ready"


def test_a_queued_build_cancelled_while_waiting_never_builds(tmp_path, monkeypatch):
    import threading

    path = tmp_path / "qcancel.sqlite3"
    init_db(path)
    conn = connect(path)
    cid, job_id = build_jobs.prepare_topic_build(conn, query="q", settings=_settings(tmp_path))
    built = []
    monkeypatch.setattr(build_jobs, "build_collection", lambda *a, **k: built.append(1))
    monkeypatch.setattr(build_jobs, "WAIT_HEARTBEAT_SECONDS", 0.05)

    assert build_jobs._BUILD_SLOT.acquire(blocking=False)
    worker = threading.Thread(
        target=build_jobs._run,
        args=(Settings(db_path=path, models_dir=tmp_path / "m"), cid, job_id, "q", 10),
    )
    try:
        worker.start()
        assert _wait_for(lambda: repo.get_job(conn, job_id)["message"] == "waiting for another build to finish")
        with _api(path, tmp_path) as client:
            assert client.post(f"/api/collections/{cid}/cancel").status_code == 202
    finally:
        build_jobs._BUILD_SLOT.release()
        worker.join(5)
    assert not worker.is_alive() and built == []
    assert repo.get_job(conn, job_id)["error"] == "cancelled"
    assert repo.get_collection(conn, cid)["error"] == "Cancelled."
