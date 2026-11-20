
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from ingest.chunk import Chunk
from store.db import connect, init_db
from store.repo import (
    PaperRow,
    create_collection,
    replace_chunks,
    save_embedding,
    set_collection_status,
    upsert_paper,
)

DIM = 4


def _paper(title, arxiv_id):
    return PaperRow(
        arxiv_id=arxiv_id,
        openalex_id=None,
        doi=None,
        title=title,
        abstract="a",
        authors=(),
        year=2024,
        venue=None,
        pdf_url=None,
        citation_count=0,
        source="arxiv",
        parse_status="ok",
        parse_note=None,
        has_fulltext=1,
    )


def seed(path, n=20, *, status="ready"):
    conn = connect(path)
    cid = create_collection(conn, mode="topic", query="q", embedding_model="m")
    for i in range(n):
        pid = upsert_paper(conn, collection_id=cid, paper=_paper(f"P{i}", f"2402.000{i:02d}"))
        replace_chunks(conn, pid, [Chunk(seq=0, text=f"text {i}", n_tokens=2)])
        chunk_id = conn.execute("SELECT id FROM chunks WHERE paper_id=?", (pid,)).fetchone()[0]
        vector = [((i + 1) * (j + 1)) % 5 + 1.0 for j in range(DIM)]
        save_embedding(conn, owner_type="chunk", owner_id=chunk_id, vec=vector, model="m")
    set_collection_status(conn, cid, status)
    conn.close()
    return cid


def fake_projector(vectors, *, method, seed, n_neighbors):
    return [(float(index * 2), float(index)) for index in range(len(vectors))]


def fake_clusterer(points, *, min_cluster_size):
    return [index % 3 for index in range(len(points))]


def client(path, *, projector=fake_projector, clusterer=fake_clusterer):
    app = create_app(Settings(db_path=path), warm_query_embedder=False)
    app.state.map_projector = projector
    app.state.map_clusterer = clusterer
    return TestClient(app)


def test_a_big_enough_collection_returns_a_payload(db):
    cid = seed(db, n=20)
    with client(db) as api:
        response = api.get(f"/api/collections/{cid}/map")

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["points"]) == 20
    assert payload["method"] == "umap-hdbscan"
    assert payload["params_hash"]
    assert payload["excluded"] == 0
    assert payload["cluster_note"] is None
    point = payload["points"][0]
    assert set(point) == {"paper_id", "x", "y", "cluster"}


def test_too_few_papers_is_409_map_needs_more_papers(db):
    cid = seed(db, n=4)
    with client(db) as api:
        response = api.get(f"/api/collections/{cid}/map")

    assert response.status_code == 409
    error = response.json()["error"]
    assert error["code"] == "map_needs_more_papers"
    assert error["detail"] == {"reason": "too_few_papers", "needed": 15, "have": 4}


def test_unknown_collection_is_404(db):
    with client(db) as api:
        response = api.get("/api/collections/nope/map")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_a_collection_that_is_not_ready_is_409(db):
    cid = seed(db, n=20, status="building")
    with client(db) as api:
        response = api.get(f"/api/collections/{cid}/map")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "not_ready"


def test_the_second_request_reuses_the_cached_projection(db):
    cid = seed(db, n=20)
    calls = []

    def counting_projector(vectors, *, method, seed, n_neighbors):
        calls.append(len(vectors))
        return fake_projector(vectors, method=method, seed=seed, n_neighbors=n_neighbors)

    with client(db, projector=counting_projector) as api:
        first = api.get(f"/api/collections/{cid}/map").json()
        second = api.get(f"/api/collections/{cid}/map").json()

    assert calls == [20]
    assert first["points"] == second["points"]
    assert first["params_hash"] == second["params_hash"]


def test_an_unreliable_clustering_is_reported_in_the_payload(db):
    cid = seed(db, n=20)

    def all_noise(points, *, min_cluster_size):
        return [-1] * len(points)

    with client(db, clusterer=all_noise) as api:
        payload = api.get(f"/api/collections/{cid}/map").json()

    assert payload["cluster_note"] == "clusters_unreliable"
    assert {point["cluster"] for point in payload["points"]} == {-1}


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "map_api.sqlite3"
    init_db(path)
    return path
