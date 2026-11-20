
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from store.db import connect, init_db
from store.repo import (
    PaperRow,
    create_collection,
    set_collection_status,
    upsert_edge,
    upsert_paper,
)


def _paper(title, arxiv_id, citations):
    return PaperRow(
        arxiv_id=arxiv_id,
        openalex_id=f"W{arxiv_id[-3:]}",
        doi=None,
        title=title,
        abstract="a",
        authors=(),
        year=2024,
        venue=None,
        pdf_url=None,
        citation_count=citations,
        source="arxiv",
        parse_status="ok",
        parse_note=None,
        has_fulltext=1,
    )


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "graph_api.sqlite3"
    init_db(path)
    return path


def seed(path, n=7, edges=6, *, status="ready", external=0):
    conn = connect(path)
    cid = create_collection(conn, mode="topic", query="q", embedding_model="m")
    ids = {}
    for i in range(n):
        ids[i] = upsert_paper(
            conn, collection_id=cid, paper=_paper(f"P{i}", f"2402.00{i}", i * 10)
        )
    for i in range(edges):
        upsert_edge(
            conn,
            collection_id=cid,
            src_paper_id=ids[i],
            dst_paper_id=ids[(i + 1) % n],
            dst_openalex_id=f"W00{(i + 1) % n}",
            dst_title=f"P{(i + 1) % n}",
            dst_doi=None,
        )
    for i in range(external):
        upsert_edge(
            conn,
            collection_id=cid,
            src_paper_id=ids[0],
            dst_paper_id=None,
            dst_openalex_id=f"W9{i:02d}",
            dst_title=f"External {i}",
            dst_doi=None,
        )
    set_collection_status(conn, cid, status)
    conn.close()
    return cid, ids


def client(path):
    return TestClient(create_app(Settings(db_path=path), warm_query_embedder=False))


def test_a_collection_with_enough_edges_returns_the_graph(db):
    cid, ids = seed(db)
    with client(db) as api:
        response = api.get(f"/api/collections/{cid}/graph")

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["nodes"]) == 7
    assert len(payload["edges"]) == 6
    assert "reason" not in payload
    edge = payload["edges"][0]
    assert set(edge) == {"source", "target", "resolved"}
    assert edge["source"] in ids.values()
    assert edge["target"] in ids.values()


def test_rankings_are_returned_and_ordered(db):
    cid, ids = seed(db)
    with client(db) as api:
        payload = api.get(f"/api/collections/{cid}/graph").json()

    rankings = payload["rankings"]
    assert len(rankings) == 7
    assert rankings[0]["score"] >= rankings[-1]["score"]
    assert all({"paper_id", "score", "complete"} <= set(row) for row in rankings)


def test_too_few_edges_is_200_with_a_reason_and_no_nodes(db):
    cid, _ = seed(db, edges=2)
    with client(db) as api:
        response = api.get(f"/api/collections/{cid}/graph")

    assert response.status_code == 200
    payload = response.json()
    assert payload["reason"] == "not_enough_edges"
    assert payload["needed"] == 5
    assert payload["have"] == 2
    assert payload["nodes"] == []
    assert payload["edges"] == []


def test_external_references_are_opt_in(db):
    cid, _ = seed(db, external=2)
    with client(db) as api:
        without = api.get(f"/api/collections/{cid}/graph").json()
        with_external = api.get(
            f"/api/collections/{cid}/graph", params={"include_external": "true"}
        ).json()

    assert not [node for node in without["nodes"] if not node["resolved"]]
    externals = [node for node in with_external["nodes"] if not node["resolved"]]
    assert len(externals) == 2
    assert all(node["title"].startswith("External") for node in externals)


def test_unknown_collection_is_404(db):
    with client(db) as api:
        assert api.get("/api/collections/nope/graph").status_code == 404


def test_a_collection_that_is_not_ready_is_409(db):
    cid, _ = seed(db, status="failed")
    with client(db) as api:
        response = api.get(f"/api/collections/{cid}/graph")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "not_ready"
