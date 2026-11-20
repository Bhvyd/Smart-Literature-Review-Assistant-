
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


class FakeQueryEmbedder:
    def __init__(self, vector=None):
        self._vector = vector
        self.calls = 0

    @property
    def available(self):
        return self._vector is not None

    def vector(self, text):
        self.calls += 1
        return self._vector

    def warm(self):
        return self._vector is not None


def _paper(title, arxiv_id, *, citations=0, year=2024):
    return PaperRow(
        arxiv_id=arxiv_id,
        openalex_id=None,
        doi=None,
        title=title,
        abstract=f"abstract for {title}",
        authors=("Ada",),
        year=year,
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
    path = tmp_path / "api.sqlite3"
    init_db(path)
    return path


def seed(path, papers, *, vector=(1.0, 0.0), status="ready"):
    conn = connect(path)
    cid = create_collection(conn, mode="topic", query="q", embedding_model="m")
    for title, arxiv_id, citations, year, texts in papers:
        pid = upsert_paper(
            conn,
            collection_id=cid,
            paper=_paper(title, arxiv_id, citations=citations, year=year),
        )
        chunks = [Chunk(seq=i, text=text, n_tokens=len(text.split())) for i, text in enumerate(texts)]
        replace_chunks(conn, pid, chunks)
        for row in conn.execute("SELECT id FROM chunks WHERE paper_id=? ORDER BY seq", (pid,)):
            save_embedding(conn, owner_type="chunk", owner_id=row[0], vec=list(vector), model="m")
    set_collection_status(conn, cid, status)
    conn.close()
    return cid


def client(path, *, query_embedder=None):
    app = create_app(Settings(db_path=path), warm_query_embedder=False)
    app.state.query_embedder = query_embedder or FakeQueryEmbedder()
    return TestClient(app)


def test_unknown_collection_is_404_not_found(db):
    with client(db) as api:
        response = api.get("/api/collections/nope/papers")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_collection_still_building_is_409_not_ready(db):
    with client(db) as api:
        cid = seed(db, [("A", "2402.00001", 0, 2024, ["text"])], status="building")
        response = api.get(f"/api/collections/{cid}/papers")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "not_ready"
    assert response.json()["error"]["detail"]["status"] == "building"


def test_a_query_ranks_the_matching_paper_first(db):
    cid = seed(
        db,
        [
            ("Attention", "2402.00001", 0, 2024, ["self attention transformer"]),
            ("Cooking", "2402.00002", 0, 2024, ["sourdough starter recipe"]),
        ],
    )
    with client(db) as api:
        payload = api.get(f"/api/collections/{cid}/papers", params={"q": "attention"}).json()

    assert payload["mode"] == "lexical"
    assert payload["degraded"] == []
    assert payload["papers"][0]["title"] == "Attention"
    assert payload["papers"][0]["score"] is not None


def test_no_query_orders_by_influence(db):
    cid = seed(
        db,
        [
            ("Low", "2402.00001", 1, 2024, ["x"]),
            ("High", "2402.00002", 99, 2024, ["x"]),
            ("Unknown", "2402.00003", None, 2024, ["x"]),
        ],
    )
    with client(db) as api:
        payload = api.get(f"/api/collections/{cid}/papers").json()

    assert payload["mode"] == "influence"
    assert [paper["title"] for paper in payload["papers"]] == ["High", "Low", "Unknown"]


def test_sort_year_puts_the_newest_first_and_unknown_years_last(db):
    cid = seed(
        db,
        [
            ("Old", "2402.00001", 0, 2019, ["x"]),
            ("New", "2402.00002", 0, 2025, ["x"]),
            ("Undated", "2402.00003", 0, None, ["x"]),
        ],
    )
    with client(db) as api:
        payload = api.get(f"/api/collections/{cid}/papers", params={"sort": "year"}).json()

    assert [paper["title"] for paper in payload["papers"]] == ["New", "Old", "Undated"]


def test_sort_influence_ignores_a_query(db):
    cid = seed(
        db,
        [
            ("Relevant", "2402.00001", 0, 2024, ["attention text"]),
            ("Cited", "2402.00002", 50, 2024, ["unrelated"]),
        ],
    )
    with client(db) as api:
        payload = api.get(
            f"/api/collections/{cid}/papers",
            params={"q": "attention", "sort": "influence"},
        ).json()

    assert payload["mode"] == "influence"
    assert payload["papers"][0]["title"] == "Cited"


def test_limit_is_applied(db):
    cid = seed(db, [(f"P{i}", f"2402.0000{i}", i, 2024, ["shared keyword"]) for i in range(5)])
    with client(db) as api:
        payload = api.get(
            f"/api/collections/{cid}/papers", params={"q": "shared keyword", "limit": 2}
        ).json()
    assert len(payload["papers"]) == 2
    assert payload["total"] == 5
    assert payload["offset"] == 0
    assert payload["limit"] == 2


def test_a_page_carries_the_metadata_the_ui_renders(db):
    cid = seed(db, [("Attention", "2402.00001", 7, 2024, ["self attention transformer"])])
    with client(db) as api:
        payload = api.get(f"/api/collections/{cid}/papers", params={"q": "attention"}).json()

    paper = payload["papers"][0]
    assert paper["abstract"] == "abstract for Attention"
    assert paper["authors"] == ["Ada"]
    assert paper["venue"] is None
    assert paper["pdf_url"] is None
    assert paper["parse_status"] == "ok"
    assert paper["has_fulltext"] == 1
    assert paper["arxiv_id"] == "2402.00001"


def test_offset_skips_papers_in_the_same_order(db):
    cid = seed(db, [(f"P{i}", f"2402.0000{i}", i, 2024, ["shared keyword"]) for i in range(5)])
    with client(db) as api:
        first = api.get(
            f"/api/collections/{cid}/papers",
            params={"q": "shared keyword", "limit": 2},
        ).json()
        second = api.get(
            f"/api/collections/{cid}/papers",
            params={"q": "shared keyword", "limit": 2, "offset": 2},
        ).json()

    assert [p["paper_id"] for p in first["papers"]] != [
        p["paper_id"] for p in second["papers"]
    ]
    assert second["offset"] == 2
    assert second["total"] == 5


def test_an_unknown_paper_is_404(db):
    with client(db) as api:
        response = api.get("/api/papers/nope")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_paper_detail_returns_the_stored_row(db):
    cid = seed(db, [("Attention", "2402.00001", 7, 2024, ["self attention transformer"])])
    with client(db) as api:
        listed = api.get(f"/api/collections/{cid}/papers").json()["papers"][0]
        response = api.get(f"/api/papers/{listed['paper_id']}")

    assert response.status_code == 200
    paper = response.json()
    assert paper["title"] == "Attention"
    assert paper["abstract"] == "abstract for Attention"
    assert paper["authors"] == ["Ada"]
    assert paper["citation_count"] == 7
    assert paper["chunk_count"] == 1
    assert paper["collection_id"] == cid


def test_an_unknown_sort_is_a_422(db):
    cid = seed(db, [("A", "2402.00001", 0, 2024, ["x"])])
    with client(db) as api:
        response = api.get(f"/api/collections/{cid}/papers", params={"sort": "bogus"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"


def test_hybrid_without_an_embedder_degrades_to_lexical(db):
    cid = seed(db, [("Attention", "2402.00001", 0, 2024, ["attention text"])])
    with client(db, query_embedder=FakeQueryEmbedder(vector=None)) as api:
        payload = api.get(
            f"/api/collections/{cid}/papers", params={"q": "attention", "mode": "hybrid"}
        ).json()

    assert payload["mode"] == "lexical"
    assert "query_vector_missing" in payload["degraded"]
    assert payload["papers"]


def test_hybrid_with_an_embedder_uses_the_dense_path(db):
    cid = seed(
        db,
        [
            ("Near", "2402.00001", 0, 2024, ["x"]),
            ("Far", "2402.00002", 0, 2024, ["y"]),
        ],
        vector=(1.0, 0.0),
    )
    embedder = FakeQueryEmbedder(vector=[1.0, 0.0])
    with client(db, query_embedder=embedder) as api:
        payload = api.get(
            f"/api/collections/{cid}/papers", params={"q": "anything", "mode": "hybrid"}
        ).json()

    assert payload["mode"] == "hybrid"
    assert payload["degraded"] == []
    assert embedder.calls == 1


def test_an_unknown_mode_is_a_422(db):
    cid = seed(db, [("A", "2402.00001", 0, 2024, ["x"])])
    with client(db) as api:
        response = api.get(f"/api/collections/{cid}/papers", params={"mode": "telepathic"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"


def test_health_still_answers_with_a_database_configured(db):
    with client(db) as api:
        assert api.get("/api/health").json()["status"] == "ok"
