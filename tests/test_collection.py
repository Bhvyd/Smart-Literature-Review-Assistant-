
import pytest

from services.collection import abstract_paper_text, build_collection
from store import repo
from store.db import connect, init_db


class FakeArxiv:
    def __init__(self, records):
        self.records = records

    def search(self, query, limit):
        return self.records[:limit]


class FakeOpenAlex:
    def __init__(self, works, titles=None, raises=None):
        self.works = works
        self.titles = titles or {}
        self.raises = raises
        self.title_calls = 0

    def lookup_by_arxiv(self, arxiv_id):
        if self.raises:
            raise self.raises
        return self.works.get(arxiv_id)

    def titles_for(self, ids, on_error=None):
        self.title_calls += 1
        return {i: self.titles[i] for i in ids if i in self.titles}


class FakeEmbedder:
    dim = 4

    def __init__(self):
        self.batches = 0

    def count_tokens(self, text):
        return len(text.split())

    def embed_texts(self, texts, batch_size=8):
        self.batches += 1
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    def close(self):
        pass


class FakeSettings:
    embedding_model = "m"
    cache_dir = "./data/cache"
    contact_email = None
    http_user_agent = "t"
    models_dir = "./models"


def record(arxiv_id="2402.01788", title="T", abstract="Abstract body " * 40, pdf_url="u"):
    return type(
        "R",
        (),
        dict(
            arxiv_id=arxiv_id,
            title=title,
            abstract=abstract,
            authors=("Ada",),
            published="2024-01-01",
            pdf_url=pdf_url,
        ),
    )()


def work(openalex_id="W1", references=(), cited=7):
    return type(
        "W",
        (),
        dict(
            openalex_id=openalex_id,
            doi="10.1/x",
            title="T",
            year=2024,
            cited_by_count=cited,
            referenced_works=tuple(references),
            venue="V",
        ),
    )()


@pytest.fixture
def conn(tmp_path):
    db = tmp_path / "t.sqlite3"
    init_db(db)
    return connect(db)


def build(conn, **kw):
    kw.setdefault("query", "q")
    kw.setdefault("limit", 10)
    kw.setdefault("settings", FakeSettings())
    kw.setdefault("openalex", FakeOpenAlex({}))
    kw.setdefault("embedder", FakeEmbedder())
    kw.setdefault("fetch_pdf", lambda client, url: None)
    kw.setdefault("sleep", lambda _: None)
    kw.setdefault("log", lambda *_: None)
    return build_collection(conn, **kw)


def test_abstract_paper_text_joins_title_and_abstract():
    assert abstract_paper_text("T", "A") == "T\nA"


def test_build_stores_papers_chunks_and_embeddings(conn):
    report = build(conn, arxiv=FakeArxiv([record()]))
    assert report.n_papers == 1
    assert report.n_chunks >= 1
    assert report.n_embeddings == report.n_chunks
    assert conn.execute("SELECT count(*) FROM embeddings").fetchone()[0] == report.n_embeddings


def test_enrichment_stores_citation_count_and_venue(conn):
    build(conn, arxiv=FakeArxiv([record()]), openalex=FakeOpenAlex({"2402.01788": work()}))
    row = conn.execute("SELECT citation_count, venue, openalex_id, doi FROM papers").fetchone()
    assert row["citation_count"] == 7
    assert row["venue"] == "V"
    assert row["openalex_id"] == "W1"
    assert row["doi"] == "10.1/x"


def test_openalex_rate_limit_degrades_but_still_finishes(conn):
    from ingest.http import RateLimited

    report = build(
        conn,
        arxiv=FakeArxiv([record(), record(arxiv_id="2402.00002")]),
        openalex=FakeOpenAlex({}, raises=RateLimited("429")),
    )
    assert "openalex_rate_limited" in report.degraded
    assert report.n_papers == 2
    row = conn.execute("SELECT citation_count FROM papers LIMIT 1").fetchone()
    assert row["citation_count"] is None


def test_rate_limit_is_not_retried_for_every_paper(conn):
    from ingest.http import RateLimited

    class CountingOpenAlex(FakeOpenAlex):
        def __init__(self):
            super().__init__({}, raises=RateLimited("429"))
            self.calls = 0

        def lookup_by_arxiv(self, arxiv_id):
            self.calls += 1
            return super().lookup_by_arxiv(arxiv_id)

    openalex = CountingOpenAlex()
    build(
        conn,
        arxiv=FakeArxiv([record(arxiv_id=f"2402.0000{i}") for i in range(5)]),
        openalex=openalex,
    )
    assert openalex.calls == 1


def test_an_empty_search_fails_with_the_reason_instead_of_a_ready_empty_collection(conn):
    with pytest.raises(Exception):
        build(conn, query="artificial interlligence", arxiv=FakeArxiv([]))
    row = conn.execute("SELECT status, error, n_papers FROM collections").fetchone()
    assert row["status"] == "failed"
    assert "No arXiv papers match" in row["error"] and "interlligence" in row["error"]
    job = conn.execute("SELECT status, error FROM jobs").fetchone()
    assert job["status"] == "failed"


def test_a_paper_with_no_pdf_is_still_retrievable_from_its_abstract(conn):
    build(conn, arxiv=FakeArxiv([record()]))
    row = conn.execute("SELECT parse_status, has_fulltext FROM papers").fetchone()
    assert row["parse_status"] in {"failed", "not_attempted"}
    assert row["has_fulltext"] == 0
    assert conn.execute("SELECT count(*) FROM chunks").fetchone()[0] >= 1


def test_garbled_pdf_is_marked_low_quality_and_not_embedded_from_its_text(conn):
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((40, 60), "@@@@ #### $$$$ %%%% " * 20, fontsize=9)
    soup = doc.tobytes()

    build(conn, arxiv=FakeArxiv([record()]), fetch_pdf=lambda client, url: soup)
    row = conn.execute("SELECT parse_status, has_fulltext FROM papers").fetchone()
    assert row["parse_status"] == "low_quality"
    assert row["has_fulltext"] == 0

    chunk = conn.execute("SELECT text FROM chunks LIMIT 1").fetchone()["text"]
    assert "Abstract body" in chunk
    assert "@@@@" not in chunk


def test_a_parsed_pdf_is_embedded_from_its_full_text(conn):
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    y = 60
    for _ in range(10):
        page.insert_text((40, y), "Full text body sentence about retrieval. " * 3, fontsize=9)
        y += 20

    build(conn, arxiv=FakeArxiv([record(abstract="short")]), fetch_pdf=lambda c, u: doc.tobytes())
    row = conn.execute("SELECT parse_status, has_fulltext FROM papers").fetchone()
    assert row["parse_status"] == "ok"
    assert row["has_fulltext"] == 1
    chunk = conn.execute("SELECT text FROM chunks LIMIT 1").fetchone()["text"]
    assert "Full text body" in chunk


def test_rebuild_is_idempotent(conn):
    kwargs = dict(arxiv=FakeArxiv([record()]))
    first = build(conn, **kwargs)
    second = build(conn, **kwargs)
    assert first.collection_id == second.collection_id
    assert first.n_papers == second.n_papers == 1
    assert conn.execute("SELECT count(*) FROM papers").fetchone()[0] == 1
    assert conn.execute("SELECT count(*) FROM chunks").fetchone()[0] == second.n_chunks
    assert conn.execute("SELECT count(*) FROM embeddings").fetchone()[0] == second.n_embeddings


def test_reuse_false_builds_a_separate_collection(conn):
    first = build(conn, arxiv=FakeArxiv([record()]))
    second = build(conn, arxiv=FakeArxiv([record()]), reuse=False)
    assert first.collection_id != second.collection_id
    assert conn.execute("SELECT count(*) FROM collections").fetchone()[0] == 2


def test_a_different_query_does_not_reuse_the_collection(conn):
    a = build(conn, query="attention", arxiv=FakeArxiv([record()]))
    b = build(conn, query="retrieval", arxiv=FakeArxiv([record()]))
    assert a.collection_id != b.collection_id


def test_no_stored_chunk_exceeds_the_model_limit(conn):
    class DenseEmbedder(FakeEmbedder):
        def count_tokens(self, text):
            return sum(7 if word == "dense" else 1 for word in text.split())

    body = " ".join(["filler"] * 50 + ["dense"] * 20) * 5
    build(conn, arxiv=FakeArxiv([record(abstract=body)]), embedder=DenseEmbedder())
    worst = conn.execute("SELECT max(n_tokens) FROM chunks").fetchone()[0]
    assert worst <= 512


def test_an_embedder_that_drops_inputs_fails_the_build(conn):
    from services.collection import CollectionBuildError

    class DroppingEmbedder(FakeEmbedder):
        def embed_texts(self, texts, batch_size=8):
            return []

    with pytest.raises(CollectionBuildError):
        build(conn, arxiv=FakeArxiv([record()]), embedder=DroppingEmbedder())


class CountingClient:
    def __init__(self):
        self.network_requests = 0

    def close(self):
        pass


def test_pdf_fetches_are_spaced_out(conn):
    from services.collection import PDF_FETCH_INTERVAL

    client = CountingClient()

    def network_fetch(c, url):
        c.network_requests += 1
        return None

    pauses = []
    build(
        conn,
        arxiv=FakeArxiv([record(arxiv_id=f"2402.0000{i}") for i in range(3)]),
        sleep=pauses.append,
        client=client,
        fetch_pdf=network_fetch,
    )
    assert pauses == [PDF_FETCH_INTERVAL, PDF_FETCH_INTERVAL]


def test_a_rebuild_served_from_cache_does_not_wait(conn):
    pauses = []
    build(
        conn,
        arxiv=FakeArxiv([record(arxiv_id=f"2402.0000{i}") for i in range(3)]),
        sleep=pauses.append,
        client=CountingClient(),
    )
    assert pauses == []


def test_build_degradation_is_persisted_on_the_collection(conn):
    from ingest.http import RateLimited
    from store import repo

    report = build(
        conn,
        arxiv=FakeArxiv([record()]),
        openalex=FakeOpenAlex({}, raises=RateLimited("limited", None)),
    )
    assert "openalex_rate_limited" in report.degraded
    stored = repo.get_collection(conn, report.collection_id)
    assert stored["degraded"] == ["openalex_rate_limited"]


def test_a_clean_build_stores_an_empty_degraded_list(conn):
    from store import repo

    report = build(conn, arxiv=FakeArxiv([record()]))
    assert repo.get_collection(conn, report.collection_id)["degraded"] == []


def test_each_paper_gets_a_title_abstract_vector(conn):
    class PaperEmbedder(FakeEmbedder):
        def paper_text(self, title, abstract):
            return f"{title}[SEP]{abstract}"

    report = build(conn, arxiv=FakeArxiv([record()]), embedder=PaperEmbedder())
    rows = conn.execute(
        "SELECT owner_type, count(*) AS n FROM embeddings GROUP BY owner_type"
    ).fetchall()
    counts = {row["owner_type"]: row["n"] for row in rows}
    assert counts["paper"] == 1
    assert counts["chunk"] == report.n_embeddings


def test_edge_identity_does_not_depend_on_whether_the_title_was_fetched(conn):
    from store import repo

    paper = build(conn, arxiv=FakeArxiv([record()])).collection_id
    pid = conn.execute("SELECT id FROM papers").fetchone()["id"]
    for title in (None, "A real title"):
        repo.upsert_edge(
            conn, collection_id=paper, src_paper_id=pid, dst_paper_id=None,
            dst_openalex_id="W9", dst_title=title, dst_doi=None,
        )
    rows = conn.execute("SELECT dst_title FROM edges WHERE dst_openalex_id='W9'").fetchall()
    assert [row["dst_title"] for row in rows] == ["A real title"]


def test_embedding_provenance_comes_from_the_embedder(conn):
    class NamedEmbedder(FakeEmbedder):
        model_name = "allenai/specter2_base+allenai/specter2"

    build(conn, arxiv=FakeArxiv([record()]), embedder=NamedEmbedder())
    row = conn.execute("SELECT model FROM embeddings LIMIT 1").fetchone()
    assert row["model"] == "allenai/specter2_base+allenai/specter2"


def test_references_inside_the_collection_resolve_to_papers(conn):
    a = record(arxiv_id="2402.00001")
    b = record(arxiv_id="2402.00002")
    works = {
        "2402.00001": work(openalex_id="W1", references=("W2",)),
        "2402.00002": work(openalex_id="W2"),
    }
    report = build(conn, arxiv=FakeArxiv([a, b]), openalex=FakeOpenAlex(works))
    assert report.n_edges == 1
    row = conn.execute("SELECT resolved, dst_paper_id FROM edges").fetchone()
    assert row["resolved"] == 1
    assert row["dst_paper_id"] is not None


def test_external_references_store_the_real_title(conn):
    works = {"2402.01788": work(openalex_id="W1", references=("W9",))}
    openalex = FakeOpenAlex(works, titles={"W9": "Some Other Paper"})
    report = build(conn, arxiv=FakeArxiv([record()]), openalex=openalex)
    assert report.n_edges == 1
    assert openalex.title_calls == 1
    row = conn.execute("SELECT resolved, dst_title, dst_openalex_id FROM edges").fetchone()
    assert row["resolved"] == 0
    assert row["dst_title"] == "Some Other Paper"
    assert row["dst_openalex_id"] == "W9"


def test_job_is_marked_done(conn):
    build(conn, arxiv=FakeArxiv([record()]))
    row = conn.execute("SELECT status, progress FROM jobs").fetchone()
    assert row["status"] == "done"
    assert row["progress"] == 1.0


def test_a_hard_failure_marks_collection_and_job_failed(conn):
    from services.collection import CollectionBuildError

    class ExplodingArxiv:
        def search(self, query, limit):
            raise RuntimeError("arXiv unreachable after 3 attempts")

    with pytest.raises(CollectionBuildError):
        build(conn, arxiv=ExplodingArxiv())

    row = conn.execute("SELECT status, error FROM collections").fetchone()
    assert row["status"] == "failed"
    assert "arXiv unreachable" in row["error"]
    job = conn.execute("SELECT status FROM jobs").fetchone()
    assert job["status"] == "failed"


def test_collection_counts_match_the_stored_rows(conn):
    build(conn, arxiv=FakeArxiv([record()]))
    row = conn.execute("SELECT n_papers, n_chunks FROM collections").fetchone()
    assert row["n_papers"] == 1
    assert row["n_chunks"] == conn.execute("SELECT count(*) FROM chunks").fetchone()[0]


def _pdf_with_title_page():
    import pymupdf

    document = pymupdf.open()
    first = document.new_page()
    head = ["A Study of Chunking", "Valentin Kreileder", "Deggendorf Institute of Technology", "kreileder@gmx.net", "Abstract"]
    for n, line in enumerate(head):
        first.insert_text((40, 60 + 12 * n), line, fontsize=9)
    sentence = "Chunk boundaries change which passages a retriever can find. "
    for n in range(60):
        first.insert_text((40, 130 + 11 * n), sentence, fontsize=8)
    return document.tobytes()


def test_a_rebuild_replaces_stored_title_page_chunks_with_front_matter_free_ones(conn):
    from store.repo import replace_chunks
    from ingest.chunk import Chunk

    arxiv = FakeArxiv([record()])
    pdf = _pdf_with_title_page()
    first = build(conn, arxiv=arxiv, fetch_pdf=lambda client, url: pdf)
    paper_id = conn.execute("SELECT id FROM papers").fetchone()["id"]
    replace_chunks(conn, paper_id, [Chunk(seq=0, text="Valentin Kreileder kreileder@gmx.net Abstract", n_tokens=6)])

    second = build(conn, arxiv=arxiv, fetch_pdf=lambda client, url: pdf, collection_id=first.collection_id)

    texts = [row["text"] for row in conn.execute("SELECT text FROM chunks WHERE paper_id=?", (paper_id,))]
    assert texts and not any("@" in t or "Deggendorf" in t for t in texts)
    assert any("Chunk boundaries" in t for t in texts)
    assert second.collection_id == first.collection_id


class CancellingEmbedder(FakeEmbedder):
    def __init__(self, db_path, collection_id, after):
        super().__init__()
        self.db_path, self.collection_id, self.after, self.papers = db_path, collection_id, after, 0

    def paper_text(self, title, abstract):
        return f"PAPER::{title}"

    def embed_texts(self, texts, batch_size=8):
        if len(texts) == 1 and texts[0].startswith("PAPER::"):
            self.papers += 1
            if self.papers == self.after:
                other = connect(self.db_path)
                try:
                    assert repo.cancel_build(other, self.collection_id) == 1
                finally:
                    other.close()
        return super().embed_texts(texts, batch_size)


def test_a_cancelled_build_stops_after_the_current_paper_and_keeps_the_cancelled_state(tmp_path):
    from services.collection import BuildCancelled

    db = tmp_path / "cancel.sqlite3"
    init_db(db)
    conn = connect(db)
    cid = repo.create_collection(conn, mode="topic", query="q", embedding_model="m")
    job = repo.create_job(conn, collection_id=cid, kind="build")
    arxiv = FakeArxiv([record(arxiv_id=f"2402.0000{i}", title=f"P{i}") for i in range(1, 6)])

    with pytest.raises(BuildCancelled):
        build(
            conn, arxiv=arxiv, collection_id=cid, job_id=job,
            embedder=CancellingEmbedder(db, cid, after=2),
        )

    titles = [r["title"] for r in conn.execute("SELECT title FROM papers ORDER BY title")]
    assert titles == ["P1", "P2"]
    assert repo.get_job(conn, job)["status"] == "failed"
    assert repo.get_job(conn, job)["error"] == "cancelled"
    collection = repo.get_collection(conn, cid)
    assert (collection["status"], collection["error"]) == ("failed", "Cancelled.")


def test_a_build_cancelled_before_it_starts_writes_nothing(conn):
    from services.collection import BuildCancelled

    cid = repo.create_collection(conn, mode="topic", query="q", embedding_model="m")
    job = repo.create_job(conn, collection_id=cid, kind="build")
    repo.cancel_build(conn, cid)

    with pytest.raises(BuildCancelled):
        build(conn, arxiv=FakeArxiv([record()]), collection_id=cid, job_id=job)

    assert conn.execute("SELECT count(*) FROM papers").fetchone()[0] == 0
    assert repo.get_collection(conn, cid)["status"] == "failed"
