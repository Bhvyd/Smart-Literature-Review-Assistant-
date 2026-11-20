
import hashlib
import io

import pymupdf
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from ingest.http import RateLimited
from ingest.pdf import extract_metadata
from services.upload import (
    MAX_FILES,
    UploadRejected,
    build_upload_collection,
    upload_path,
    validate_and_store,
)
from store import repo
from store.db import connect, init_db

ABSTRACT = (
    "We study how retrieval grounds language model answers in source documents and measure "
    "how much it reduces unsupported claims across several question answering benchmarks, "
    "showing that evidence selection matters more than model size for factual accuracy."
)
BODY = ("Retrieval augmented generation combines a retriever with a generator. " * 30).strip()


def make_pdf(
    title="Grounded Answers From Retrieved Evidence",
    *,
    stamp=None,
    doi=None,
    abstract=ABSTRACT,
    body=BODY,
    meta_title=None,
    pages=2,
):
    doc = pymupdf.open()
    page = doc.new_page()
    y = 60
    page.insert_text((40, y), title, fontsize=20)
    y += 40
    if stamp:
        page.insert_text((40, 20), f"arXiv:{stamp}v2 [cs.CL] 14 Jun 2025", fontsize=8)
    if doi:
        page.insert_text((40, y), f"https://doi.org/{doi}", fontsize=9)
        y += 20
    if abstract:
        page.insert_text((40, y), "Abstract", fontsize=11)
        y += 16
        for start in range(0, len(abstract), 80):
            page.insert_text((40, y), abstract[start : start + 80], fontsize=9)
            y += 12
        page.insert_text((40, y + 10), "1 Introduction", fontsize=11)
    for _ in range(pages - 1):
        extra = doc.new_page()
        yy = 60
        for start in range(0, len(body), 80):
            extra.insert_text((40, yy), body[start : start + 80], fontsize=9)
            yy += 12
            if yy > 760:
                break
    if meta_title is not None:
        doc.set_metadata({"title": meta_title})
    return doc.tobytes()


def test_the_title_is_the_text_set_in_the_largest_font():
    assert extract_metadata(make_pdf()).title == "Grounded Answers From Retrieved Evidence"


def test_embedded_pdf_metadata_wins_over_layout_when_it_is_a_real_title():
    meta = extract_metadata(make_pdf(meta_title="A Properly Embedded Title"))
    assert meta.title == "A Properly Embedded Title"


def test_a_placeholder_metadata_title_is_ignored():
    meta = extract_metadata(make_pdf(meta_title="Microsoft Word"))
    assert meta.title == "Grounded Answers From Retrieved Evidence"


def test_the_abstract_is_taken_up_to_the_introduction():
    meta = extract_metadata(make_pdf())
    assert meta.abstract.startswith("We study how retrieval")
    assert "Introduction" not in meta.abstract


def test_an_arxiv_stamp_and_a_doi_are_read_from_the_first_pages():
    meta = extract_metadata(make_pdf(stamp="2506.06962", doi="10.1145/3511808.3557123"))
    assert meta.arxiv_id == "2506.06962"
    assert meta.doi == "10.1145/3511808.3557123"


def test_missing_fields_stay_missing_instead_of_being_invented():
    meta = extract_metadata(make_pdf(abstract=None))
    assert meta.abstract is None and meta.doi is None and meta.arxiv_id is None
    assert meta.authors == ()


def test_unreadable_bytes_give_empty_metadata_not_an_exception():
    meta = extract_metadata(b"not a pdf at all")
    assert (meta.title, meta.abstract, meta.doi, meta.arxiv_id) == (None, None, None, None)


def test_files_are_stored_by_content_hash(tmp_path):
    data = make_pdf()
    stored = validate_and_store(tmp_path, "c1", [("my paper.pdf", data)])
    sha = hashlib.sha256(data).hexdigest()
    assert stored[0].sha256 == sha
    assert (tmp_path / "uploads" / "c1" / f"{sha}.pdf").read_bytes() == data


def test_the_same_file_twice_is_stored_once(tmp_path):
    data = make_pdf()
    assert len(validate_and_store(tmp_path, "c1", [("a.pdf", data), ("b.pdf", data)])) == 1


def test_a_non_pdf_is_rejected_by_name(tmp_path):
    with pytest.raises(UploadRejected, match="notes.txt is not a PDF"):
        validate_and_store(tmp_path, "c1", [("notes.txt", b"hello")])


def test_empty_and_oversized_uploads_are_rejected(tmp_path):
    with pytest.raises(UploadRejected):
        validate_and_store(tmp_path, "c1", [])
    with pytest.raises(UploadRejected, match="at most"):
        validate_and_store(tmp_path, "c1", [("a.pdf", make_pdf())] * (MAX_FILES + 1))


def test_a_path_that_is_not_a_sha256_never_resolves(tmp_path):
    assert upload_path(tmp_path, "c1", "../../etc/passwd") is None
    assert upload_path(tmp_path, "c1", "a" * 63) is None
    assert upload_path(tmp_path, "c1", "a" * 64) is not None


class FakeWork:
    def __init__(self, openalex_id="W1", title="Grounded Answers From Retrieved Evidence",
                 refs=("W7", "W8"), year=2025, cites=5, doi="10.1/x"):
        self.openalex_id, self.title, self.referenced_works = openalex_id, title, refs
        self.year, self.cited_by_count, self.doi, self.venue = year, cites, doi, "ACL"


class FakeOpenAlex:
    def __init__(self, by_arxiv=None, by_doi=None, by_title=None, raises=None):
        self.by_arxiv, self.by_doi, self.by_title, self.raises = by_arxiv or {}, by_doi or {}, by_title or {}, raises
        self.calls = []

    def _maybe_raise(self):
        if self.raises:
            raise self.raises

    def lookup_by_arxiv(self, arxiv_id):
        self.calls.append(("arxiv", arxiv_id)); self._maybe_raise()
        return self.by_arxiv.get(arxiv_id)

    def lookup_by_doi(self, doi):
        self.calls.append(("doi", doi)); self._maybe_raise()
        return self.by_doi.get(doi)

    def lookup_by_title(self, title):
        self.calls.append(("title", title)); self._maybe_raise()
        return self.by_title.get(title)

    def titles_for(self, ids, on_error=None):
        return {i: f"Title of {i}" for i in ids}


class FakeEmbedder:
    model_name = "fake+adapter"

    def count_tokens(self, text):
        return len(text.split())

    def embed_texts(self, texts, batch_size=8):
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    def paper_text(self, title, abstract):
        return f"{title}[SEP]{abstract}"

    def close(self):
        pass


class Settings_:
    cache_dir = "./data/cache"
    contact_email = None
    embedding_model = "m"

    def __init__(self, data_dir):
        self.data_dir = data_dir


@pytest.fixture
def conn(tmp_path):
    db = tmp_path / "u.sqlite3"
    init_db(db)
    return connect(db)


def build(conn, tmp_path, pdfs, openalex=None):
    cid = repo.create_collection(conn, mode="upload", query="Uploaded", embedding_model="m")
    job = repo.create_job(conn, collection_id=cid, kind="build")
    uploads = validate_and_store(tmp_path, cid, pdfs)
    degraded = build_upload_collection(
        conn, collection_id=cid, job_id=job, uploads=uploads, settings=Settings_(tmp_path),
        openalex=openalex or FakeOpenAlex(), embedder=FakeEmbedder(), client=object(),
        log=lambda *_: None,
    )
    return cid, degraded


def test_an_uploaded_paper_is_parsed_chunked_embedded_and_marked_as_an_upload(conn, tmp_path):
    cid, degraded = build(conn, tmp_path, [("p.pdf", make_pdf())])
    paper = conn.execute("SELECT * FROM papers WHERE collection_id=?", (cid,)).fetchone()
    assert paper["source"] == "upload"
    assert paper["title"] == "Grounded Answers From Retrieved Evidence"
    assert paper["parse_status"] == "ok" and paper["has_fulltext"] == 1
    assert paper["pdf_url"].startswith("upload://")
    assert conn.execute("SELECT count(*) FROM chunks").fetchone()[0] >= 1
    kinds = {r["owner_type"] for r in conn.execute("SELECT owner_type FROM embeddings")}
    assert kinds == {"paper", "chunk"}
    assert repo.get_collection(conn, cid)["status"] == "ready"
    assert degraded == []


def test_a_paper_is_enriched_from_openalex_by_its_arxiv_stamp(conn, tmp_path):
    openalex = FakeOpenAlex(by_arxiv={"2506.06962": FakeWork()})
    cid, _ = build(conn, tmp_path, [("p.pdf", make_pdf(stamp="2506.06962"))], openalex)
    paper = conn.execute("SELECT * FROM papers").fetchone()
    assert (paper["arxiv_id"], paper["openalex_id"], paper["citation_count"], paper["year"]) == (
        "2506.06962", "W1", 5, 2025,
    )
    assert conn.execute("SELECT count(*) FROM edges WHERE collection_id=?", (cid,)).fetchone()[0] == 2


def test_enrichment_tries_identifiers_before_the_title(conn, tmp_path):
    openalex = FakeOpenAlex(by_doi={"10.1145/3511808.3557123": FakeWork(openalex_id="W2")})
    build(conn, tmp_path, [("p.pdf", make_pdf(doi="10.1145/3511808.3557123"))], openalex)
    assert openalex.calls == [("doi", "10.1145/3511808.3557123")]
    assert conn.execute("SELECT openalex_id FROM papers").fetchone()[0] == "W2"


def test_an_unmatched_paper_gets_no_borrowed_metadata(conn, tmp_path):
    build(conn, tmp_path, [("p.pdf", make_pdf())], FakeOpenAlex())
    paper = conn.execute("SELECT * FROM papers").fetchone()
    assert paper["openalex_id"] is None and paper["citation_count"] is None and paper["year"] is None
    assert conn.execute("SELECT count(*) FROM edges").fetchone()[0] == 0


def test_a_pdf_without_a_readable_title_is_shown_by_file_name_with_a_note(conn, tmp_path):
    doc = pymupdf.open()
    doc.new_page().insert_text((40, 60), "tiny", fontsize=9)
    build(conn, tmp_path, [("my_great-paper.pdf", doc.tobytes())])
    paper = conn.execute("SELECT * FROM papers").fetchone()
    assert paper["title"] == "my great paper"
    assert "file name" in paper["parse_note"]


def test_an_unparseable_pdf_is_kept_with_its_reason_and_has_no_vectors(conn, tmp_path):
    broken = b"%PDF-1.4\nthis is not a real pdf body"
    cid, _ = build(conn, tmp_path, [("broken.pdf", broken)])
    paper = conn.execute("SELECT * FROM papers WHERE collection_id=?", (cid,)).fetchone()
    assert paper["parse_status"] == "failed" and "could not be parsed" in paper["parse_note"]
    assert conn.execute("SELECT count(*) FROM chunks").fetchone()[0] == 0
    assert repo.get_collection(conn, cid)["status"] == "ready"


def test_an_openalex_rate_limit_degrades_but_still_finishes_and_is_persisted(conn, tmp_path):
    openalex = FakeOpenAlex(raises=RateLimited("limited", None))
    cid, degraded = build(conn, tmp_path, [("p.pdf", make_pdf(stamp="2506.06962"))], openalex)
    assert "openalex_rate_limited" in degraded
    assert repo.get_collection(conn, cid)["degraded"] == ["openalex_rate_limited"]
    assert repo.get_collection(conn, cid)["status"] == "ready"


def make_client(tmp_path):
    settings = Settings(db_path=tmp_path / "api.sqlite3", data_dir=tmp_path)
    app = create_app(settings, warm_query_embedder=False)
    started = []
    app.state.upload_starter = lambda settings, **kw: started.append(kw)
    return TestClient(app), started


def test_uploading_a_pdf_creates_a_building_upload_collection(tmp_path):
    client, started = make_client(tmp_path)
    data = make_pdf()
    with client:
        response = client.post(
            "/api/collections/upload",
            files=[("files", ("paper.pdf", io.BytesIO(data), "application/pdf"))],
            data={"name": "My reading list"},
        )
        assert response.status_code == 202
        body = response.json()
        collection = client.get(f"/api/collections/{body['collection_id']}").json()
    assert collection["mode"] == "upload" and collection["query"] == "My reading list"
    assert collection["status"] == "building"
    assert started[0]["uploads"][0].sha256 == hashlib.sha256(data).hexdigest()


def test_a_non_pdf_upload_is_422_and_leaves_no_phantom_build(tmp_path):
    client, started = make_client(tmp_path)
    with client:
        response = client.post(
            "/api/collections/upload", files=[("files", ("x.txt", io.BytesIO(b"hello"), "text/plain"))]
        )
        collections = client.get("/api/collections").json()["collections"]
    assert response.status_code == 422 and response.json()["error"]["code"] == "invalid_upload"
    assert started == []
    assert collections == []
    assert not (tmp_path / "uploads").exists()


def test_a_request_over_the_total_cap_is_413_before_any_rows_exist(tmp_path, monkeypatch):
    data = make_pdf()
    monkeypatch.setattr("app.main.MAX_TOTAL_UPLOAD_BYTES", int(len(data) * 1.5), raising=False)
    client, started = make_client(tmp_path)
    with client:
        response = client.post(
            "/api/collections/upload",
            files=[
                ("files", ("a.pdf", io.BytesIO(data), "application/pdf")),
                ("files", ("b.pdf", io.BytesIO(data + b" "), "application/pdf")),
            ],
        )
        collections = client.get("/api/collections").json()["collections"]
    assert response.status_code == 413 and response.json()["error"]["code"] == "invalid_upload"
    assert "together" in response.json()["error"]["message"]
    assert started == [] and collections == []


def test_an_uploaded_file_can_be_read_back_but_nothing_else_can(tmp_path):
    client, _ = make_client(tmp_path)
    data = make_pdf()
    with client:
        init_db(tmp_path / "api.sqlite3")
        conn = connect(tmp_path / "api.sqlite3")
        cid = repo.create_collection(conn, mode="upload", query="u", embedding_model="m")
        uploads = validate_and_store(tmp_path, cid, [("p.pdf", data)])
        pid = repo.upsert_paper(conn, collection_id=cid, paper=repo.PaperRow(
            arxiv_id=None, openalex_id=None, doi=None, title="T", abstract=None, authors=(),
            year=None, venue=None, pdf_url=f"upload://{uploads[0].sha256}", citation_count=None,
            source="upload", parse_status="ok", parse_note=None, has_fulltext=1))
        arxiv_pid = repo.upsert_paper(conn, collection_id=cid, paper=repo.PaperRow(
            arxiv_id="2401.00001", openalex_id=None, doi=None, title="A", abstract=None, authors=(),
            year=None, venue=None, pdf_url="https://arxiv.org/pdf/2401.00001", citation_count=None,
            source="arxiv", parse_status="ok", parse_note=None, has_fulltext=0))
        ok = client.get(f"/api/papers/{pid}/pdf")
        arxiv = client.get(f"/api/papers/{arxiv_pid}/pdf")
        missing = client.get("/api/papers/nope/pdf")
        detail = client.get(f"/api/papers/{pid}").json()
    assert ok.status_code == 200 and ok.content == data
    assert arxiv.status_code == 404 and missing.status_code == 404
    assert detail["source_url"] == f"/api/papers/{pid}/pdf"


def test_line_break_hyphenation_is_repaired_in_the_extracted_abstract():
    hyphenated = (
        "We study how retrieval aug-\nmented generation grounds answers in source docu-\nments and "
        "measure how much it reduces unsupported claims across several question answering "
        "benchmarks, showing that evidence selection matters more than model size for accuracy."
    )
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((40, 60), "A Sufficiently Long Title For The Test", fontsize=20)
    page.insert_text((40, 100), "Abstract", fontsize=11)
    y = 116
    for line in hyphenated.split("\n"):
        page.insert_text((40, y), line, fontsize=9)
        y += 12
    page.insert_text((40, y + 20), "1 Introduction", fontsize=11)
    abstract = extract_metadata(doc.tobytes()).abstract
    assert "augmented" in abstract and "documents" in abstract and "aug-" not in abstract


def test_a_vertical_arxiv_stamp_is_never_taken_as_the_title():
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((20, 400), "arXiv:2305.06983v2 [cs.CL] 22 Oct 2023", fontsize=24, rotate=90)
    page.insert_text((60, 80), "The Real Title Of This Paper", fontsize=18)
    assert extract_metadata(doc.tobytes()).title == "The Real Title Of This Paper"


def test_a_horizontal_identifier_line_is_not_a_title_either():
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((60, 40), "arXiv:2305.06983v2 [cs.CL] 22 Oct 2023", fontsize=24)
    page.insert_text((60, 90), "The Real Title Of This Paper", fontsize=18)
    assert extract_metadata(doc.tobytes()).title == "The Real Title Of This Paper"


@pytest.mark.parametrize(
    "bad",
    ["Paper Title (use style: paper title)", "expert mind ieee verified.docx", "arXiv:2501.17397v1  [cs.CL]  29 Jan 2025", "Microsoft Word - draft.docx"],
)
def test_junk_embedded_metadata_titles_fall_back_to_the_layout_title(bad):
    assert extract_metadata(make_pdf(meta_title=bad)).title == "Grounded Answers From Retrieved Evidence"


def _footnote_pdf(doi=None):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((40, 60), "Our Own Novel Method For Sparse Retrieval", fontsize=20)
    page.insert_text((40, 120), "We build on prior work (see arXiv:1706.03762 for details).", fontsize=9)
    if doi:
        page.insert_text((40, 140), f"See also https://doi.org/{doi}", fontsize=9)
    return doc.tobytes()


def test_an_arxiv_id_cited_in_the_body_is_not_this_papers_id():
    meta = extract_metadata(_footnote_pdf())
    assert meta.arxiv_id is None


def test_only_the_papers_own_arxiv_stamp_counts():
    assert extract_metadata(make_pdf(stamp="2506.06962")).arxiv_id == "2506.06962"


def test_a_doi_whose_record_has_a_different_title_is_not_accepted(conn, tmp_path):
    other = FakeWork(openalex_id="W-OTHER", title="Attention Is All You Need", doi="10.1/x")
    openalex = FakeOpenAlex(by_doi={"10.5555/cited.elsewhere": other})
    pdf = _footnote_pdf(doi="10.5555/cited.elsewhere")
    build(conn, tmp_path, [("p.pdf", pdf)], openalex)
    paper = conn.execute("SELECT * FROM papers").fetchone()
    assert paper["title"] == "Our Own Novel Method For Sparse Retrieval"
    assert paper["openalex_id"] is None and paper["citation_count"] is None


def test_two_different_files_with_the_same_title_are_both_kept(conn, tmp_path):
    def pdf(body):
        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text((40, 60), "A Shared Title For Two Different Files", fontsize=20)
        page.insert_text((40, 100), body * 30, fontsize=9)
        return doc.tobytes()

    build(conn, tmp_path, [("a.pdf", pdf("alpha ")), ("b.pdf", pdf("beta "))])
    assert conn.execute("SELECT count(*) FROM papers").fetchone()[0] == 2


def test_an_unparseable_upload_gets_no_vectors_and_is_not_searchable(conn, tmp_path):
    from services.retrieval import rank_papers

    cid, _ = build(conn, tmp_path, [("scan_0001.pdf", b"%PDF-1.4\nbroken body")])
    assert conn.execute("SELECT count(*) FROM embeddings").fetchone()[0] == 0
    hits = rank_papers(conn, cid, query="scan 0001", mode="lexical", top_k=5)
    assert hits.papers == []


def test_the_word_keyword_inside_an_abstract_does_not_end_it():
    abstract = (
        "We study keyword search and an introduction to dense retrieval for scientific papers, "
        "measuring how each approach handles paraphrased queries across several benchmarks and "
        "showing that evidence selection matters more than model size for factual accuracy."
    )
    meta = extract_metadata(make_pdf(abstract=abstract))
    assert meta.abstract is not None
    assert "keyword search" in meta.abstract and meta.abstract.rstrip(".").endswith("accuracy")


def test_an_oversized_declared_length_is_413_before_the_body_is_parsed(tmp_path, monkeypatch):
    data = make_pdf()
    monkeypatch.setattr("app.main.MAX_TOTAL_UPLOAD_BYTES", len(data) // 2, raising=False)
    monkeypatch.setattr("app.main.UPLOAD_OVERHEAD_BYTES", 0, raising=False)

    def parsed(*args, **kwargs):
        raise AssertionError("the body was parsed")

    monkeypatch.setattr("app.main._read_capped", parsed)
    client, started = make_client(tmp_path)
    with client:
        response = client.post(
            "/api/collections/upload",
            files=[("files", ("a.pdf", io.BytesIO(data), "application/pdf"))],
        )
        collections = client.get("/api/collections").json()["collections"]
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "invalid_upload"
    assert started == [] and collections == []


def test_a_request_within_the_declared_limit_still_reaches_the_handler(tmp_path):
    client, started = make_client(tmp_path)
    with client:
        response = client.post(
            "/api/collections/upload",
            files=[("files", ("a.pdf", io.BytesIO(make_pdf()), "application/pdf"))],
        )
    assert response.status_code == 202 and len(started) == 1


def test_a_cancelled_upload_build_stops_before_the_next_file(conn, tmp_path):
    from services.collection import BuildCancelled

    cid = repo.create_collection(conn, mode="upload", query="Uploaded", embedding_model="m")
    job = repo.create_job(conn, collection_id=cid, kind="build")
    uploads = validate_and_store(
        tmp_path, cid,
        [(f"{n}.pdf", make_pdf(title=f"Paper number {n} on retrieval", body=BODY + f" {n}")) for n in range(3)],
    )

    class CancelAfterFirst(FakeEmbedder):
        done = 0

        def embed_texts(self, texts, batch_size=8):
            if len(texts) == 1 and "[SEP]" in texts[0]:
                CancelAfterFirst.done += 1
                if CancelAfterFirst.done == 1:
                    assert repo.cancel_build(conn, cid) == 1
            return super().embed_texts(texts, batch_size)

    with pytest.raises(BuildCancelled):
        build_upload_collection(
            conn, collection_id=cid, job_id=job, uploads=uploads, settings=Settings_(tmp_path),
            openalex=FakeOpenAlex(), embedder=CancelAfterFirst(), client=object(), log=lambda *_: None,
        )

    assert conn.execute("SELECT count(*) FROM papers").fetchone()[0] == 1
    assert repo.get_job(conn, job)["error"] == "cancelled"
    assert repo.get_collection(conn, cid)["error"] == "Cancelled."
