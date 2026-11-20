
import time
import zlib

import pymupdf

import _isolation_targets as targets
from ingest.pdf import extract_metadata, extract_text, parse_isolated


def _pdf():
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((40, 60), "An Ordinary Paper About Retrieval Systems", fontsize=20)
    page.insert_text((40, 100), "body text " * 60, fontsize=9)
    return doc.tobytes()


def test_a_normal_pdf_parses_to_the_same_result_as_in_process_parsing():
    data = _pdf()
    meta, text, problem = parse_isolated(data, timeout=60)
    assert problem is None
    assert meta == extract_metadata(data) and text == extract_text(data)


def test_a_parser_that_hangs_is_killed_at_the_timeout_and_reported():
    started = time.perf_counter()
    meta, text, problem = parse_isolated(b"%PDF-", timeout=1.5, target=targets.hang_forever)
    elapsed = time.perf_counter() - started
    assert elapsed < 15
    assert text.quality == "failed" and meta.title is None
    assert "longer than 1.5 s" in problem


def test_a_parser_that_crashes_is_reported_not_propagated():
    _, text, problem = parse_isolated(b"%PDF-", timeout=30, target=targets.crash_hard)
    assert text.quality == "failed" and "crashed" in problem


def test_an_exception_in_the_child_becomes_a_reason():
    _, text, problem = parse_isolated(b"%PDF-", timeout=30, target=targets.raise_error)
    assert text.quality == "failed" and "boom" in problem


def test_a_timeout_in_an_upload_leaves_the_file_listed_with_its_reason(tmp_path, monkeypatch):
    import ingest.pdf as pdf_module
    from services.upload import build_upload_collection, validate_and_store
    from store import repo
    from store.db import connect, init_db

    real = pdf_module.parse_isolated
    monkeypatch.setattr(
        "services.upload.parse_isolated",
        lambda data, timeout: real(data, timeout=1.0, target=targets.hang_forever),
    )

    class FakeOpenAlex:
        def lookup_by_arxiv(self, a): return None
        def lookup_by_doi(self, d): return None
        def lookup_by_title(self, t): return None
        def titles_for(self, ids, on_error=None): return {}

    class FakeEmbedder:
        model_name = "m"
        def count_tokens(self, t): return len(t.split())
        def embed_texts(self, texts, batch_size=8): return [[1.0, 0.0] for _ in texts]
        def paper_text(self, t, a): return t
        def close(self): pass

    class Settings_:
        cache_dir = str(tmp_path); contact_email = None; embedding_model = "m"; data_dir = str(tmp_path)

    init_db(tmp_path / "i.sqlite3")
    conn = connect(tmp_path / "i.sqlite3")
    cid = repo.create_collection(conn, mode="upload", query="u", embedding_model="m")
    job = repo.create_job(conn, collection_id=cid, kind="build")
    uploads = validate_and_store(tmp_path, cid, [("bomb.pdf", _pdf())])
    build_upload_collection(conn, collection_id=cid, job_id=job, uploads=uploads, settings=Settings_(),
                            openalex=FakeOpenAlex(), embedder=FakeEmbedder(), client=object(), log=lambda *_: None)
    paper = conn.execute("SELECT * FROM papers").fetchone()
    assert paper["parse_status"] == "failed" and "longer than" in paper["parse_note"]
    assert conn.execute("SELECT count(*) FROM embeddings").fetchone()[0] == 0
    assert repo.get_collection(conn, cid)["status"] == "ready"
