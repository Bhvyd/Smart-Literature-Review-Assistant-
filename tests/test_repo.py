
import pytest

from ingest.chunk import Chunk
from store.db import connect, init_db
from store.repo import (
    PaperRow,
    count_chunks,
    count_papers,
    create_collection,
    create_job,
    load_embedding,
    replace_chunks,
    save_embedding,
    set_collection_status,
    update_job,
    upsert_edge,
    upsert_paper,
)


@pytest.fixture
def conn(tmp_path):
    db = tmp_path / "t.sqlite3"
    init_db(db)
    return connect(db)


def _paper(arxiv_id="2402.01788", **kw):
    base = dict(
        arxiv_id=arxiv_id,
        openalex_id="W1",
        doi=f"10.1/{arxiv_id}",
        title="T",
        abstract="A",
        authors=("Ada",),
        year=2024,
        venue="V",
        pdf_url="u",
        citation_count=3,
        source="arxiv",
        parse_status="ok",
        parse_note=None,
        has_fulltext=1,
    )
    base.update(kw)
    return PaperRow(**base)


def test_collection_lifecycle_and_job_progress(conn):
    cid = create_collection(conn, mode="topic", query="attention", embedding_model="specter2")
    set_collection_status(conn, cid, "ready")
    job = create_job(conn, collection_id=cid, kind="build")
    update_job(conn, job, status="done", progress=1.0, message="ok")
    row = conn.execute("SELECT status FROM collections WHERE id=?", (cid,)).fetchone()
    assert row["status"] == "ready"
    job_row = conn.execute("SELECT status, progress FROM jobs WHERE id=?", (job,)).fetchone()
    assert job_row["status"] == "done"
    assert job_row["progress"] == 1.0


def test_new_collection_starts_building(conn):
    cid = create_collection(conn, mode="topic", query="q", embedding_model="m")
    row = conn.execute("SELECT status, mode, query FROM collections WHERE id=?", (cid,)).fetchone()
    assert row["status"] == "building"
    assert row["mode"] == "topic"
    assert row["query"] == "q"


def test_upsert_paper_is_idempotent(conn):
    cid = create_collection(conn, mode="topic", query="q", embedding_model="m")
    p1 = upsert_paper(conn, collection_id=cid, paper=_paper())
    p2 = upsert_paper(conn, collection_id=cid, paper=_paper(title="Updated"))
    assert p1 == p2
    assert count_papers(conn, cid) == 1
    title = conn.execute("SELECT title FROM papers WHERE id=?", (p1,)).fetchone()["title"]
    assert title == "Updated"


def test_uploaded_papers_without_arxiv_ids_do_not_collide(conn):
    cid = create_collection(conn, mode="upload", query=None, embedding_model="m")
    a = upsert_paper(
        conn,
        collection_id=cid,
        paper=_paper(arxiv_id=None, openalex_id=None, doi=None, title="One", source="upload"),
    )
    b = upsert_paper(
        conn,
        collection_id=cid,
        paper=_paper(arxiv_id=None, openalex_id=None, doi=None, title="Two", source="upload"),
    )
    assert a != b
    assert count_papers(conn, cid) == 2


def test_paper_authors_roundtrip_as_json(conn):
    cid = create_collection(conn, mode="topic", query="q", embedding_model="m")
    pid = upsert_paper(conn, collection_id=cid, paper=_paper(authors=("Ada", "Grace")))
    raw = conn.execute("SELECT authors FROM papers WHERE id=?", (pid,)).fetchone()["authors"]
    import json

    assert json.loads(raw) == ["Ada", "Grace"]


def test_two_listings_of_one_doi_collapse_onto_a_single_row(conn):
    cid = create_collection(conn, mode="topic", query="q", embedding_model="m")
    first = upsert_paper(conn, collection_id=cid, paper=_paper(arxiv_id="2402.00001"))
    second = upsert_paper(
        conn,
        collection_id=cid,
        paper=_paper(arxiv_id="2402.99999", openalex_id="W99", doi="10.1/2402.00001"),
    )
    assert first == second
    assert count_papers(conn, cid) == 1


def test_replace_chunks_does_not_duplicate(conn):
    cid = create_collection(conn, mode="topic", query="q", embedding_model="m")
    pid = upsert_paper(conn, collection_id=cid, paper=_paper())
    chunks = [Chunk(seq=0, text="a b", n_tokens=2), Chunk(seq=1, text="c d", n_tokens=2)]
    assert replace_chunks(conn, pid, chunks) == 2
    assert replace_chunks(conn, pid, chunks) == 2
    assert count_chunks(conn, cid) == 2


def test_replacing_chunks_removes_their_orphaned_embeddings(conn):
    cid = create_collection(conn, mode="topic", query="q", embedding_model="m")
    pid = upsert_paper(conn, collection_id=cid, paper=_paper())
    chunks = [Chunk(seq=0, text="a b", n_tokens=2), Chunk(seq=1, text="c d", n_tokens=2)]
    replace_chunks(conn, pid, chunks)
    for chunk_id, _ in conn.execute("SELECT id, seq FROM chunks").fetchall():
        save_embedding(conn, owner_type="chunk", owner_id=chunk_id, vec=[1.0], model="m")
    assert conn.execute("SELECT count(*) FROM embeddings").fetchone()[0] == 2

    replace_chunks(conn, pid, [Chunk(seq=0, text="a b", n_tokens=2)])
    assert conn.execute("SELECT count(*) FROM chunks").fetchone()[0] == 1
    assert conn.execute("SELECT count(*) FROM embeddings").fetchone()[0] == 0


def test_embeddings_roundtrip_preserves_values(conn):
    save_embedding(conn, owner_type="chunk", owner_id="c1", vec=[0.25, -1.5, 3.0], model="specter2")
    assert load_embedding(conn, "chunk", "c1") == pytest.approx([0.25, -1.5, 3.0])


def test_embedding_upsert_replaces_instead_of_duplicating(conn):
    save_embedding(conn, owner_type="chunk", owner_id="c1", vec=[1.0], model="m")
    save_embedding(conn, owner_type="chunk", owner_id="c1", vec=[2.0], model="m")
    assert load_embedding(conn, "chunk", "c1") == pytest.approx([2.0])
    assert conn.execute("SELECT count(*) FROM embeddings").fetchone()[0] == 1


def test_load_embedding_missing_returns_none(conn):
    assert load_embedding(conn, "chunk", "nope") is None


def test_edges_are_idempotent_and_counted(conn):
    cid = create_collection(conn, mode="topic", query="q", embedding_model="m")
    p1 = upsert_paper(conn, collection_id=cid, paper=_paper(arxiv_id="2402.00001", openalex_id="W1"))
    p2 = upsert_paper(conn, collection_id=cid, paper=_paper(arxiv_id="2402.00002", openalex_id="W2"))
    for _ in range(2):
        upsert_edge(
            conn,
            collection_id=cid,
            src_paper_id=p1,
            dst_paper_id=p2,
            dst_openalex_id="W2",
            dst_title="T2",
            dst_doi=None,
        )
    assert conn.execute("SELECT count(*) FROM edges").fetchone()[0] == 1


def test_resolved_edge_is_marked_resolved(conn):
    cid = create_collection(conn, mode="topic", query="q", embedding_model="m")
    p1 = upsert_paper(conn, collection_id=cid, paper=_paper(arxiv_id="2402.00001", openalex_id="W1"))
    p2 = upsert_paper(conn, collection_id=cid, paper=_paper(arxiv_id="2402.00002", openalex_id="W2"))
    upsert_edge(
        conn,
        collection_id=cid,
        src_paper_id=p1,
        dst_paper_id=p2,
        dst_openalex_id="W2",
        dst_title="T2",
        dst_doi=None,
    )
    assert conn.execute("SELECT resolved FROM edges").fetchone()["resolved"] == 1


def test_unresolved_edge_is_stored_with_its_real_title(conn):
    cid = create_collection(conn, mode="topic", query="q", embedding_model="m")
    p1 = upsert_paper(conn, collection_id=cid, paper=_paper())
    upsert_edge(
        conn,
        collection_id=cid,
        src_paper_id=p1,
        dst_paper_id=None,
        dst_openalex_id="W9",
        dst_title="Some Other Paper",
        dst_doi=None,
    )
    row = conn.execute("SELECT resolved, dst_title FROM edges").fetchone()
    assert row["resolved"] == 0
    assert row["dst_title"] == "Some Other Paper"


def test_job_failure_records_the_error(conn):
    cid = create_collection(conn, mode="topic", query="q", embedding_model="m")
    job = create_job(conn, collection_id=cid, kind="build")
    update_job(conn, job, status="failed", progress=0.4, error="arXiv unreachable")
    row = conn.execute("SELECT status, error, finished_at FROM jobs WHERE id=?", (job,)).fetchone()
    assert row["status"] == "failed"
    assert row["error"] == "arXiv unreachable"
    assert row["finished_at"] is not None


def test_delete_collection_removes_its_rows_and_only_its_rows(conn):
    from store.repo import delete_collection

    def build(query, arxiv_id):
        cid = create_collection(conn, mode="topic", query=query, embedding_model="specter2")
        pid = upsert_paper(conn, collection_id=cid, paper=_paper(arxiv_id))
        replace_chunks(conn, pid, [Chunk(seq=0, text="hello world", n_tokens=2)])
        upsert_edge(conn, collection_id=cid, src_paper_id=pid, dst_paper_id=None,
                    dst_openalex_id="W9", dst_title="Ref", dst_doi=None)
        conn.execute("INSERT INTO embeddings (id, owner_type, owner_id, dim, vec, model) VALUES (?, 'paper', ?, 1, ?, 'm')",
                     (f"e-{pid}", pid, b"\x00\x00\x00\x00"))
        conn.commit()
        create_job(conn, collection_id=cid, kind="build")
        set_collection_status(conn, cid, "ready")
        return cid

    keep, drop = build("keep me", "2402.00001"), build("drop me", "2402.00002")
    removed = delete_collection(conn, drop)
    assert removed["collections"] == 1 and removed["papers"] == 1 and removed["chunks"] == 1
    assert removed["edges"] == 1 and removed["jobs"] == 1 and removed["embeddings"] >= 1

    for table, column in (("papers", "collection_id"), ("edges", "collection_id"), ("jobs", "collection_id")):
        assert conn.execute(f"SELECT COUNT(*) FROM {table} WHERE {column}=?", (drop,)).fetchone()[0] == 0
        assert conn.execute(f"SELECT COUNT(*) FROM {table} WHERE {column}=?", (keep,)).fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0] == 1


def test_delete_collection_refuses_a_running_build_and_an_unknown_id(conn):
    from store.repo import delete_collection

    cid = create_collection(conn, mode="topic", query="busy", embedding_model="specter2")
    with pytest.raises(ValueError):
        delete_collection(conn, cid)
    with pytest.raises(KeyError):
        delete_collection(conn, "nope")


def test_a_failed_job_that_later_finishes_done_loses_its_stale_error(conn):
    cid = create_collection(conn, mode="topic", query="q", embedding_model="m")
    job = create_job(conn, collection_id=cid, kind="build")
    update_job(conn, job, status="failed", error="interrupted by a server restart")
    update_job(conn, job, status="done", progress=1.0, message="collection ready")
    row = conn.execute("SELECT status, error FROM jobs WHERE id=?", (job,)).fetchone()
    assert row["status"] == "done"
    assert row["error"] is None


def test_an_explicit_error_is_kept_even_when_the_status_is_done(conn):
    cid = create_collection(conn, mode="topic", query="q", embedding_model="m")
    job = create_job(conn, collection_id=cid, kind="build")
    update_job(conn, job, status="done", error="finished with warnings")
    assert conn.execute("SELECT error FROM jobs WHERE id=?", (job,)).fetchone()["error"] == "finished with warnings"


def test_a_late_progress_write_does_not_resurrect_a_failed_job(conn):
    cid = create_collection(conn, mode="topic", query="q", embedding_model="m")
    job = create_job(conn, collection_id=cid, kind="build")
    update_job(conn, job, status="failed", error="interrupted by a server restart")
    before = conn.execute("SELECT status, error, finished_at FROM jobs WHERE id=?", (job,)).fetchone()
    for late in ("running", "queued"):
        update_job(conn, job, status=late, progress=0.7, message="ingested 7/10")
    after = conn.execute("SELECT status, error, finished_at, progress FROM jobs WHERE id=?", (job,)).fetchone()
    assert (after["status"], after["error"], after["finished_at"]) == tuple(before)
    assert after["progress"] != 0.7
    update_job(conn, job, status="done", progress=1.0)
    done = conn.execute("SELECT status, error FROM jobs WHERE id=?", (job,)).fetchone()
    assert (done["status"], done["error"]) == ("done", None)
