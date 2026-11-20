
import json
import shutil
import sqlite3
import zipfile

import pytest

from ingest.chunk import Chunk
from store import repo, snapshot
from store.db import connect, init_db


def seed(conn, query, titles):
    cid = repo.create_collection(conn, mode="topic", query=query, embedding_model="specter2")
    for n, title in enumerate(titles):
        pid = repo.upsert_paper(conn, collection_id=cid, paper=repo.PaperRow(
            arxiv_id=f"2401.{abs(hash(query)) % 9000 + n:05d}", openalex_id=f"W{n}", doi=None, title=title,
            abstract=f"abstract of {title}", authors=("A B",), year=2024, venue=None, pdf_url=None,
            citation_count=n, source="arxiv", parse_status="ok", parse_note=None, has_fulltext=1))
        repo.replace_chunks(conn, pid, [Chunk(seq=0, text=f"chunk text of {title}", n_tokens=5)])
        (chunk_id,) = conn.execute("SELECT id FROM chunks WHERE paper_id=?", (pid,)).fetchone()
        repo.save_embedding(conn, owner_type="chunk", owner_id=chunk_id, vec=[0.5, 0.25 * (n + 1)], model="m")
        repo.save_embedding(conn, owner_type="paper", owner_id=pid, vec=[1.0, float(n)], model="m")
        repo.upsert_edge(conn, collection_id=cid, src_paper_id=pid, dst_paper_id=None, dst_openalex_id=f"W9{n}",
                         dst_title=f"cited by {title}", dst_doi=None)
    repo.save_answer(conn, collection_id=cid, question="q?", answer="a [1].", citations=[{"n": 1}], refused=False,
                     model="m", latency_ms=5, retrieval_snapshot={"passages": []})
    conn.execute("INSERT OR REPLACE INTO llm_cache (key, provider, model, response, created_at) VALUES ('k','gemini','m','cached reply','t')")
    conn.commit()
    repo.set_collection_counts(conn, cid, n_papers=len(titles), n_chunks=len(titles))
    repo.set_collection_status(conn, cid, "ready")
    return cid


@pytest.fixture
def source(tmp_path):
    db = tmp_path / "src.sqlite3"
    init_db(db)
    conn = connect(db)
    keep = seed(conn, "retrieval", ["Alpha", "Beta", "Gamma"])
    other = seed(conn, "graphs", ["Delta", "Epsilon"])
    conn.close()
    return db, keep, other, tmp_path


def rows(db, sql, *params):
    conn = sqlite3.connect(db)
    try:
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


def test_only_the_chosen_collection_is_exported_and_the_manifest_is_honest(source):
    db, keep, other, tmp = source
    manifest = snapshot.export_collection(db, keep, tmp / "s.zip")
    assert manifest["counts"]["collections"] == 1 and manifest["counts"]["papers"] == 3
    assert manifest["counts"]["chunks"] == 3 and manifest["counts"]["embeddings"] == 6
    assert manifest["counts"]["edges"] == 3 and manifest["counts"]["answers"] == 1
    assert manifest["contains_pdfs"] is False and manifest["contains_credentials"] is False
    assert snapshot.verify(tmp / "s.zip")["collection"]["id"] == keep


def test_a_round_trip_reproduces_text_and_vectors_exactly(source):
    db, keep, _, tmp = source
    snapshot.export_collection(db, keep, tmp / "s.zip")
    fresh = tmp / "fresh.sqlite3"
    assert snapshot.import_snapshot(tmp / "s.zip", fresh)["collection_id"] == keep

    queries = {
        "papers": ("SELECT id, title, abstract, year, citation_count, openalex_id FROM papers WHERE collection_id=? ORDER BY id", (keep,)),
        "chunks": ("SELECT c.id, c.text, c.n_tokens FROM chunks c JOIN papers p ON p.id=c.paper_id WHERE p.collection_id=? ORDER BY c.id", (keep,)),
        "vectors": ("SELECT e.owner_type, e.owner_id, e.dim, e.vec FROM embeddings e WHERE "
                    "(e.owner_type='paper' AND e.owner_id IN (SELECT id FROM papers WHERE collection_id=?)) OR "
                    "(e.owner_type='chunk' AND e.owner_id IN (SELECT c.id FROM chunks c JOIN papers p ON p.id=c.paper_id WHERE p.collection_id=?)) "
                    "ORDER BY e.owner_type, e.owner_id", (keep, keep)),
        "edges": ("SELECT id, src_paper_id, dst_openalex_id, dst_title FROM edges WHERE collection_id=? ORDER BY id", (keep,)),
        "answers": ("SELECT question, answer, citations, retrieval_snapshot FROM answers WHERE collection_id=? ORDER BY id", (keep,)),
    }
    for name, (sql, params) in queries.items():
        original, copied = rows(db, sql, *params), rows(fresh, sql, *params)
        assert original and copied == original, f"{name} differs after a round trip"
    assert rows(fresh, "SELECT response FROM llm_cache")[0][0] == "cached reply"


def test_the_other_collection_never_leaks_into_the_snapshot(source):
    db, keep, other, tmp = source
    snapshot.export_collection(db, keep, tmp / "s.zip")
    fresh = tmp / "fresh.sqlite3"
    snapshot.import_snapshot(tmp / "s.zip", fresh)
    titles = {row[0] for row in rows(fresh, "SELECT title FROM papers")}
    assert titles == {"Alpha", "Beta", "Gamma"} and not titles & {"Delta", "Epsilon"}


def test_a_modified_database_is_rejected(source):
    db, keep, _, tmp = source
    snapshot.export_collection(db, keep, tmp / "s.zip")
    tampered = tmp / "t.zip"
    with zipfile.ZipFile(tmp / "s.zip") as src, zipfile.ZipFile(tampered, "w") as dst:
        for item in src.infolist():
            data = src.read(item.filename)
            if item.filename == snapshot.DB_NAME:
                data = data[:-8] + b"\x00" * 8
            dst.writestr(item, data)
    with pytest.raises(snapshot.SnapshotError, match="checksum|integrity"):
        snapshot.verify(tampered)
    with pytest.raises(snapshot.SnapshotError):
        snapshot.import_snapshot(tampered, tmp / "never.sqlite3")


def test_a_manifest_that_lies_about_counts_is_rejected(source):
    db, keep, _, tmp = source
    snapshot.export_collection(db, keep, tmp / "s.zip")
    forged = tmp / "f.zip"
    with zipfile.ZipFile(tmp / "s.zip") as src, zipfile.ZipFile(forged, "w") as dst:
        for item in src.infolist():
            data = src.read(item.filename)
            if item.filename == snapshot.MANIFEST_NAME:
                manifest = json.loads(data)
                manifest["counts"]["papers"] = 99
                data = json.dumps(manifest).encode()
            dst.writestr(item, data)
    with pytest.raises(snapshot.SnapshotError, match="row counts differ"):
        snapshot.verify(forged)


def test_garbage_and_unsafe_archives_are_rejected(tmp_path):
    (tmp_path / "x.zip").write_bytes(b"not a zip")
    with pytest.raises(snapshot.SnapshotError, match="not a readable snapshot"):
        snapshot.verify(tmp_path / "x.zip")
    evil = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil, "w") as archive:
        archive.writestr(snapshot.MANIFEST_NAME, json.dumps({"format_version": 1, "db_sha256": "x", "counts": {}}))
        archive.writestr(snapshot.DB_NAME, b"x")
        archive.writestr("../escape.txt", b"x")
    with pytest.raises(snapshot.SnapshotError, match="unsafe path"):
        snapshot.verify(evil)


def test_import_never_silently_overwrites_and_replace_is_explicit(source):
    db, keep, _, tmp = source
    snapshot.export_collection(db, keep, tmp / "s.zip")
    fresh = tmp / "fresh.sqlite3"
    snapshot.import_snapshot(tmp / "s.zip", fresh)
    with pytest.raises(snapshot.SnapshotError, match="already exists"):
        snapshot.import_snapshot(tmp / "s.zip", fresh)
    conn = connect(fresh)
    conn.execute("UPDATE papers SET title='locally edited'")
    conn.commit()
    conn.close()
    snapshot.import_snapshot(tmp / "s.zip", fresh, replace=True)
    assert {row[0] for row in rows(fresh, "SELECT title FROM papers")} == {"Alpha", "Beta", "Gamma"}
    assert rows(fresh, "SELECT count(*) FROM papers")[0][0] == 3


def test_only_a_finished_collection_can_be_snapshotted(source):
    db, keep, _, tmp = source
    conn = connect(db)
    repo.set_collection_status(conn, keep, "building")
    conn.close()
    with pytest.raises(snapshot.SnapshotError, match="not ready"):
        snapshot.export_collection(db, keep, tmp / "s.zip")
    with pytest.raises(snapshot.SnapshotError, match="does not exist"):
        snapshot.export_collection(db, "nope", tmp / "s.zip")


def test_an_imported_snapshot_serves_the_real_api_endpoints(source):
    from fastapi.testclient import TestClient

    from app.config import Settings
    from app.main import create_app

    db, keep, _, tmp = source
    snapshot.export_collection(db, keep, tmp / "s.zip")
    fresh = tmp / "clean" / "app.sqlite3"
    snapshot.import_snapshot(tmp / "s.zip", fresh)
    with TestClient(create_app(Settings(db_path=fresh, data_dir=tmp / "clean"), warm_query_embedder=False)) as client:
        papers = client.get(f"/api/collections/{keep}/papers", params={"q": "Alpha", "mode": "lexical"}).json()
        graph = client.get(f"/api/collections/{keep}/graph", params={"include_external": "true"}).json()
        listing = client.get("/api/collections").json()["collections"]
    assert [p["title"] for p in papers["papers"]][:1] == ["Alpha"]
    assert graph["reason"] == "not_enough_edges" and graph["have"] == 3
    assert listing[0]["id"] == keep and listing[0]["status"] == "ready"
