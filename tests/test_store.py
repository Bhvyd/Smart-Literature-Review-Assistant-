
import sqlite3

import pytest

from store.db import connect, init_db, transaction

EXPECTED = {
    "collections",
    "papers",
    "chunks",
    "embeddings",
    "edges",
    "projections",
    "answers",
    "jobs",
    "llm_cache",
}


def test_init_creates_all_tables(tmp_path):
    db = tmp_path / "t.sqlite3"
    init_db(db)
    conn = connect(db)
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    assert EXPECTED <= {r["name"] for r in rows}


def test_init_is_idempotent(tmp_path):
    db = tmp_path / "t.sqlite3"
    init_db(db)
    init_db(db)
    conn = connect(db)
    assert conn.execute("SELECT count(*) FROM sqlite_master WHERE type='table'").fetchone()[0] >= len(
        EXPECTED
    )


def test_rows_are_addressable_by_name(tmp_path):
    db = tmp_path / "t.sqlite3"
    init_db(db)
    conn = connect(db)
    conn.execute(
        "INSERT INTO collections (id, mode, status, created_at, updated_at) VALUES (?,?,?,?,?)",
        ("c1", "topic", "building", "2026-11-10", "2026-11-10"),
    )
    conn.commit()
    row = conn.execute("SELECT id, mode FROM collections WHERE id = ?", ("c1",)).fetchone()
    assert row["id"] == "c1"
    assert row["mode"] == "topic"


def test_foreign_keys_enforced(tmp_path):
    db = tmp_path / "t.sqlite3"
    init_db(db)
    conn = connect(db)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO papers (id, collection_id, title, source) VALUES (?,?,?,?)",
            ("p1", "missing-collection", "T", "arxiv"),
        )
        conn.commit()


def test_duplicate_arxiv_id_in_same_collection_rejected(tmp_path):
    db = tmp_path / "t.sqlite3"
    init_db(db)
    conn = connect(db)
    conn.execute(
        "INSERT INTO collections (id, mode, status, created_at, updated_at) VALUES (?,?,?,?,?)",
        ("c1", "topic", "building", "2026-11-10", "2026-11-10"),
    )
    conn.execute(
        "INSERT INTO papers (id, collection_id, arxiv_id, title, source) VALUES (?,?,?,?,?)",
        ("p1", "c1", "2402.01788", "A", "arxiv"),
    )
    conn.commit()
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO papers (id, collection_id, arxiv_id, title, source) VALUES (?,?,?,?,?)",
            ("p2", "c1", "2402.01788", "B", "arxiv"),
        )
        conn.commit()


def test_same_arxiv_id_in_a_different_collection_is_allowed(tmp_path):
    db = tmp_path / "t.sqlite3"
    init_db(db)
    conn = connect(db)
    for collection in ("c1", "c2"):
        conn.execute(
            "INSERT INTO collections (id, mode, status, created_at, updated_at) VALUES (?,?,?,?,?)",
            (collection, "topic", "building", "2026-11-10", "2026-11-10"),
        )
    conn.execute(
        "INSERT INTO papers (id, collection_id, arxiv_id, title, source) VALUES (?,?,?,?,?)",
        ("p1", "c1", "2402.01788", "A", "arxiv"),
    )
    conn.execute(
        "INSERT INTO papers (id, collection_id, arxiv_id, title, source) VALUES (?,?,?,?,?)",
        ("p2", "c2", "2402.01788", "A", "arxiv"),
    )
    conn.commit()
    assert conn.execute("SELECT count(*) FROM papers").fetchone()[0] == 2


def test_null_arxiv_ids_do_not_collide(tmp_path):
    db = tmp_path / "t.sqlite3"
    init_db(db)
    conn = connect(db)
    conn.execute(
        "INSERT INTO collections (id, mode, status, created_at, updated_at) VALUES (?,?,?,?,?)",
        ("c1", "upload", "building", "2026-11-10", "2026-11-10"),
    )
    for paper in ("p1", "p2"):
        conn.execute(
            "INSERT INTO papers (id, collection_id, title, source) VALUES (?,?,?,?)",
            (paper, "c1", "T", "upload"),
        )
    conn.commit()
    assert conn.execute("SELECT count(*) FROM papers").fetchone()[0] == 2


def test_transaction_rolls_back_on_error(tmp_path):
    db = tmp_path / "t.sqlite3"
    init_db(db)
    conn = connect(db)
    with pytest.raises(RuntimeError):
        with transaction(conn):
            conn.execute(
                "INSERT INTO collections (id, mode, status, created_at, updated_at) VALUES (?,?,?,?,?)",
                ("c1", "topic", "building", "2026-11-10", "2026-11-10"),
            )
            raise RuntimeError("boom")
    assert conn.execute("SELECT count(*) FROM collections").fetchone()[0] == 0


def test_transaction_commits_on_success(tmp_path):
    db = tmp_path / "t.sqlite3"
    init_db(db)
    conn = connect(db)
    with transaction(conn):
        conn.execute(
            "INSERT INTO collections (id, mode, status, created_at, updated_at) VALUES (?,?,?,?,?)",
            ("c1", "topic", "building", "2026-11-10", "2026-11-10"),
        )
    assert conn.execute("SELECT count(*) FROM collections").fetchone()[0] == 1


def test_in_memory_database_works():
    init_db(":memory:")
