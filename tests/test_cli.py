
import json

import pytest

from cli import main
from store.db import connect, init_db
from store.repo import create_collection, set_collection_status


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "cli.sqlite3"
    init_db(path)
    return path


def test_stats_reports_a_real_collection(db, capsys):
    conn = connect(db)
    cid = create_collection(conn, mode="topic", query="attention", embedding_model="m")
    set_collection_status(conn, cid, "ready")
    conn.close()

    assert main(["--db", str(db), "stats", "--collection", cid]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["collection_id"] == cid
    assert payload["query"] == "attention"
    assert payload["status"] == "ready"
    assert payload["n_papers"] == 0


def test_stats_rejects_an_unknown_collection(db, capsys):
    assert main(["--db", str(db), "stats", "--collection", "nope"]) == 1
    assert "not found" in capsys.readouterr().err


def test_stats_reports_a_missing_database(tmp_path, capsys):
    missing = tmp_path / "absent.sqlite3"
    assert main(["--db", str(missing), "stats", "--collection", "x"]) == 1
    assert "no database" in capsys.readouterr().err


def test_a_subcommand_is_required(capsys):
    with pytest.raises(SystemExit):
        main([])
