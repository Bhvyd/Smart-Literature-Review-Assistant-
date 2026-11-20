
from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

SCHEMA_PATH = Path(__file__).with_name("schema.sql")
MEMORY = ":memory:"


def connect(path: str | Path) -> sqlite3.Connection:
    target = str(path)
    if target != MEMORY:
        target = str(Path(path))
    conn = sqlite3.connect(target, check_same_thread=False, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(path: str | Path) -> None:
    if str(path) != MEMORY:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = connect(path)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        _migrate(conn)
        conn.commit()
    finally:
        conn.close()


def _migrate(conn: sqlite3.Connection) -> None:
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(collections)")}
    if "degraded" not in columns:
        conn.execute("ALTER TABLE collections ADD COLUMN degraded TEXT")
    job_columns = {row["name"] for row in conn.execute("PRAGMA table_info(jobs)")}
    if "updated_at" not in job_columns:
        conn.execute("ALTER TABLE jobs ADD COLUMN updated_at TEXT")


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    else:
        conn.commit()
