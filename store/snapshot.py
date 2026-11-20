
from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from store.db import connect, init_db

FORMAT_VERSION = 1
DB_NAME = "collection.sqlite3"
MANIFEST_NAME = "manifest.json"
EVAL_NAME = "evaluation.json"
_COLLECTION_TABLES = ("papers", "edges", "projections", "answers")


class SnapshotError(Exception):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _counts(conn: sqlite3.Connection) -> dict[str, int]:
    tables = ("collections", "papers", "chunks", "embeddings", "edges", "projections", "answers", "llm_cache")
    return {table: int(conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]) for table in tables}


def _columns(conn: sqlite3.Connection, table: str, schema: str = "main") -> list[str]:
    return [row[1] for row in conn.execute(f"PRAGMA {schema}.table_info({table})")]


def export_collection(
    source_db: str | Path,
    collection_id: str,
    out_zip: str | Path,
    *,
    evaluation_json: str | Path | None = None,
    include_llm_cache: bool = True,
) -> dict[str, Any]:
    source = connect(source_db)
    try:
        row = source.execute("SELECT * FROM collections WHERE id=?", (collection_id,)).fetchone()
        if row is None:
            raise SnapshotError(f"collection {collection_id} does not exist")
        if row["status"] != "ready":
            raise SnapshotError(f"collection is {row['status']}, not ready; only a finished build is snapshotted")

        with tempfile.TemporaryDirectory() as tmp:
            snapshot_db = Path(tmp) / DB_NAME
            init_db(snapshot_db)
            target = connect(snapshot_db)
            try:
                target.execute("ATTACH DATABASE ? AS src", (str(source_db),))
                _insert(target, "collections", "id = ?", (collection_id,))
                for table in _COLLECTION_TABLES:
                    _insert(target, table, "collection_id = ?", (collection_id,))
                _insert(target, "chunks", "paper_id IN (SELECT id FROM src.papers WHERE collection_id = ?)", (collection_id,))
                _insert(
                    target, "embeddings",
                    "(owner_type='paper' AND owner_id IN (SELECT id FROM src.papers WHERE collection_id = ?)) OR "
                    "(owner_type='chunk' AND owner_id IN (SELECT c.id FROM src.chunks c JOIN src.papers p ON p.id=c.paper_id "
                    "WHERE p.collection_id = ?))",
                    (collection_id, collection_id),
                )
                if include_llm_cache:
                    _insert(target, "llm_cache", "1=1", ())
                target.commit()
                target.execute("DETACH DATABASE src")
                counts = _counts(target)
                target.execute("VACUUM")
            finally:
                target.close()

            manifest = {
                "format_version": FORMAT_VERSION,
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "collection": {
                    "id": collection_id, "mode": row["mode"], "query": row["query"],
                    "embedding_model": row["embedding_model"], "n_papers": row["n_papers"], "n_chunks": row["n_chunks"],
                },
                "counts": counts,
                "db_sha256": _sha256(snapshot_db),
                "contains_pdfs": False,
                "contains_model_weights": False,
                "contains_credentials": False,
                "note": "Real data fetched from arXiv/OpenAlex and embedded with SPECTER2. Import with "
                "`python -m cli snapshot import`. Dense query search needs the SPECTER2 weights locally.",
            }
            out = Path(out_zip)
            out.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
                archive.write(snapshot_db, DB_NAME)
                archive.writestr(MANIFEST_NAME, json.dumps(manifest, indent=2))
                if evaluation_json and Path(evaluation_json).is_file():
                    archive.write(evaluation_json, EVAL_NAME)
            return manifest
    finally:
        source.close()


def _insert(target: sqlite3.Connection, table: str, where: str, params: tuple) -> None:
    columns = _columns(target, table)
    names = ", ".join(columns)
    target.execute(f"INSERT OR REPLACE INTO main.{table} ({names}) SELECT {names} FROM src.{table} WHERE {where}", params)


def read_manifest(zip_path: str | Path) -> dict[str, Any]:
    try:
        with zipfile.ZipFile(zip_path) as archive:
            return json.loads(archive.read(MANIFEST_NAME))
    except (zipfile.BadZipFile, KeyError, ValueError, OSError) as exc:
        raise SnapshotError(f"not a readable snapshot: {exc}") from exc


def verify(zip_path: str | Path) -> dict[str, Any]:
    manifest = read_manifest(zip_path)
    if manifest.get("format_version") != FORMAT_VERSION:
        raise SnapshotError(f"unsupported snapshot format {manifest.get('format_version')!r}")
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(zip_path) as archive:
            names = set(archive.namelist())
            if DB_NAME not in names:
                raise SnapshotError("the snapshot has no database")
            if any(name.startswith(("/", "\\")) or ".." in Path(name).parts for name in names):
                raise SnapshotError("the snapshot contains an unsafe path")
            archive.extract(DB_NAME, tmp)
        db = Path(tmp) / DB_NAME
        if _sha256(db) != manifest.get("db_sha256"):
            raise SnapshotError("the database does not match its recorded checksum: the snapshot was modified")
        conn = sqlite3.connect(db)
        try:
            if conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise SnapshotError("the snapshot database failed SQLite's integrity check")
            conn.row_factory = sqlite3.Row
            actual = _counts(conn)
            if actual != manifest.get("counts"):
                raise SnapshotError(f"row counts differ from the manifest: {actual} != {manifest.get('counts')}")
            if conn.execute("PRAGMA foreign_key_check").fetchall():
                raise SnapshotError("the snapshot has rows that reference missing parents")
        finally:
            conn.close()
    return manifest


def import_snapshot(
    zip_path: str | Path, target_db: str | Path, *, replace: bool = False, data_dir: str | Path | None = None
) -> dict[str, Any]:
    manifest = verify(zip_path)
    collection_id = manifest["collection"]["id"]
    init_db(target_db)
    target = connect(target_db)
    try:
        existing = target.execute("SELECT id FROM collections WHERE id=?", (collection_id,)).fetchone()
        if existing and not replace:
            raise SnapshotError(f"collection {collection_id} already exists here; pass replace=True to overwrite it")
        with tempfile.TemporaryDirectory() as tmp:
            with zipfile.ZipFile(zip_path) as archive:
                archive.extract(DB_NAME, tmp)
                if data_dir and EVAL_NAME in archive.namelist():
                    report_dir = Path(data_dir) / "eval"
                    report_dir.mkdir(parents=True, exist_ok=True)
                    (report_dir / f"{collection_id}.json").write_bytes(archive.read(EVAL_NAME))
            target.execute("ATTACH DATABASE ? AS src", (str(Path(tmp) / DB_NAME),))
            try:
                if existing:
                    _delete_collection(target, collection_id)
                with target:
                    _insert(target, "collections", "id = ?", (collection_id,))
                    _insert(target, "papers", "collection_id = ?", (collection_id,))
                    for table in ("edges", "projections", "answers"):
                        _insert(target, table, "collection_id = ?", (collection_id,))
                    _insert(target, "chunks", "1=1", ())
                    _insert(target, "embeddings", "1=1", ())
                    _insert(target, "llm_cache", "1=1", ())
            finally:
                target.execute("DETACH DATABASE src")
        if target.execute("PRAGMA foreign_key_check").fetchall():
            raise SnapshotError("the import left rows that reference missing parents")
        imported = _counts(target)
    finally:
        target.close()
    return {"collection_id": collection_id, "imported": imported, "manifest": manifest}


def _delete_collection(conn: sqlite3.Connection, collection_id: str) -> None:
    with conn:
        conn.execute("DELETE FROM embeddings WHERE owner_type='chunk' AND owner_id IN "
                     "(SELECT c.id FROM chunks c JOIN papers p ON p.id=c.paper_id WHERE p.collection_id=?)", (collection_id,))
        conn.execute("DELETE FROM embeddings WHERE owner_type='paper' AND owner_id IN "
                     "(SELECT id FROM papers WHERE collection_id=?)", (collection_id,))
        conn.execute("DELETE FROM chunks WHERE paper_id IN (SELECT id FROM papers WHERE collection_id=?)", (collection_id,))
        for table in ("edges", "projections", "answers", "jobs", "papers"):
            conn.execute(f"DELETE FROM {table} WHERE collection_id=?", (collection_id,))
        conn.execute("DELETE FROM collections WHERE id=?", (collection_id,))
