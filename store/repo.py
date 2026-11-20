
from __future__ import annotations

import hashlib
import json
import struct
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from ingest.chunk import Chunk
from store.db import transaction

_FLOAT32 = "<f"


def new_id() -> str:
    return uuid.uuid4().hex


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _stable_id(*parts: str) -> str:
    return hashlib.sha256(":".join(parts).encode("utf-8")).hexdigest()[:32]


def _row_dict(row: Any) -> dict[str, Any] | None:
    if row is None:
        return None
    return {key: row[key] for key in row.keys()}


@dataclass(frozen=True)
class PaperRow:
    arxiv_id: str | None
    openalex_id: str | None
    doi: str | None
    title: str
    abstract: str | None
    authors: Sequence[str]
    year: int | None
    venue: str | None
    pdf_url: str | None
    citation_count: int | None
    source: str
    parse_status: str
    parse_note: str | None
    has_fulltext: int
    identity: str | None = None


def create_collection(
    conn: Any, *, mode: str, query: str | None, embedding_model: str | None
) -> str:
    collection_id = new_id()
    now = _now()
    with transaction(conn):
        conn.execute(
            """
            INSERT INTO collections
                (id, mode, query, status, embedding_model, n_papers, n_chunks,
                 created_at, updated_at)
            VALUES (?, ?, ?, 'building', ?, 0, 0, ?, ?)
            """,
            (collection_id, mode, query, embedding_model, now, now),
        )
    return collection_id


def delete_collection(conn: Any, collection_id: str) -> dict[str, int]:
    row = conn.execute("SELECT status FROM collections WHERE id=?", (collection_id,)).fetchone()
    if row is None:
        raise KeyError(collection_id)
    if row["status"] == "building":
        raise ValueError("a collection that is still building cannot be deleted")
    paper_ids = "SELECT id FROM papers WHERE collection_id=?"
    chunk_ids = f"SELECT id FROM chunks WHERE paper_id IN ({paper_ids})"
    removed: dict[str, int] = {}
    with transaction(conn):
        for name, sql in (
            ("embeddings", f"DELETE FROM embeddings WHERE (owner_type='chunk' AND owner_id IN ({chunk_ids})) "
                           f"OR (owner_type='paper' AND owner_id IN ({paper_ids}))"),
            ("chunks", f"DELETE FROM chunks WHERE paper_id IN ({paper_ids})"),
            ("edges", "DELETE FROM edges WHERE collection_id=?"),
            ("projections", "DELETE FROM projections WHERE collection_id=?"),
            ("answers", "DELETE FROM answers WHERE collection_id=?"),
            ("jobs", "DELETE FROM jobs WHERE collection_id=?"),
            ("papers", "DELETE FROM papers WHERE collection_id=?"),
            ("collections", "DELETE FROM collections WHERE id=?"),
        ):
            removed[name] = conn.execute(sql, (collection_id,) * sql.count("?")).rowcount
    return removed


def set_collection_status(
    conn: Any, collection_id: str, status: str, *, error: str | None = None
) -> None:
    with transaction(conn):
        conn.execute(
            "UPDATE collections SET status=?, error=?, updated_at=? WHERE id=?",
            (status, error, _now(), collection_id),
        )


STALE_BUILD_SECONDS = 300


def fail_interrupted_builds(conn: Any, *, stale_seconds: int = STALE_BUILD_SECONDS) -> tuple[int, int]:
    from datetime import timedelta

    reason = "interrupted: no progress was recorded for %d s (the build's process is gone)" % stale_seconds
    now = datetime.now(timezone.utc)
    cutoff = (now - timedelta(seconds=stale_seconds)).isoformat(timespec="seconds")
    stamp = now.isoformat(timespec="seconds")
    with transaction(conn):
        jobs = conn.execute(
            "UPDATE jobs SET status='failed', error=?, finished_at=?, updated_at=? "
            "WHERE status IN ('queued', 'running') AND (updated_at IS NULL OR updated_at <= ?)",
            (reason, stamp, stamp, cutoff),
        ).rowcount
        collections = conn.execute(
            "UPDATE collections SET status='failed', error=?, updated_at=? "
            "WHERE status='building' AND NOT EXISTS ("
            "  SELECT 1 FROM jobs j WHERE j.collection_id = collections.id AND j.status IN ('queued', 'running'))",
            (reason, stamp),
        ).rowcount
    return collections, jobs


def set_collection_counts(conn: Any, collection_id: str, *, n_papers: int, n_chunks: int) -> None:
    with transaction(conn):
        conn.execute(
            "UPDATE collections SET n_papers=?, n_chunks=?, updated_at=? WHERE id=?",
            (n_papers, n_chunks, _now(), collection_id),
        )


def set_collection_embedding_model(
    conn: Any, collection_id: str, embedding_model: str
) -> None:
    with transaction(conn):
        conn.execute(
            "UPDATE collections SET embedding_model=?, updated_at=? WHERE id=?",
            (embedding_model, _now(), collection_id),
        )


def set_collection_degraded(conn: Any, collection_id: str, reasons: Sequence[str]) -> None:
    with transaction(conn):
        conn.execute(
            "UPDATE collections SET degraded=?, updated_at=? WHERE id=?",
            (json.dumps(sorted(set(reasons))), _now(), collection_id),
        )


def _with_degraded(row: Any) -> dict[str, Any] | None:
    data = _row_dict(row)
    if data is None:
        return None
    raw = data.get("degraded")
    try:
        parsed = json.loads(raw) if raw else []
    except ValueError:
        parsed = []
    data["degraded"] = parsed if isinstance(parsed, list) else []
    return data


def find_ready_collection(conn: Any, *, mode: str, query: str | None) -> str | None:
    row = conn.execute(
        """
        SELECT id FROM collections
        WHERE mode=? AND query IS ? AND status='ready'
        ORDER BY created_at DESC, rowid DESC
        LIMIT 1
        """,
        (mode, query),
    ).fetchone()
    return str(row["id"]) if row else None


def find_building_collection(conn: Any, *, mode: str, query: str | None) -> str | None:
    row = conn.execute(
        """
        SELECT id FROM collections
        WHERE mode=? AND query IS ? AND status='building'
        ORDER BY created_at DESC, rowid DESC
        LIMIT 1
        """,
        (mode, query),
    ).fetchone()
    return str(row["id"]) if row else None


def cancel_build(conn: Any, collection_id: str) -> int:
    stamp = _now()
    with transaction(conn):
        jobs = conn.execute(
            "UPDATE jobs SET status='failed', error='cancelled', finished_at=?, updated_at=? "
            "WHERE collection_id=? AND status IN ('queued', 'running')",
            (stamp, stamp, collection_id),
        ).rowcount
        if jobs:
            conn.execute(
                "UPDATE collections SET status='failed', error='Cancelled.', updated_at=? "
                "WHERE id=? AND status='building'",
                (stamp, collection_id),
            )
    return jobs


def _with_live_job(conn: Any, data: dict[str, Any]) -> dict[str, Any]:
    job = latest_job_for_collection(conn, data["id"]) if data.get("status") == "building" else None
    data["job"] = (
        {k: job[k] for k in ("id", "status", "progress", "message")} if job else None
    )
    return data


def list_collections(conn: Any, *, limit: int = 50) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT id, mode, query, status, error, degraded, embedding_model, n_papers,
               n_chunks, created_at, updated_at
        FROM collections
        ORDER BY created_at DESC, rowid DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [_with_live_job(conn, dict(_with_degraded(row) or {})) for row in rows]


def get_collection(conn: Any, collection_id: str) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT id, mode, query, status, error, degraded, embedding_model, n_papers,
               n_chunks, created_at, updated_at
        FROM collections WHERE id=?
        """,
        (collection_id,),
    ).fetchone()
    data = _with_degraded(row)
    return _with_live_job(conn, data) if data else None


def paper_id_for(
    collection_id: str,
    *,
    arxiv_id: str | None,
    openalex_id: str | None,
    title: str,
    identity: str | None = None,
) -> str:
    return _stable_id(collection_id, arxiv_id or openalex_id or identity or title)


def _existing_id(
    conn: Any,
    collection_id: str,
    *,
    arxiv_id: str | None,
    openalex_id: str | None,
    doi: str | None,
) -> str | None:
    for column, value in (
        ("arxiv_id", arxiv_id),
        ("openalex_id", openalex_id),
        ("doi", doi),
    ):
        if not value:
            continue
        row = conn.execute(
            f"SELECT id FROM papers WHERE collection_id=? AND {column}=? LIMIT 1",
            (collection_id, value),
        ).fetchone()
        if row is not None:
            return str(row["id"])
    return None


def upsert_paper(conn: Any, *, collection_id: str, paper: PaperRow) -> str:
    paper_id = _existing_id(
        conn,
        collection_id,
        arxiv_id=paper.arxiv_id,
        openalex_id=paper.openalex_id,
        doi=paper.doi,
    ) or paper_id_for(
        collection_id,
        arxiv_id=paper.arxiv_id,
        openalex_id=paper.openalex_id,
        title=paper.title,
        identity=paper.identity,
    )
    with transaction(conn):
        conn.execute(
            """
            INSERT INTO papers
                (id, collection_id, arxiv_id, openalex_id, doi, title, abstract,
                 authors, year, venue, pdf_url, citation_count, source,
                 parse_status, parse_note, has_fulltext, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                arxiv_id=excluded.arxiv_id,
                openalex_id=excluded.openalex_id,
                doi=excluded.doi,
                title=excluded.title,
                abstract=excluded.abstract,
                authors=excluded.authors,
                year=excluded.year,
                venue=excluded.venue,
                pdf_url=excluded.pdf_url,
                citation_count=excluded.citation_count,
                source=excluded.source,
                parse_status=excluded.parse_status,
                parse_note=excluded.parse_note,
                has_fulltext=excluded.has_fulltext
            """,
            (
                paper_id,
                collection_id,
                paper.arxiv_id,
                paper.openalex_id,
                paper.doi,
                paper.title,
                paper.abstract,
                json.dumps(list(paper.authors)),
                paper.year,
                paper.venue,
                paper.pdf_url,
                paper.citation_count,
                paper.source,
                paper.parse_status,
                paper.parse_note,
                paper.has_fulltext,
                _now(),
            ),
        )
    return paper_id


def count_papers(conn: Any, collection_id: str) -> int:
    row = conn.execute(
        "SELECT count(*) FROM papers WHERE collection_id=?", (collection_id,)
    ).fetchone()
    return int(row[0])


def replace_chunks(conn: Any, paper_id: str, chunks: Sequence[Chunk]) -> int:
    with transaction(conn):
        conn.execute(
            """
            DELETE FROM embeddings
            WHERE owner_type='chunk'
              AND owner_id IN (SELECT id FROM chunks WHERE paper_id=?)
            """,
            (paper_id,),
        )
        conn.execute("DELETE FROM chunks WHERE paper_id=?", (paper_id,))
        conn.executemany(
            "INSERT INTO chunks (id, paper_id, seq, text, n_tokens) VALUES (?, ?, ?, ?, ?)",
            [
                (_stable_id(paper_id, str(chunk.seq)), paper_id, chunk.seq, chunk.text, chunk.n_tokens)
                for chunk in chunks
            ],
        )
    return len(chunks)


def count_chunks(conn: Any, collection_id: str) -> int:
    row = conn.execute(
        """
        SELECT count(*) FROM chunks c
        JOIN papers p ON p.id = c.paper_id
        WHERE p.collection_id=?
        """,
        (collection_id,),
    ).fetchone()
    return int(row[0])


def chunk_ids_for_paper(conn: Any, paper_id: str) -> list[tuple[str, int]]:
    rows = conn.execute(
        "SELECT id, seq FROM chunks WHERE paper_id=? ORDER BY seq", (paper_id,)
    ).fetchall()
    return [(row["id"], row["seq"]) for row in rows]


def save_embedding(
    conn: Any, *, owner_type: str, owner_id: str, vec: Sequence[float], model: str
) -> None:
    values = list(vec)
    blob = struct.pack(f"<{len(values)}f", *values)
    with transaction(conn):
        conn.execute(
            """
            INSERT INTO embeddings (id, owner_type, owner_id, dim, vec, model)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(owner_type, owner_id) DO UPDATE SET
                dim=excluded.dim, vec=excluded.vec, model=excluded.model
            """,
            (_stable_id(owner_type, owner_id), owner_type, owner_id, len(values), blob, model),
        )


def unpack_vector(blob: bytes, dim: int) -> list[float]:
    expected = dim * struct.calcsize(_FLOAT32)
    if len(blob) != expected:
        raise ValueError(f"embedding declares dim={dim} but blob is {len(blob)} bytes")
    return list(struct.unpack(f"<{dim}f", blob))


def load_embedding(conn: Any, owner_type: str, owner_id: str) -> list[float] | None:
    row = conn.execute(
        "SELECT dim, vec FROM embeddings WHERE owner_type=? AND owner_id=?",
        (owner_type, owner_id),
    ).fetchone()
    if row is None:
        return None
    return unpack_vector(bytes(row["vec"]), int(row["dim"]))


def upsert_edge(
    conn: Any,
    *,
    collection_id: str,
    src_paper_id: str,
    dst_paper_id: str | None,
    dst_openalex_id: str | None,
    dst_title: str | None,
    dst_doi: str | None,
) -> None:
    resolved = 1 if dst_paper_id else 0
    edge_id = _stable_id(
        collection_id,
        src_paper_id,
        dst_openalex_id or dst_doi or dst_title or "",
    )
    with transaction(conn):
        conn.execute(
            """
            INSERT INTO edges
                (id, collection_id, src_paper_id, dst_paper_id, dst_openalex_id,
                 dst_title, dst_doi, resolved)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                dst_paper_id=excluded.dst_paper_id,
                dst_title=COALESCE(excluded.dst_title, edges.dst_title),
                dst_doi=COALESCE(excluded.dst_doi, edges.dst_doi),
                resolved=excluded.resolved
            """,
            (
                edge_id,
                collection_id,
                src_paper_id,
                dst_paper_id,
                dst_openalex_id,
                dst_title,
                dst_doi,
                resolved,
            ),
        )


def count_edges(conn: Any, collection_id: str) -> int:
    row = conn.execute(
        "SELECT count(*) FROM edges WHERE collection_id=?", (collection_id,)
    ).fetchone()
    return int(row[0])


def paper_ids_by_openalex(conn: Any, collection_id: str) -> dict[str, str]:
    rows = conn.execute(
        "SELECT id, openalex_id FROM papers WHERE collection_id=? AND openalex_id IS NOT NULL",
        (collection_id,),
    ).fetchall()
    return {row["openalex_id"]: row["id"] for row in rows}


def create_job(conn: Any, *, collection_id: str | None, kind: str) -> str:
    job_id = new_id()
    with transaction(conn):
        conn.execute(
            """
            INSERT INTO jobs (id, collection_id, kind, status, progress, created_at, updated_at)
            VALUES (?, ?, ?, 'queued', 0, ?, ?)
            """,
            (job_id, collection_id, kind, _now(), _now()),
        )
    return job_id


def update_job(
    conn: Any,
    job_id: str,
    *,
    status: str,
    progress: float | None = None,
    message: str | None = None,
    error: str | None = None,
) -> None:
    finished = _now() if status in {"done", "failed"} else None
    with transaction(conn):
        conn.execute(
            """
            UPDATE jobs SET
                status=:status,
                progress=COALESCE(:progress, progress),
                message=COALESCE(:message, message),
                -- a job that was marked failed and later finishes must not keep the old error
                error=CASE WHEN :error IS NOT NULL THEN :error
                           WHEN :status='done' THEN NULL
                           ELSE error END,
                finished_at=COALESCE(:finished, finished_at),
                updated_at=:now
            WHERE id=:id
              -- a reaped (failed) job is not resurrected by a late progress write
              AND NOT (status='failed' AND :status IN ('running', 'queued'))
            """,
            {
                "status": status,
                "progress": progress,
                "message": message,
                "error": error,
                "finished": finished,
                "now": _now(),
                "id": job_id,
            },
        )


def get_job(conn: Any, job_id: str) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT id, collection_id, kind, status, progress, message, error,
               created_at, finished_at
        FROM jobs WHERE id=?
        """,
        (job_id,),
    ).fetchone()
    return _row_dict(row)


def latest_job_for_collection(conn: Any, collection_id: str) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT id, collection_id, kind, status, progress, message, error,
               created_at, finished_at
        FROM jobs WHERE collection_id=?
        ORDER BY rowid DESC
        LIMIT 1
        """,
        (collection_id,),
    ).fetchone()
    return _row_dict(row)


def save_answer(
    conn: Any,
    *,
    collection_id: str,
    question: str,
    answer: str | None,
    citations: Any,
    refused: bool,
    model: str | None,
    latency_ms: int,
    retrieval_snapshot: Any,
) -> str:
    answer_id = new_id()
    with transaction(conn):
        conn.execute(
            """
            INSERT INTO answers
                (id, collection_id, question, answer, citations, refused, model,
                 latency_ms, retrieval_snapshot, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                answer_id,
                collection_id,
                question,
                answer,
                json.dumps(citations),
                1 if refused else 0,
                model,
                latency_ms,
                json.dumps(retrieval_snapshot),
                _now(),
            ),
        )
    return answer_id
