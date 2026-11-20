
from __future__ import annotations

import hashlib
import re
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ingest.cache import Cache
from ingest.http import FetchError, HttpClient, RateLimited
from ingest.pdf import parse_isolated
from services.collection import (
    BuildCancelled,
    CollectionBuildError,
    _embed_and_store,
    _OpenAlexClient,
    _store_edges,
    abstract_paper_text,
    raise_if_cancelled,
)
from store import repo
from store.repo import PaperRow

MAX_FILES = 25
MAX_FILE_BYTES = 30 * 1024 * 1024
MAX_TOTAL_UPLOAD_BYTES = 200 * 1024 * 1024
UPLOAD_SCHEME = "upload://"
_SHA = re.compile(r"^[0-9a-f]{64}$")


class UploadRejected(ValueError):
    pass


@dataclass(frozen=True)
class StoredUpload:
    sha256: str
    filename: str
    size: int


def upload_dir(data_dir: str | Path, collection_id: str) -> Path:
    return Path(data_dir) / "uploads" / collection_id


def upload_path(data_dir: str | Path, collection_id: str, sha256: str) -> Path | None:
    if not _SHA.match(sha256):
        return None
    return upload_dir(data_dir, collection_id) / f"{sha256}.pdf"


def validate_uploads(files: Sequence[tuple[str, bytes]]) -> list[StoredUpload]:
    if not files:
        raise UploadRejected("no files were uploaded")
    if len(files) > MAX_FILES:
        raise UploadRejected(f"at most {MAX_FILES} PDFs per collection; got {len(files)}")

    stored: dict[str, StoredUpload] = {}
    for filename, data in files:
        label = filename or "(unnamed file)"
        if len(data) > MAX_FILE_BYTES:
            raise UploadRejected(f"{label} is larger than {MAX_FILE_BYTES // (1024 * 1024)} MB")
        if not data.startswith(b"%PDF-"):
            raise UploadRejected(f"{label} is not a PDF (missing the %PDF- header)")
        sha = hashlib.sha256(data).hexdigest()
        stored.setdefault(sha, StoredUpload(sha256=sha, filename=filename, size=len(data)))
    return list(stored.values())


def store_uploads(
    data_dir: str | Path, collection_id: str, files: Sequence[tuple[str, bytes]]
) -> None:
    folder = upload_dir(data_dir, collection_id)
    folder.mkdir(parents=True, exist_ok=True)
    for _, data in files:
        (folder / f"{hashlib.sha256(data).hexdigest()}.pdf").write_bytes(data)


def validate_and_store(
    data_dir: str | Path, collection_id: str, files: Sequence[tuple[str, bytes]]
) -> list[StoredUpload]:
    stored = validate_uploads(files)
    store_uploads(data_dir, collection_id, files)
    return stored


def remove_uploads(data_dir: str | Path, collection_id: str) -> int:
    folder = upload_dir(data_dir, collection_id)
    if not folder.is_dir():
        return 0
    count = sum(1 for item in folder.iterdir() if item.is_file())
    shutil.rmtree(folder, ignore_errors=True)
    return count


def _file_title(filename: str) -> str:
    stem = Path(filename).stem.replace("_", " ").replace("-", " ").strip()
    return stem or "Untitled upload"


def _same_title(a: str | None, b: str | None) -> bool:
    squash = lambda t: "".join(ch for ch in (t or "").lower() if ch.isalnum())
    return bool(squash(a)) and squash(a) == squash(b)


def _resolve_work(openalex: Any, meta: Any, title: str, degraded: list[str], state: dict) -> Any:
    if state["skip"]:
        return None
    try:
        if meta.arxiv_id:
            work = openalex.lookup_by_arxiv(meta.arxiv_id)
            if work is not None:
                return work
        if meta.doi:
            work = openalex.lookup_by_doi(meta.doi)
            if work is not None and _same_title(work.title, meta.title):
                return work
        if meta.title:
            return openalex.lookup_by_title(meta.title)
    except RateLimited:
        degraded.append("openalex_rate_limited")
        state["skip"] = True
    except FetchError:
        if "openalex_fetch_failed" not in degraded:
            degraded.append("openalex_fetch_failed")
    return None


def build_upload_collection(
    conn: Any,
    *,
    collection_id: str,
    job_id: str,
    uploads: Sequence[StoredUpload],
    settings: Any,
    openalex: Any | None = None,
    embedder: Any | None = None,
    client: HttpClient | None = None,
    log: Callable[..., None] = print,
    parse_timeout: float = 30.0,
) -> list[str]:
    raise_if_cancelled(conn, job_id, ("queued", "running"))
    degraded: list[str] = []
    owns_client = client is None
    owns_embedder = embedder is None

    if client is None:
        client = HttpClient(
            Cache(getattr(settings, "cache_dir", "./data/cache")),
            user_agent=getattr(settings, "http_user_agent", "litreview/0.1"),
            timeout=30.0,
            max_retries=3,
        )
    if openalex is None:
        openalex = _OpenAlexClient(client, getattr(settings, "contact_email", None))
    if embedder is None:
        from embed.specter2 import Specter2Embedder

        embedder = Specter2Embedder(model_dir=str(getattr(settings, "models_dir", "./models")))

    model_label = getattr(embedder, "model_name", None) or getattr(settings, "embedding_model", "m")
    repo.set_collection_embedding_model(conn, collection_id, model_label)
    repo.update_job(conn, job_id, status="running", progress=0.0, message="reading PDFs")

    folder = upload_dir(getattr(settings, "data_dir", "./data"), collection_id)
    pending_edges: list[tuple[str, str]] = []
    state = {"skip": False}

    try:
        for index, upload in enumerate(uploads, start=1):
            raise_if_cancelled(conn, job_id)
            data = (folder / f"{upload.sha256}.pdf").read_bytes()
            meta, extracted, parse_problem = parse_isolated(data, timeout=parse_timeout)
            work = _resolve_work(openalex, meta, meta.title or "", degraded, state)

            note: str | None = None
            if work is not None:
                title = work.title or meta.title or _file_title(upload.filename)
            elif meta.title:
                title = meta.title
            else:
                title = _file_title(upload.filename)
                note = "no title found in the PDF; showing the file name"

            if extracted.quality == "ok":
                body = extracted.text
            elif meta.abstract:
                body = abstract_paper_text(title, meta.abstract)
                note = (note + "; " if note else "") + (
                    f"text quality too low (alpha_ratio={extracted.alpha_ratio:.2f}); abstract only"
                )
            else:
                body = ""
                note = (note + "; " if note else "") + (
                    (parse_problem or "PDF could not be parsed") if extracted.quality == "failed"
                    else f"text quality too low (alpha_ratio={extracted.alpha_ratio:.2f}) and no abstract found"
                )

            raise_if_cancelled(conn, job_id)
            paper_id = repo.upsert_paper(
                conn,
                collection_id=collection_id,
                paper=PaperRow(
                    arxiv_id=meta.arxiv_id or None,
                    openalex_id=work.openalex_id if work else None,
                    doi=(work.doi if work and work.doi else None) or meta.doi,
                    title=title,
                    abstract=meta.abstract,
                    authors=meta.authors,
                    year=work.year if work else None,
                    venue=work.venue if work else None,
                    pdf_url=f"{UPLOAD_SCHEME}{upload.sha256}",
                    citation_count=work.cited_by_count if work else None,
                    source="upload",
                    parse_status=extracted.quality,
                    parse_note=note,
                    has_fulltext=1 if extracted.quality == "ok" else 0,
                    identity=upload.sha256,
                ),
            )
            _embed_and_store(
                conn,
                paper_id=paper_id,
                body=body,
                embedder=embedder,
                model_label=model_label,
                title=title,
                abstract=meta.abstract,
            )
            if work is not None:
                pending_edges.extend((paper_id, ref) for ref in work.referenced_works)

            repo.update_job(
                conn, job_id, status="running",
                progress=round(index / (len(uploads) + 1), 3),
                message=f"processed {index}/{len(uploads)}",
            )

        raise_if_cancelled(conn, job_id)
        _store_edges(
            conn, collection_id=collection_id, pending=pending_edges,
            openalex=openalex, degraded=degraded, log=log,
        )
        repo.set_collection_counts(
            conn,
            collection_id,
            n_papers=repo.count_papers(conn, collection_id),
            n_chunks=repo.count_chunks(conn, collection_id),
        )
        repo.set_collection_degraded(conn, collection_id, degraded)
        repo.set_collection_status(conn, collection_id, "ready")
        repo.update_job(conn, job_id, status="done", progress=1.0, message="collection ready")
    except BuildCancelled:
        raise
    except Exception as exc:
        job = repo.get_job(conn, job_id)
        if job is None or job["status"] != "running":
            raise BuildCancelled(job_id) from exc
        repo.set_collection_status(conn, collection_id, "failed", error=str(exc))
        repo.update_job(conn, job_id, status="failed", error=str(exc))
        raise CollectionBuildError(str(exc)) from exc
    finally:
        if owns_client:
            client.close()
        if owns_embedder:
            embedder.close()
    return degraded
