
from __future__ import annotations

import logging
import threading
from typing import Any

from services.collection import BuildCancelled, CollectionBuildError, build_collection
from store import repo
from store.db import connect

logger = logging.getLogger(__name__)

_BUILD_SLOT = threading.Semaphore(1)
WAIT_HEARTBEAT_SECONDS = 30.0


def _acquire_build_slot(conn: Any, job_id: str) -> bool:
    if _BUILD_SLOT.acquire(blocking=False):
        return True
    while True:
        repo.update_job(
            conn, job_id, status="queued", progress=0.0,
            message="waiting for another build to finish",
        )
        job = repo.get_job(conn, job_id)
        if job is None or job["status"] not in ("queued", "running"):
            return False
        if _BUILD_SLOT.acquire(timeout=WAIT_HEARTBEAT_SECONDS):
            job = repo.get_job(conn, job_id)
            if job is not None and job["status"] in ("queued", "running"):
                return True
            _BUILD_SLOT.release()
            return False


def prepare_topic_build(
    conn: Any, *, query: str, settings: Any, reuse: bool = False
) -> tuple[str, str]:
    collection_id = (
        repo.find_ready_collection(conn, mode="topic", query=query) if reuse else None
    )
    if collection_id is None:
        collection_id = repo.create_collection(
            conn,
            mode="topic",
            query=query,
            embedding_model=getattr(settings, "embedding_model", None),
        )
    else:
        repo.set_collection_status(conn, collection_id, "building")
    job_id = repo.create_job(conn, collection_id=collection_id, kind="build")
    return collection_id, job_id


def start_topic_build(
    settings: Any,
    *,
    collection_id: str,
    job_id: str,
    query: str,
    max_papers: int,
) -> threading.Thread:
    thread = threading.Thread(
        target=_run,
        args=(settings, collection_id, job_id, query, max_papers),
        name=f"build-{collection_id[:8]}",
        daemon=True,
    )
    thread.start()
    return thread


def prepare_upload_build(conn: Any, *, name: str, settings: Any) -> tuple[str, str]:
    collection_id = repo.create_collection(
        conn, mode="upload", query=name, embedding_model=getattr(settings, "embedding_model", None)
    )
    return collection_id, repo.create_job(conn, collection_id=collection_id, kind="build")


def start_upload_build(
    settings: Any, *, collection_id: str, job_id: str, uploads: list[Any]
) -> threading.Thread:
    thread = threading.Thread(
        target=_run_upload,
        args=(settings, collection_id, job_id, uploads),
        name=f"upload-{collection_id[:8]}",
        daemon=True,
    )
    thread.start()
    return thread


def _run_upload(settings: Any, collection_id: str, job_id: str, uploads: list[Any]) -> None:
    from services.upload import build_upload_collection

    conn = None
    try:
        conn = connect(settings.db_path)
        if not _acquire_build_slot(conn, job_id):
            logger.warning("upload collection %s was reaped or deleted while queued", collection_id)
            return
        try:
            build_upload_collection(
                conn, collection_id=collection_id, job_id=job_id, uploads=uploads, settings=settings
            )
        finally:
            _BUILD_SLOT.release()
    except BuildCancelled:
        logger.info("upload collection %s build cancelled", collection_id)
    except CollectionBuildError:
        logger.warning("upload collection %s failed to build", collection_id)
    except Exception as exc:
        logger.exception("upload collection %s crashed", collection_id)
        if conn is not None:
            repo.set_collection_status(conn, collection_id, "failed", error=str(exc))
            repo.update_job(conn, job_id, status="failed", error=str(exc))
    finally:
        if conn is not None:
            conn.close()


def reconcile_interrupted_builds(conn: Any) -> tuple[int, int]:
    collections, jobs = repo.fail_interrupted_builds(conn)
    if collections or jobs:
        logger.warning(
            "failed %d interrupted collection(s) and %d job(s) left by a previous process",
            collections,
            jobs,
        )
    return collections, jobs


def _run(
    settings: Any,
    collection_id: str,
    job_id: str,
    query: str,
    max_papers: int,
) -> None:
    conn = None
    try:
        conn = connect(settings.db_path)
        if not _acquire_build_slot(conn, job_id):
            logger.warning("collection %s was reaped or deleted while queued", collection_id)
            return
        try:
            build_collection(
                conn,
                query=query,
                limit=max_papers,
                settings=settings,
                collection_id=collection_id,
                job_id=job_id,
            )
        finally:
            _BUILD_SLOT.release()
    except BuildCancelled:
        logger.info("collection %s build cancelled", collection_id)
    except CollectionBuildError:
        logger.warning("collection %s failed to build", collection_id)
    except Exception as exc:
        logger.exception("collection %s crashed before build_collection could report", collection_id)
        if conn is not None:
            repo.set_collection_status(conn, collection_id, "failed", error=str(exc))
            repo.update_job(conn, job_id, status="failed", error=str(exc))
    finally:
        if conn is not None:
            conn.close()
