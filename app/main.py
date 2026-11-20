
from __future__ import annotations

import importlib.util
import logging
import sqlite3
import threading
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

import httpx

import rag_core
from evaluation.harness import load_report
from rag_core import MODES
from services.answer import ask as answer_question
from services.build_jobs import (
    prepare_topic_build,
    prepare_upload_build,
    reconcile_interrupted_builds,
    start_topic_build,
    start_upload_build,
)
from services.contradiction import (
    MAX_PAIRS,
    MAX_PAPERS as CONTRADICTION_MAX_PAPERS,
    ContradictionError,
    compare_papers,
)
from services.graph import build_graph
from services.labels import central_ids, cluster_labels
from services.llm import LLMClient, LLMUnavailable
from services.map import MapDependencyMissing, MapUnavailable, build_map
from services.query_embedder import QueryEmbedder
from services.related_work import MAX_PAPERS as RELATED_MAX_PAPERS
from services.related_work import RelatedWorkError, draft_related_work
from services.summarize import PaperNotFound, paper_summary
from services.upload import (
    MAX_FILE_BYTES,
    MAX_FILES,
    MAX_TOTAL_UPLOAD_BYTES,
    UPLOAD_SCHEME,
    UploadRejected,
    remove_uploads,
    store_uploads,
    upload_path,
    validate_uploads,
)
from services.retrieval import SORTS, evict_collection, paper_detail, paper_details, rank_papers
from store import repo
from store.db import connect, init_db

from .config import Settings
from .errors import ApiError, install_error_handlers

API_PREFIX = "/api"
DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 200
DEFAULT_BUILD_PAPERS = 100
MIN_BUILD_PAPERS = 10
MAX_BUILD_PAPERS = 200
COLLECTION_LIST_LIMIT = 50
DEFAULT_ASK_K = 8
MAX_ASK_K = 20
UPLOAD_OVERHEAD_BYTES = 1024 * 1024
WEB_DIST = Path(__file__).resolve().parent.parent / "web" / "dist"


def _generation_status(settings: Settings) -> tuple[bool, str | None]:
    if importlib.util.find_spec("services.answer") is None:
        return False, "not_implemented"
    if settings.llm_api_key is None:
        return False, "no_api_key"
    return True, None


class CollectionCreate(BaseModel):
    query: str
    max_papers: int = Field(DEFAULT_BUILD_PAPERS, ge=MIN_BUILD_PAPERS, le=MAX_BUILD_PAPERS)
    mode: str = "topic"


class AskBody(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    k: int = Field(DEFAULT_ASK_K, ge=1, le=MAX_ASK_K)
    mode: str = "hybrid"

    @field_validator("question")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if len(stripped) < 3:
            raise ValueError("question must contain at least 3 non-space characters")
        return stripped


class ContradictionBody(BaseModel):
    paper_ids: list[str] | None = None
    cluster_id: int | None = None
    max_pairs: int = Field(5, ge=1, le=MAX_PAIRS)


class RelatedWorkBody(BaseModel):
    paper_ids: list[str] | None = None
    cluster_id: int | None = None


def _read_capped(upload: UploadFile, request_remaining: int = MAX_TOTAL_UPLOAD_BYTES) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        block = upload.file.read(1024 * 1024)
        if not block:
            break
        total += len(block)
        if total > MAX_FILE_BYTES:
            raise ApiError(
                "invalid_upload",
                f"{upload.filename or 'a file'} is larger than {MAX_FILE_BYTES // (1024 * 1024)} MB",
                413,
            )
        if total > request_remaining:
            raise ApiError(
                "invalid_upload",
                f"the files together are larger than {MAX_TOTAL_UPLOAD_BYTES // (1024 * 1024)} MB",
                413,
            )
        chunks.append(block)
    return b"".join(chunks)


def _generation_error(exc: LLMUnavailable) -> ApiError:
    return ApiError(
        "generation_unavailable",
        exc.message,
        exc.status,
        {"reason": exc.code},
    )


def _require_ready(conn: sqlite3.Connection, collection_id: str) -> sqlite3.Row:
    row = conn.execute(
        "SELECT id, status, error FROM collections WHERE id=?", (collection_id,)
    ).fetchone()
    if row is None:
        raise ApiError(
            "not_found",
            f"collection {collection_id} does not exist",
            404,
            {"collection_id": collection_id},
        )
    if row["status"] != "ready":
        raise ApiError(
            "not_ready",
            f"collection is {row['status']}, not ready",
            409,
            {"status": row["status"], "error": row["error"]},
        )
    return row


def create_app(
    settings: Settings | None = None,
    *,
    warm_query_embedder: bool = True,
    static_dir: str | Path | None = None,
    llm_transport: httpx.BaseTransport | None = None,
) -> FastAPI:
    resolved = settings if settings is not None else Settings.from_env()
    query_embedder = QueryEmbedder(model_dir=str(resolved.models_dir))
    llm = LLMClient(
        api_key=resolved.llm_api_key,
        model=resolved.llm_model,
        db_path=resolved.db_path,
        provider=resolved.llm_provider,
        transport=llm_transport,
    )
    if resolved.llm_config_note:
        logging.getLogger(__name__).warning("LLM configuration: %s", resolved.llm_config_note)
    build_gate = threading.Lock()
    generation_available, generation_reason = _generation_status(resolved)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        init_db(resolved.db_path)
        try:
            conn = connect(resolved.db_path)
            try:
                reconcile_interrupted_builds(conn)
            finally:
                conn.close()
        except sqlite3.Error:
            logging.getLogger(__name__).warning(
                "could not reconcile interrupted builds at startup", exc_info=True
            )
        if warm_query_embedder and resolved.embedding_provider == "specter2":
            threading.Thread(
                target=query_embedder.warm, name="query-embedder-warm", daemon=True
            ).start()
        yield

    app = FastAPI(
        title="Smart Literature Review Assistant",
        version=rag_core.__version__,
        description="Ranked papers, a semantic map, a citation graph and cited answers.",
        lifespan=lifespan,
    )
    app.state.settings = resolved
    app.state.query_embedder = query_embedder
    app.state.llm = llm
    app.state.map_projector = None
    app.state.map_clusterer = None
    app.state.build_starter = start_topic_build
    app.state.upload_starter = start_upload_build

    install_error_handlers(app)

    @app.middleware("http")
    async def reject_oversized_upload(request: Request, call_next):
        if request.method == "POST" and request.url.path == f"{API_PREFIX}/collections/upload":
            declared = request.headers.get("content-length", "")
            if declared.isdigit() and int(declared) > MAX_TOTAL_UPLOAD_BYTES + UPLOAD_OVERHEAD_BYTES:
                return JSONResponse(
                    status_code=413,
                    content={
                        "error": {
                            "code": "invalid_upload",
                            "message": "the files together are larger than "
                            f"{MAX_TOTAL_UPLOAD_BYTES // (1024 * 1024)} MB",
                            "detail": None,
                        }
                    },
                )
        return await call_next(request)

    def get_conn() -> Iterator[sqlite3.Connection]:
        conn = connect(resolved.db_path)
        try:
            yield conn
        finally:
            conn.close()

    @app.get(f"{API_PREFIX}/health")
    def health() -> dict[str, object]:
        return {
            "status": "ok",
            "version": rag_core.__version__,
            "embedding_model": resolved.embedding_model,
            "generation_available": generation_available,
            "generation_reason": generation_reason,
            "generation_configured": resolved.llm_api_key is not None,
            "generation_provider": resolved.llm_provider,
            "generation_model": resolved.llm_model,
            "generation_config_note": resolved.llm_config_note,
            "generation_last_error": llm.last_error,
            "refusal_tau": resolved.refusal_tau,
            "refusal_tau_mode": resolved.refusal_tau_mode,
        }

    @app.post(f"{API_PREFIX}/collections", status_code=202)
    def create_collection_endpoint(
        request: Request,
        body: CollectionCreate,
        conn: sqlite3.Connection = Depends(get_conn),
    ) -> dict[str, object]:
        query = body.query.strip()
        if not query:
            raise ApiError(
                "invalid_request", "query must not be blank", 422, {"field": "query"}
            )
        if body.mode != "topic":
            raise ApiError(
                "invalid_request",
                "this endpoint builds topic collections; upload PDFs with "
                "POST /api/collections/upload",
                422,
                {"supported": ["topic"], "received": body.mode},
            )

        with build_gate:
            repo.fail_interrupted_builds(conn)
            building = repo.find_building_collection(conn, mode="topic", query=query)
            if building is not None:
                job = repo.latest_job_for_collection(conn, building)
                raise ApiError(
                    "build_in_progress",
                    f"a build for {query!r} is already running",
                    409,
                    {
                        "collection_id": building,
                        "job_id": job["id"] if job else None,
                    },
                )
            collection_id, job_id = prepare_topic_build(
                conn, query=query, settings=resolved
            )
        request.app.state.build_starter(
            resolved,
            collection_id=collection_id,
            job_id=job_id,
            query=query,
            max_papers=body.max_papers,
        )
        return {
            "collection_id": collection_id,
            "job_id": job_id,
            "status": "building",
        }

    @app.post(f"{API_PREFIX}/collections/upload", status_code=202)
    def upload_collection_endpoint(
        request: Request,
        files: list[UploadFile] = File(...),
        name: str = Form(""),
        conn: sqlite3.Connection = Depends(get_conn),
    ) -> dict[str, object]:
        if len(files) > MAX_FILES:
            raise ApiError(
                "invalid_upload", f"at most {MAX_FILES} PDFs per collection; got {len(files)}", 422
            )
        blobs: list[tuple[str, bytes]] = []
        total = 0
        for upload in files:
            data = _read_capped(upload, MAX_TOTAL_UPLOAD_BYTES - total)
            total += len(data)
            blobs.append((upload.filename or "", data))
        try:
            stored = validate_uploads(blobs)
        except UploadRejected as exc:
            raise ApiError("invalid_upload", str(exc), 422) from exc
        label = name.strip() or f"Uploaded PDFs ({len(blobs)})"
        with build_gate:
            collection_id, job_id = prepare_upload_build(conn, name=label, settings=resolved)
        try:
            store_uploads(resolved.data_dir, collection_id, blobs)
        except OSError as exc:
            repo.set_collection_status(conn, collection_id, "failed", error=str(exc))
            repo.update_job(conn, job_id, status="failed", error=str(exc))
            raise ApiError("invalid_upload", f"could not store the files: {exc}", 500) from exc
        request.app.state.upload_starter(
            resolved, collection_id=collection_id, job_id=job_id, uploads=stored
        )
        return {"collection_id": collection_id, "job_id": job_id, "status": "building"}

    @app.get(f"{API_PREFIX}/papers/{{paper_id}}/pdf")
    def uploaded_pdf(
        paper_id: str, conn: sqlite3.Connection = Depends(get_conn)
    ) -> FileResponse:
        row = conn.execute(
            "SELECT collection_id, source, pdf_url, title FROM papers WHERE id=?", (paper_id,)
        ).fetchone()
        if row is None or row["source"] != "upload" or not (row["pdf_url"] or "").startswith(UPLOAD_SCHEME):
            raise ApiError(
                "not_found", "no uploaded file is stored for this paper", 404, {"paper_id": paper_id}
            )
        path = upload_path(
            resolved.data_dir, row["collection_id"], row["pdf_url"][len(UPLOAD_SCHEME):]
        )
        if path is None or not path.is_file():
            raise ApiError(
                "not_found", "the uploaded file is no longer on disk", 404, {"paper_id": paper_id}
            )
        return FileResponse(path, media_type="application/pdf")

    @app.get(f"{API_PREFIX}/collections")
    def list_collections_endpoint(
        limit: int = Query(COLLECTION_LIST_LIMIT, ge=1, le=MAX_PAGE_SIZE),
        conn: sqlite3.Connection = Depends(get_conn),
    ) -> dict[str, object]:
        return {"collections": repo.list_collections(conn, limit=limit)}

    @app.get(f"{API_PREFIX}/collections/{{collection_id}}")
    def get_collection_endpoint(
        collection_id: str, conn: sqlite3.Connection = Depends(get_conn)
    ) -> dict[str, object]:
        row = repo.get_collection(conn, collection_id)
        if row is None:
            raise ApiError(
                "not_found",
                f"collection {collection_id} does not exist",
                404,
                {"collection_id": collection_id},
            )
        return row

    @app.delete(f"{API_PREFIX}/collections/{{collection_id}}")
    def delete_collection_endpoint(
        collection_id: str, conn: sqlite3.Connection = Depends(get_conn)
    ) -> dict[str, object]:
        with build_gate:
            repo.fail_interrupted_builds(conn)
            try:
                removed = repo.delete_collection(conn, collection_id)
            except KeyError:
                raise ApiError(
                    "not_found",
                    f"collection {collection_id} does not exist",
                    404,
                    {"collection_id": collection_id},
                ) from None
            except ValueError:
                raise ApiError(
                    "build_in_progress",
                    "a collection that is still building cannot be deleted",
                    409,
                    {"collection_id": collection_id},
                ) from None
        files = remove_uploads(resolved.data_dir, collection_id)
        evict_collection(collection_id)
        return {"collection_id": collection_id, "deleted": {**removed, "uploaded_files": files}}

    @app.post(f"{API_PREFIX}/collections/{{collection_id}}/cancel", status_code=202)
    def cancel_collection_build_endpoint(
        collection_id: str, conn: sqlite3.Connection = Depends(get_conn)
    ) -> dict[str, object]:
        with build_gate:
            repo.fail_interrupted_builds(conn)
            row = conn.execute("SELECT status FROM collections WHERE id=?", (collection_id,)).fetchone()
            if row is None:
                raise ApiError(
                    "not_found",
                    f"collection {collection_id} does not exist",
                    404,
                    {"collection_id": collection_id},
                )
            if row["status"] != "building" or repo.cancel_build(conn, collection_id) == 0:
                raise ApiError(
                    "not_building",
                    "this collection is not building, so there is nothing to cancel",
                    409,
                    {"collection_id": collection_id, "status": row["status"]},
                )
        return {"collection_id": collection_id, "status": "cancelling"}

    @app.get(f"{API_PREFIX}/jobs/{{job_id}}")
    def get_job_endpoint(
        job_id: str, conn: sqlite3.Connection = Depends(get_conn)
    ) -> dict[str, object]:
        job = repo.get_job(conn, job_id)
        if job is None:
            raise ApiError(
                "not_found", f"job {job_id} does not exist", 404, {"job_id": job_id}
            )
        return job

    @app.get(f"{API_PREFIX}/collections/{{collection_id}}/papers")
    def list_papers(
        request: Request,
        collection_id: str,
        q: str | None = Query(None, description="query text; empty orders by sort"),
        sort: str = Query("relevance", description="relevance | influence | year"),
        mode: str = Query("lexical", description="lexical | tfidf | dense | hybrid"),
        limit: int = Query(DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
        offset: int = Query(0, ge=0),
        conn: sqlite3.Connection = Depends(get_conn),
    ) -> dict[str, object]:
        _require_ready(conn, collection_id)

        if sort not in SORTS:
            raise ApiError(
                "invalid_request",
                f"unknown sort {sort!r}",
                422,
                {"allowed": list(SORTS)},
            )
        if mode not in MODES:
            raise ApiError(
                "invalid_request",
                f"unknown mode {mode!r}",
                422,
                {"allowed": list(MODES)},
            )

        query_vector = None
        if q and mode in {"dense", "hybrid"}:
            query_vector = request.app.state.query_embedder.vector(q)

        result = rank_papers(
            conn,
            collection_id,
            query=q,
            mode=mode,
            top_k=limit,
            offset=offset,
            query_vector=query_vector,
            sort=sort,
        )
        details = paper_details(conn, [paper.paper_id for paper in result.papers])
        return {
            "papers": [
                {
                    "paper_id": paper.paper_id,
                    "title": paper.title,
                    "year": paper.year,
                    "citation_count": paper.citation_count,
                    "score": paper.score,
                    "best_chunk_id": paper.best_chunk_id,
                    **details.get(paper.paper_id, {}),
                }
                for paper in result.papers
            ],
            "mode": result.mode,
            "degraded": result.degraded,
            "total": result.total,
            "offset": offset,
            "limit": limit,
        }

    @app.get(f"{API_PREFIX}/papers/{{paper_id}}")
    def get_paper(
        paper_id: str, conn: sqlite3.Connection = Depends(get_conn)
    ) -> dict[str, object]:
        detail = paper_detail(conn, paper_id)
        if detail is None:
            raise ApiError(
                "not_found",
                f"paper {paper_id} does not exist",
                404,
                {"paper_id": paper_id},
            )
        return detail

    @app.get(f"{API_PREFIX}/collections/{{collection_id}}/map")
    def collection_map(
        request: Request,
        collection_id: str,
        conn: sqlite3.Connection = Depends(get_conn),
    ) -> dict[str, object]:
        _require_ready(conn, collection_id)
        try:
            payload = build_map(
                conn,
                collection_id,
                projector=request.app.state.map_projector,
                clusterer=request.app.state.map_clusterer,
            )
        except MapDependencyMissing as exc:
            raise ApiError("map_dependency_missing", str(exc), 503) from exc
        except MapUnavailable as exc:
            raise ApiError(
                "map_needs_more_papers",
                f"a similarity map needs at least {exc.needed} papers with vectors; "
                f"this collection has {exc.have}",
                409,
                {"reason": exc.reason, "needed": exc.needed, "have": exc.have},
            ) from exc
        return payload.as_dict()

    @app.get(f"{API_PREFIX}/collections/{{collection_id}}/graph")
    def citation_graph(
        collection_id: str,
        include_external: bool = Query(
            False, description="render references outside the collection as nodes"
        ),
        conn: sqlite3.Connection = Depends(get_conn),
    ) -> dict[str, object]:
        _require_ready(conn, collection_id)
        payload = build_graph(conn, collection_id, include_external=include_external)
        return payload.as_dict()

    @app.post(f"{API_PREFIX}/collections/{{collection_id}}/ask")
    def ask_endpoint(
        request: Request,
        collection_id: str,
        body: AskBody,
        conn: sqlite3.Connection = Depends(get_conn),
    ) -> dict[str, object]:
        _require_ready(conn, collection_id)
        if body.mode not in MODES:
            raise ApiError(
                "invalid_request", f"unknown mode {body.mode!r}", 422, {"allowed": list(MODES)}
            )
        query_vector = None
        if body.mode in {"dense", "hybrid"}:
            query_vector = request.app.state.query_embedder.vector(body.question)
        return answer_question(
            conn,
            collection_id,
            body.question.strip(),
            llm=llm,
            query_vector=query_vector,
            k=body.k,
            mode=body.mode,
            refusal_tau=resolved.refusal_tau,
            tau_mode=resolved.refusal_tau_mode,
        )

    @app.post(f"{API_PREFIX}/papers/{{paper_id}}/summary")
    def summary_endpoint(
        paper_id: str, conn: sqlite3.Connection = Depends(get_conn)
    ) -> dict[str, object]:
        try:
            return paper_summary(conn, paper_id, llm=llm)
        except PaperNotFound as exc:
            raise ApiError(
                "not_found", f"paper {paper_id} does not exist", 404, {"paper_id": paper_id}
            ) from exc
        except LLMUnavailable as exc:
            raise _generation_error(exc) from exc

    @app.post(f"{API_PREFIX}/collections/{{collection_id}}/related-work")
    def related_work_endpoint(
        request: Request,
        collection_id: str,
        body: RelatedWorkBody,
        conn: sqlite3.Connection = Depends(get_conn),
    ) -> dict[str, object]:
        _require_ready(conn, collection_id)
        paper_ids = list(body.paper_ids or [])
        if body.cluster_id is not None:
            if paper_ids:
                raise ApiError(
                    "invalid_request", "send paper_ids or cluster_id, not both", 422
                )
            try:
                payload = build_map(
                    conn,
                    collection_id,
                    projector=request.app.state.map_projector,
                    clusterer=request.app.state.map_clusterer,
                )
            except MapDependencyMissing as exc:
                raise ApiError("map_dependency_missing", str(exc), 503) from exc
            except MapUnavailable as exc:
                raise ApiError(
                    "map_needs_more_papers",
                    f"clusters need a map of at least {exc.needed} papers with vectors; "
                    f"this collection has {exc.have}",
                    409,
                    {"reason": exc.reason, "needed": exc.needed, "have": exc.have},
                ) from exc
            points = [p.__dict__ for p in payload.points]
            members = central_ids(points).get(body.cluster_id)
            if not members:
                raise ApiError(
                    "not_found",
                    f"cluster {body.cluster_id} does not exist on this map",
                    404,
                    {"cluster_id": body.cluster_id},
                )
            paper_ids = members[:RELATED_MAX_PAPERS]
        try:
            return draft_related_work(conn, collection_id, paper_ids, llm=llm)
        except RelatedWorkError as exc:
            raise ApiError("invalid_request", str(exc), 422) from exc
        except LLMUnavailable as exc:
            raise _generation_error(exc) from exc

    @app.post(f"{API_PREFIX}/collections/{{collection_id}}/contradictions")
    def contradictions_endpoint(
        request: Request,
        collection_id: str,
        body: ContradictionBody,
        conn: sqlite3.Connection = Depends(get_conn),
    ) -> dict[str, object]:
        _require_ready(conn, collection_id)
        paper_ids = list(body.paper_ids or [])
        if body.cluster_id is not None:
            if paper_ids:
                raise ApiError("invalid_request", "send paper_ids or cluster_id, not both", 422)
            try:
                payload = build_map(
                    conn,
                    collection_id,
                    projector=request.app.state.map_projector,
                    clusterer=request.app.state.map_clusterer,
                )
            except MapDependencyMissing as exc:
                raise ApiError("map_dependency_missing", str(exc), 503) from exc
            except MapUnavailable as exc:
                raise ApiError(
                    "map_needs_more_papers",
                    f"clusters need a map of at least {exc.needed} papers with vectors; "
                    f"this collection has {exc.have}",
                    409,
                    {"reason": exc.reason, "needed": exc.needed, "have": exc.have},
                ) from exc
            members = central_ids([p.__dict__ for p in payload.points]).get(body.cluster_id)
            if not members:
                raise ApiError(
                    "not_found",
                    f"cluster {body.cluster_id} does not exist on this map",
                    404,
                    {"cluster_id": body.cluster_id},
                )
            paper_ids = members[:CONTRADICTION_MAX_PAPERS]
        try:
            return compare_papers(conn, collection_id, paper_ids, llm=llm, max_pairs=body.max_pairs)
        except ContradictionError as exc:
            raise ApiError("invalid_request", str(exc), 422) from exc

    @app.get(f"{API_PREFIX}/collections/{{collection_id}}/evaluation")
    def evaluation_report(
        collection_id: str, conn: sqlite3.Connection = Depends(get_conn)
    ) -> dict[str, object]:
        _require_ready(conn, collection_id)
        report = load_report(resolved.data_dir, collection_id)
        if report is None:
            raise ApiError(
                "no_evaluation",
                "no evaluation has been run for this collection; "
                f"run: python -m cli eval latency --collection {collection_id}",
                404,
                {"collection_id": collection_id},
            )
        return report

    @app.get(f"{API_PREFIX}/collections/{{collection_id}}/map/labels")
    def map_labels(
        request: Request,
        collection_id: str,
        conn: sqlite3.Connection = Depends(get_conn),
    ) -> dict[str, object]:
        _require_ready(conn, collection_id)
        try:
            payload = build_map(
                conn,
                collection_id,
                projector=request.app.state.map_projector,
                clusterer=request.app.state.map_clusterer,
            )
        except MapDependencyMissing as exc:
            raise ApiError("map_dependency_missing", str(exc), 503) from exc
        except MapUnavailable as exc:
            raise ApiError(
                "map_needs_more_papers",
                f"a similarity map needs at least {exc.needed} papers with vectors; "
                f"this collection has {exc.have}",
                409,
                {"reason": exc.reason, "needed": exc.needed, "have": exc.have},
            ) from exc
        rows = conn.execute(
            "SELECT id, title, abstract FROM papers WHERE collection_id=?", (collection_id,)
        ).fetchall()
        titles = {row["id"]: row["title"] for row in rows}
        abstracts = {row["id"]: row["abstract"] for row in rows}
        topic = conn.execute("SELECT query FROM collections WHERE id=?", (collection_id,)).fetchone()["query"]
        return cluster_labels(
            [p.__dict__ for p in payload.points], titles, llm=llm, topic=topic, abstracts=abstracts
        )

    @app.api_route(
        f"{API_PREFIX}/{{path:path}}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        include_in_schema=False,
    )
    def api_not_found(path: str) -> None:
        raise ApiError(
            "not_found",
            f"no API route /api/{path}",
            404,
            {"path": path},
        )

    dist = Path(static_dir) if static_dir is not None else WEB_DIST
    if (dist / "index.html").is_file():
        app.mount("/", StaticFiles(directory=str(dist), html=True), name="web")

    return app


app = create_app()
