
from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ingest.arxiv import ArxivRecord
from ingest.arxiv import search as arxiv_search
from ingest.cache import Cache
from ingest.chunk import Chunk, chunk_text, looks_like_references
from ingest.http import FetchError, HttpClient, RateLimited
from ingest.openalex import OpenAlexWork
from ingest.openalex import lookup_by_arxiv as openalex_lookup
from ingest.openalex import lookup_by_dois as openalex_by_dois
from ingest.openalex import lookup_by_title as openalex_by_title
from ingest.openalex import titles_for as openalex_titles
from ingest.pdf import extract_text
from ingest.pdf import fetch_pdf as http_fetch_pdf
from store import repo
from store.repo import PaperRow

from embed.specter2 import MAX_TOKENS

PROXIMITY_ADAPTER = "allenai/specter2"
CHUNK_TARGET_TOKENS = 350
CHUNK_OVERLAP_TOKENS = 50
CHUNK_MIN_TOKENS = 40
PDF_FETCH_INTERVAL = 3.0


@dataclass(frozen=True)
class BuildReport:
    collection_id: str
    n_papers: int
    n_chunks: int
    n_embeddings: int
    n_edges: int
    degraded: list[str] = field(default_factory=list)


class CollectionBuildError(Exception):
    pass


class BuildCancelled(Exception):
    pass


def raise_if_cancelled(conn: Any, job_id: str, live: tuple[str, ...] = ("running",)) -> None:
    job = repo.get_job(conn, job_id)
    if job is None or job["status"] not in live:
        raise BuildCancelled(job_id)


def abstract_paper_text(title: str, abstract: str) -> str:
    return f"{title}\n{abstract}"


def _year(published: str) -> int | None:
    head = published.strip()[:4]
    return int(head) if head.isdigit() else None


class _ArxivClient:
    def __init__(self, client: HttpClient, sleep: Callable[[float], None]) -> None:
        self._client = client
        self._sleep = sleep

    def search(self, query: str, limit: int) -> list[ArxivRecord]:
        return arxiv_search(self._client, query, limit, sleep=self._sleep)


class _OpenAlexClient:
    def __init__(self, client: HttpClient, mailto: str | None) -> None:
        self._client = client
        self._mailto = mailto

    def lookup_by_arxiv(self, arxiv_id: str) -> OpenAlexWork | None:
        return openalex_lookup(self._client, arxiv_id, mailto=self._mailto)

    def lookup_by_doi(self, doi: str) -> OpenAlexWork | None:
        found = openalex_by_dois(self._client, [doi], mailto=self._mailto)
        return next(iter(found.values()), None)

    def lookup_by_title(self, title: str) -> OpenAlexWork | None:
        return openalex_by_title(self._client, title, mailto=self._mailto)

    def titles_for(
        self,
        work_ids: Sequence[str],
        on_error: Callable[[Exception], None] | None = None,
    ) -> dict[str, str]:
        return openalex_titles(
            self._client, work_ids, mailto=self._mailto, on_error=on_error
        )


def build_collection(
    conn: Any,
    *,
    query: str,
    limit: int = 100,
    settings: Any,
    arxiv: Any | None = None,
    openalex: Any | None = None,
    embedder: Any | None = None,
    client: HttpClient | None = None,
    fetch_pdf: Callable[[Any, str], bytes | None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[..., None] = print,
    reuse: bool = True,
    collection_id: str | None = None,
    job_id: str | None = None,
) -> BuildReport:
    if job_id is not None:
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
            sleep=sleep,
        )
    if arxiv is None:
        arxiv = _ArxivClient(client, sleep)
    if openalex is None:
        openalex = _OpenAlexClient(client, getattr(settings, "contact_email", None))
    if fetch_pdf is None:
        fetch_pdf = http_fetch_pdf
    if embedder is None:
        from embed.specter2 import Specter2Embedder

        embedder = Specter2Embedder(model_dir=str(getattr(settings, "models_dir", "./models")))

    model_label = (
        getattr(embedder, "model_name", None)
        or getattr(settings, "embedding_model", None)
        or PROXIMITY_ADAPTER
    )
    collection_id = (
        collection_id
        if collection_id is not None
        else (repo.find_ready_collection(conn, mode="topic", query=query) if reuse else None)
    )
    if collection_id is None:
        collection_id = repo.create_collection(
            conn, mode="topic", query=query, embedding_model=model_label
        )
    else:
        repo.set_collection_status(conn, collection_id, "building")
        repo.set_collection_embedding_model(conn, collection_id, model_label)
    if job_id is None:
        job_id = repo.create_job(conn, collection_id=collection_id, kind="build")
    repo.update_job(conn, job_id, status="running", progress=0.0, message="searching arXiv")
    log(f"[build] collection={collection_id} query={query!r} limit={limit}")

    try:
        report = _run(
            conn,
            collection_id=collection_id,
            job_id=job_id,
            query=query,
            limit=limit,
            arxiv=arxiv,
            openalex=openalex,
            embedder=embedder,
            client=client,
            fetch_pdf=fetch_pdf,
            model_label=model_label,
            degraded=degraded,
            log=log,
            sleep=sleep,
        )
    except BuildCancelled:
        log("[build] cancelled")
        raise
    except Exception as exc:
        job = repo.get_job(conn, job_id)
        if job is None or job["status"] != "running":
            log(f"[build] cancelled (and then: {exc})")
            raise BuildCancelled(job_id) from exc
        repo.set_collection_status(conn, collection_id, "failed", error=str(exc))
        repo.update_job(conn, job_id, status="failed", error=str(exc))
        log(f"[build] failed: {exc}")
        raise CollectionBuildError(str(exc)) from exc
    finally:
        if owns_client:
            client.close()
        if owns_embedder:
            embedder.close()

    repo.set_collection_degraded(conn, collection_id, report.degraded)
    repo.update_job(conn, job_id, status="done", progress=1.0, message="collection ready")
    log(
        f"[build] done papers={report.n_papers} chunks={report.n_chunks} "
        f"embeddings={report.n_embeddings} edges={report.n_edges} degraded={report.degraded}"
    )
    return report


def _run(
    conn: Any,
    *,
    collection_id: str,
    job_id: str,
    query: str,
    limit: int,
    arxiv: Any,
    openalex: Any,
    embedder: Any,
    client: Any,
    fetch_pdf: Callable[[Any, str], bytes | None],
    model_label: str,
    degraded: list[str],
    log: Callable[..., None],
    sleep: Callable[[float], None],
) -> BuildReport:
    records: list[ArxivRecord] = arxiv.search(query, limit)
    log(f"[build] arxiv returned {len(records)} records")
    raise_if_cancelled(conn, job_id)

    if not records:
        raise ValueError(
            f"No arXiv papers match “{query}”. Check the spelling or try a broader topic."
        )

    skip_enrichment = False
    n_embeddings = 0
    pending_edges: list[tuple[str, str]] = []

    for index, record in enumerate(records, start=1):
        raise_if_cancelled(conn, job_id)
        requests_before = getattr(client, "network_requests", 0)
        work: OpenAlexWork | None = None
        if not skip_enrichment:
            try:
                work = openalex.lookup_by_arxiv(record.arxiv_id)
            except RateLimited:
                degraded.append("openalex_rate_limited")
                skip_enrichment = True
                log("[build] openalex rate limited; continuing without enrichment")
            except FetchError as exc:
                if "openalex_fetch_failed" not in degraded:
                    degraded.append("openalex_fetch_failed")
                log(f"[build] openalex lookup failed for {record.arxiv_id}: {exc}")

        paper_id, body = _store_paper(
            conn,
            collection_id=collection_id,
            record=record,
            work=work,
            client=client,
            fetch_pdf=fetch_pdf,
            log=log,
            before_write=lambda: raise_if_cancelled(conn, job_id),
        )
        n_embeddings += _embed_and_store(
            conn,
            paper_id=paper_id,
            body=body,
            embedder=embedder,
            model_label=model_label,
            title=record.title,
            abstract=record.abstract,
        )

        if work is not None:
            for reference in work.referenced_works:
                pending_edges.append((paper_id, reference))

        went_to_network = getattr(client, "network_requests", 1) > requests_before
        if index < len(records) and went_to_network:
            sleep(PDF_FETCH_INTERVAL)

        repo.update_job(
            conn,
            job_id,
            status="running",
            progress=round(index / (len(records) + 1), 3),
            message=f"ingested {index}/{len(records)}",
        )

    raise_if_cancelled(conn, job_id)
    n_edges = _store_edges(
        conn,
        collection_id=collection_id,
        pending=pending_edges,
        openalex=openalex,
        degraded=degraded,
        log=log,
    )

    raise_if_cancelled(conn, job_id)
    n_papers = repo.count_papers(conn, collection_id)
    n_chunks = repo.count_chunks(conn, collection_id)
    repo.set_collection_counts(conn, collection_id, n_papers=n_papers, n_chunks=n_chunks)
    repo.set_collection_status(conn, collection_id, "ready")

    return BuildReport(
        collection_id=collection_id,
        n_papers=n_papers,
        n_chunks=n_chunks,
        n_embeddings=n_embeddings,
        n_edges=n_edges,
        degraded=degraded,
    )


def _store_paper(
    conn: Any,
    *,
    collection_id: str,
    record: ArxivRecord,
    work: OpenAlexWork | None,
    client: Any,
    fetch_pdf: Callable[[Any, str], bytes | None],
    log: Callable[..., None],
    before_write: Callable[[], None] = lambda: None,
) -> tuple[str, str]:
    pdf_bytes = None
    if record.pdf_url:
        try:
            pdf_bytes = fetch_pdf(client, record.pdf_url)
        except RateLimited:
            pdf_bytes = None
            log(f"[build] pdf rate limited for {record.arxiv_id}")

    parse_status = "not_attempted"
    parse_note: str | None = None
    full_text: str | None = None

    if pdf_bytes is None:
        parse_status = "failed"
        parse_note = "pdf unavailable"
    else:
        extracted = extract_text(pdf_bytes)
        parse_status = extracted.quality
        if extracted.quality == "ok":
            full_text = extracted.text
        elif extracted.quality == "low_quality":
            parse_note = f"alpha_ratio={extracted.alpha_ratio:.2f} pages={extracted.pages}"
        else:
            parse_note = "pdf could not be parsed"

    body = full_text or abstract_paper_text(record.title, record.abstract)

    row = PaperRow(
        arxiv_id=record.arxiv_id,
        openalex_id=work.openalex_id if work else None,
        doi=work.doi if work else None,
        title=record.title,
        abstract=record.abstract,
        authors=record.authors,
        year=work.year if work and work.year else _year(record.published),
        venue=work.venue if work else None,
        pdf_url=record.pdf_url,
        citation_count=work.cited_by_count if work else None,
        source="arxiv",
        parse_status=parse_status,
        parse_note=parse_note,
        has_fulltext=1 if full_text else 0,
    )
    before_write()
    paper_id = repo.upsert_paper(conn, collection_id=collection_id, paper=row)
    return paper_id, body


def _embed_and_store(
    conn: Any,
    *,
    paper_id: str,
    body: str,
    embedder: Any,
    model_label: str,
    title: str = "",
    abstract: str | None = None,
) -> int:
    if title.strip() and (abstract or body.strip()) and hasattr(embedder, "paper_text"):
        paper_vector = embedder.embed_texts([embedder.paper_text(title, abstract)])
        if paper_vector:
            repo.save_embedding(
                conn,
                owner_type="paper",
                owner_id=paper_id,
                vec=paper_vector[0],
                model=model_label,
            )

    if not body.strip():
        return 0

    chunks: list[Chunk] = chunk_text(
        body,
        target=CHUNK_TARGET_TOKENS,
        overlap=CHUNK_OVERLAP_TOKENS,
        min_tokens=CHUNK_MIN_TOKENS,
        count_tokens=embedder.count_tokens,
        max_tokens=MAX_TOKENS,
    )
    kept = [chunk for chunk in chunks if not looks_like_references(chunk.text)]
    chunks = [Chunk(seq=i, text=c.text, n_tokens=c.n_tokens) for i, c in enumerate(kept)]
    if not chunks:
        return 0

    repo.replace_chunks(conn, paper_id, chunks)

    ids_by_seq = {seq: chunk_id for chunk_id, seq in repo.chunk_ids_for_paper(conn, paper_id)}
    vectors = embedder.embed_texts([chunk.text for chunk in chunks])
    if len(vectors) != len(chunks):
        raise CollectionBuildError(
            f"embedder returned {len(vectors)} vectors for {len(chunks)} chunks on paper {paper_id}"
        )
    stored = 0
    for chunk, vector in zip(chunks, vectors):
        chunk_id = ids_by_seq.get(chunk.seq)
        if chunk_id is None:
            continue
        repo.save_embedding(
            conn, owner_type="chunk", owner_id=chunk_id, vec=vector, model=model_label
        )
        stored += 1
    return stored


def _store_edges(
    conn: Any,
    *,
    collection_id: str,
    pending: list[tuple[str, str]],
    openalex: Any,
    degraded: list[str],
    log: Callable[..., None],
) -> int:
    if not pending:
        return 0

    known = repo.paper_ids_by_openalex(conn, collection_id)
    external = sorted({ref for _, ref in pending if ref not in known})

    def note_title_failure(exc: Exception) -> None:
        reason = (
            "openalex_rate_limited" if isinstance(exc, RateLimited) else "openalex_fetch_failed"
        )
        if reason not in degraded:
            degraded.append(reason)
        log(f"[build] reference title lookup failed: {exc}")

    titles: dict[str, str] = {}
    if external:
        try:
            titles = openalex.titles_for(external, on_error=note_title_failure)
        except (RateLimited, FetchError) as exc:
            note_title_failure(exc)

    written = 0
    for source_id, reference in pending:
        dst_paper_id = known.get(reference)
        repo.upsert_edge(
            conn,
            collection_id=collection_id,
            src_paper_id=source_id,
            dst_paper_id=dst_paper_id,
            dst_openalex_id=reference,
            dst_title=titles.get(reference),
            dst_doi=None,
        )
        written += 1

    return repo.count_edges(conn, collection_id)
