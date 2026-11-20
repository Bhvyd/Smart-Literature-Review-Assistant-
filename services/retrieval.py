
from __future__ import annotations

import json
import threading
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from rag_core import Retriever
from store import repo


@dataclass(frozen=True)
class PaperText:
    paper_id: str
    title: str
    abstract: str | None
    year: int | None
    citation_count: int | None


@dataclass(frozen=True)
class ChunkText:
    paper_id: str
    chunk_id: str
    text: str


@dataclass(frozen=True)
class RankedPaper:
    paper_id: str
    title: str
    year: int | None
    citation_count: int | None
    score: float | None
    best_chunk_id: str | None


@dataclass(frozen=True)
class RankingResult:
    papers: list[RankedPaper]
    mode: str
    degraded: list[str]
    total: int = 0


SORTS = ("relevance", "influence", "year")


def _page(items: Sequence[Any], offset: int, top_k: int | None) -> list[Any]:
    start = max(0, offset)
    end = None if top_k is None else start + max(0, top_k)
    return list(items[start:end])


def _decode_authors(raw: Any) -> list[str]:
    if not raw:
        return []
    try:
        decoded = json.loads(raw)
    except (TypeError, ValueError):
        return []
    return [str(author) for author in decoded] if isinstance(decoded, list) else []


def source_url(paper_id: str, source: str | None, arxiv_id: str | None, pdf_url: str | None) -> str | None:
    if source == "upload":
        return f"/api/papers/{paper_id}/pdf"
    if arxiv_id:
        return f"https://arxiv.org/abs/{arxiv_id}"
    return pdf_url


def paper_details(conn: Any, paper_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
    if not paper_ids:
        return {}
    placeholders = ",".join("?" for _ in paper_ids)
    rows = conn.execute(
        f"""
        SELECT id, abstract, authors, venue, pdf_url, arxiv_id, parse_status,
               parse_note, has_fulltext, source
        FROM papers WHERE id IN ({placeholders})
        """,
        list(paper_ids),
    ).fetchall()
    return {
        row["id"]: {
            "abstract": row["abstract"],
            "authors": _decode_authors(row["authors"]),
            "venue": row["venue"],
            "pdf_url": row["pdf_url"],
            "arxiv_id": row["arxiv_id"],
            "parse_status": row["parse_status"],
            "parse_note": row["parse_note"],
            "has_fulltext": row["has_fulltext"],
            "source": row["source"],
            "source_url": source_url(row["id"], row["source"], row["arxiv_id"], row["pdf_url"]),
        }
        for row in rows
    }


def paper_detail(conn: Any, paper_id: str) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT p.id, p.collection_id, p.arxiv_id, p.openalex_id, p.doi, p.title,
               p.abstract, p.authors, p.year, p.venue, p.pdf_url, p.citation_count,
               p.source, p.parse_status, p.parse_note, p.has_fulltext,
               (SELECT count(*) FROM chunks c WHERE c.paper_id=p.id) AS chunk_count
        FROM papers p WHERE p.id=?
        """,
        (paper_id,),
    ).fetchone()
    if row is None:
        return None
    return {
        "paper_id": row["id"],
        "collection_id": row["collection_id"],
        "arxiv_id": row["arxiv_id"],
        "openalex_id": row["openalex_id"],
        "doi": row["doi"],
        "title": row["title"],
        "abstract": row["abstract"],
        "authors": _decode_authors(row["authors"]),
        "year": row["year"],
        "venue": row["venue"],
        "pdf_url": row["pdf_url"],
        "citation_count": row["citation_count"],
        "source": row["source"],
        "parse_status": row["parse_status"],
        "parse_note": row["parse_note"],
        "has_fulltext": row["has_fulltext"],
        "source_url": source_url(row["id"], row["source"], row["arxiv_id"], row["pdf_url"]),
        "chunk_count": int(row["chunk_count"]),
    }


def _order_by_metadata(papers: Sequence[PaperText], sort: str) -> list[PaperText]:
    if sort == "year":
        return sorted(
            papers,
            key=lambda p: (p.year is None, -(p.year or 0), p.title, p.paper_id),
        )
    return sorted(
        papers,
        key=lambda p: (
            p.citation_count is None,
            -(p.citation_count or 0),
            p.title,
            p.paper_id,
        ),
    )


def load_papers(conn: Any, collection_id: str) -> list[PaperText]:
    rows = conn.execute(
        """
        SELECT id, title, abstract, year, citation_count
        FROM papers WHERE collection_id=?
        ORDER BY id
        """,
        (collection_id,),
    ).fetchall()
    return [
        PaperText(
            paper_id=row["id"],
            title=row["title"],
            abstract=row["abstract"],
            year=row["year"],
            citation_count=row["citation_count"],
        )
        for row in rows
    ]


def load_chunks(conn: Any, collection_id: str) -> list[ChunkText]:
    rows = conn.execute(
        """
        SELECT c.id AS chunk_id, c.paper_id, c.text
        FROM chunks c
        JOIN papers p ON p.id = c.paper_id
        WHERE p.collection_id=?
        ORDER BY c.paper_id, c.seq
        """,
        (collection_id,),
    ).fetchall()
    return [
        ChunkText(paper_id=row["paper_id"], chunk_id=row["chunk_id"], text=row["text"])
        for row in rows
    ]


def load_chunk_vectors(conn: Any, collection_id: str) -> dict[str, list[float]]:
    rows = conn.execute(
        """
        SELECT e.owner_id, e.dim, e.vec
        FROM embeddings e
        JOIN chunks c ON c.id = e.owner_id
        JOIN papers p ON p.id = c.paper_id
        WHERE e.owner_type='chunk' AND p.collection_id=?
        """,
        (collection_id,),
    ).fetchall()

    return {
        row["owner_id"]: repo.unpack_vector(bytes(row["vec"]), int(row["dim"]))
        for row in rows
    }


class CollectionIndex:
    def __init__(
        self,
        papers: Sequence[PaperText],
        chunks: Sequence[ChunkText],
        dense_vectors: dict[str, list[float]],
    ) -> None:
        self.papers = list(papers)
        self.chunks = list(chunks)
        self.dense_vectors = dense_vectors

        self.chunk_ids: list[str] = [chunk.chunk_id for chunk in self.chunks]
        self.texts: list[str] = [chunk.text for chunk in self.chunks]
        self.chunk_paper: list[str] = [chunk.paper_id for chunk in self.chunks]
        self.paper_by_chunk: dict[str, str] = {
            chunk.chunk_id: chunk.paper_id for chunk in self.chunks
        }
        self.paper_by_id = {paper.paper_id: paper for paper in self.papers}

        covered = {chunk.paper_id for chunk in self.chunks}
        self.abstract_documents: dict[str, str] = {
            paper.paper_id: paper.title + ("\n" + paper.abstract if paper.abstract else "")
            for paper in self.papers
            if paper.paper_id not in covered and paper.abstract
        }

    @classmethod
    def build(cls, conn: Any, collection_id: str) -> CollectionIndex:
        return cls(
            load_papers(conn, collection_id),
            load_chunks(conn, collection_id),
            load_chunk_vectors(conn, collection_id),
        )

    @property
    def missing_vectors(self) -> int:
        return len(self.chunk_ids) - len(self.dense_vectors)

    def documents(self) -> list[tuple[str, str]]:
        docs = [(chunk.chunk_id, chunk.text) for chunk in self.chunks]
        docs.extend(self.abstract_documents.items())
        return docs

    def retriever(self) -> Retriever:
        return Retriever.build(self.documents(), self.dense_vectors or None)

    def paper_for_document(self, doc_id: str) -> str | None:
        return self.paper_by_chunk.get(doc_id) or (
            doc_id if doc_id in self.abstract_documents else None
        )


_CACHE_SIZE = 2
_cache: OrderedDict[tuple[Any, ...], tuple[CollectionIndex, Retriever]] = OrderedDict()
_cache_lock = threading.Lock()


def _fingerprint(conn: Any, collection_id: str) -> tuple[Any, ...]:
    row = conn.execute(
        """
        SELECT
          (SELECT updated_at FROM collections WHERE id=?),
          (SELECT count(*) FROM papers WHERE collection_id=?),
          count(c.id),
          COALESCE(sum(length(c.text)), 0)
        FROM chunks c JOIN papers p ON p.id = c.paper_id
        WHERE p.collection_id=?
        """,
        (collection_id, collection_id, collection_id),
    ).fetchone()
    database = conn.execute("PRAGMA database_list").fetchone()
    return (database[2] if database else "", collection_id, *tuple(row))


def evict_collection(collection_id: str) -> int:
    with _cache_lock:
        stale = [key for key in _cache if key[1] == collection_id]
        for key in stale:
            del _cache[key]
    return len(stale)


def cached_index(conn: Any, collection_id: str) -> tuple[CollectionIndex, Retriever]:
    key = _fingerprint(conn, collection_id)
    with _cache_lock:
        hit = _cache.get(key)
        if hit is not None:
            _cache.move_to_end(key)
            return hit

    index = CollectionIndex.build(conn, collection_id)
    entry = (index, index.retriever())
    with _cache_lock:
        for stale in [k for k in _cache if k[:2] == key[:2]]:
            del _cache[stale]
        _cache[key] = entry
        while len(_cache) > _CACHE_SIZE:
            _cache.popitem(last=False)
    return entry


def paper_scores(hits: Sequence[Any], index: CollectionIndex) -> dict[str, tuple[float, str]]:
    best: dict[str, tuple[float, str]] = {}
    for hit in hits:
        paper_id = index.paper_for_document(hit.doc_id)
        if paper_id is None:
            continue
        current = best.get(paper_id)
        if current is None or hit.score > current[0]:
            best[paper_id] = (hit.score, hit.doc_id)
    return best


def rank_papers(
    conn: Any,
    collection_id: str,
    *,
    query: str | None,
    mode: str = "lexical",
    top_k: int | None = 20,
    offset: int = 0,
    query_vector: Sequence[float] | None = None,
    candidate_documents: int = 500,
    sort: str = "relevance",
) -> RankingResult:
    if sort not in SORTS:
        raise ValueError(f"unknown sort {sort!r}; expected one of {SORTS}")

    index, retriever = cached_index(conn, collection_id)
    if not index.papers:
        return RankingResult([], mode, [], total=0)

    if sort in {"influence", "year"} or query is None or not query.strip():
        effective_sort = sort if sort in {"influence", "year"} else "influence"
        ordered = _order_by_metadata(index.papers, effective_sort)
        return RankingResult(
            [
                RankedPaper(
                    paper_id=paper.paper_id,
                    title=paper.title,
                    year=paper.year,
                    citation_count=paper.citation_count,
                    score=None,
                    best_chunk_id=None,
                )
                for paper in _page(ordered, offset, top_k)
            ],
            mode=effective_sort,
            degraded=[],
            total=len(ordered),
        )

    degraded: list[str] = []
    effective = mode
    if effective in {"dense", "hybrid"} and query_vector is None:
        effective = "lexical"
        degraded.append("query_vector_missing")
    if effective in {"dense", "hybrid"} and not index.dense_vectors:
        effective = "lexical"
        degraded.append("no_dense_vectors")

    result = retriever.search(
        query, k=candidate_documents, mode=effective, query_vector=query_vector
    )
    best = paper_scores(result.hits, index)

    ranked = sorted(
        best.items(),
        key=lambda item: (-item[1][0], index.paper_by_id[item[0]].title, item[0]),
    )

    return RankingResult(
        [
            RankedPaper(
                paper_id=paper_id,
                title=index.paper_by_id[paper_id].title,
                year=index.paper_by_id[paper_id].year,
                citation_count=index.paper_by_id[paper_id].citation_count,
                score=score,
                best_chunk_id=doc_id,
            )
            for paper_id, (score, doc_id) in _page(ranked, offset, top_k)
        ],
        mode=effective,
        degraded=degraded,
        total=len(ranked),
    )
