
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from .dense import cosine_search, document_norms
from .fuse import rrf
from .index import InvertedIndex
from .lexical import bm25_search, tfidf_cosine_search

MODES = ("lexical", "tfidf", "dense", "hybrid")


@dataclass(frozen=True)
class SearchHit:
    doc_id: str
    score: float
    rank: int


@dataclass(frozen=True)
class SearchResult:
    hits: list[SearchHit] = field(default_factory=list)
    mode: str = "lexical"
    degraded: list[str] = field(default_factory=list)


class Retriever:
    def __init__(
        self,
        index: InvertedIndex,
        dense_vectors: Mapping[str, Sequence[float]] | None = None,
    ) -> None:
        self.index = index
        self.dense_vectors = dict(dense_vectors) if dense_vectors is not None else None
        self._norms: dict[str, float] | None = None

    @classmethod
    def build(
        cls,
        documents: Iterable[tuple[str, str]],
        dense_vectors: Mapping[str, Sequence[float]] | None = None,
    ) -> Retriever:
        index = InvertedIndex()
        for doc_id, text in documents:
            index.add(doc_id, text)
        index.finalize()
        return cls(index, dense_vectors)

    def search(
        self,
        query: str,
        k: int = 10,
        mode: str = "lexical",
        query_vector: Sequence[float] | None = None,
    ) -> SearchResult:
        if mode not in MODES:
            raise ValueError(f"unknown search mode: {mode!r} (expected one of {MODES})")

        if mode == "lexical":
            pairs = bm25_search(self.index, query, k)
        elif mode == "tfidf":
            pairs = tfidf_cosine_search(self.index, query, k)
        elif mode == "dense":
            pairs = self._dense_pairs(query_vector, k)
        else:
            pairs = self._hybrid_pairs(query, query_vector, k)

        hits = [
            SearchHit(doc_id=doc_id, score=score, rank=rank)
            for rank, (doc_id, score) in enumerate(pairs, start=1)
        ]
        return SearchResult(hits=hits, mode=mode, degraded=[])

    def _doc_norms(self) -> dict[str, float]:
        if self._norms is None:
            self._norms = document_norms(self.dense_vectors or {})
        return self._norms

    def _dense_pairs(
        self, query_vector: Sequence[float] | None, k: int
    ) -> list[tuple[str, float]]:
        if self.dense_vectors is None:
            raise ValueError("dense search requires document vectors")
        if query_vector is None:
            raise ValueError("dense search requires a query vector")
        return cosine_search(query_vector, self.dense_vectors, k, norms=self._doc_norms())

    def _hybrid_pairs(
        self,
        query: str,
        query_vector: Sequence[float] | None,
        k: int,
    ) -> list[tuple[str, float]]:
        if self.dense_vectors is None:
            raise ValueError("hybrid search requires document vectors")
        if query_vector is None:
            raise ValueError("hybrid search requires a query vector")

        lexical = bm25_search(self.index, query, None)
        dense = cosine_search(query_vector, self.dense_vectors, None, norms=self._doc_norms())
        fused = rrf([[doc_id for doc_id, _ in lexical], [doc_id for doc_id, _ in dense]])
        return fused if k is None else fused[:k]
