
from __future__ import annotations

import math
from collections.abc import Sequence

from .index import InvertedIndex
from .select import top_k
from .tokenize import tokenize as _tokenize

K1 = 1.5
B = 0.75


def _terms(query: str | Sequence[str]) -> list[str]:
    return _tokenize(query) if isinstance(query, str) else list(query)


def bm25_score(index: InvertedIndex, query: str | Sequence[str], doc_id: str) -> float:
    terms = _terms(query)
    if not terms:
        return 0.0
    avgdl = index.avgdl
    if avgdl <= 0.0:
        return 0.0

    dl = index.doc_length(doc_id)
    total = 0.0
    for term in terms:
        freq = index.postings(term).get(doc_id, 0)
        if not freq:
            continue
        denominator = freq + K1 * (1.0 - B + B * dl / avgdl)
        total += index.idf_bm25(term) * (freq * (K1 + 1.0)) / denominator
    return total


def bm25_search(
    index: InvertedIndex,
    query: str | Sequence[str],
    k: int | None = 10,
) -> list[tuple[str, float]]:
    terms = _terms(query)
    if not terms:
        return []
    avgdl = index.avgdl
    if avgdl <= 0.0:
        return []

    scores: dict[str, float] = {}
    for term in terms:
        idf = index.idf_bm25(term)
        for doc_id, freq in index.postings(term).items():
            dl = index.doc_length(doc_id)
            denominator = freq + K1 * (1.0 - B + B * dl / avgdl)
            scores[doc_id] = scores.get(doc_id, 0.0) + idf * (freq * (K1 + 1.0)) / denominator

    return top_k(scores, k)


def tfidf_cosine_search(
    index: InvertedIndex,
    query: str | Sequence[str],
    k: int | None = 10,
) -> list[tuple[str, float]]:
    terms = [term for term in _terms(query) if index.df(term) > 0]
    if not terms:
        return []

    query_counts: dict[str, int] = {}
    for term in terms:
        query_counts[term] = query_counts.get(term, 0) + 1

    weights = {term: count * index.idf_tfidf(term) for term, count in query_counts.items()}
    norm = math.sqrt(math.fsum(weight * weight for weight in weights.values()))
    if norm == 0.0:
        return []
    weights = {term: weight / norm for term, weight in weights.items()}

    dots: dict[str, float] = {}
    for term, query_weight in weights.items():
        idf = index.idf_tfidf(term)
        for doc_id, freq in index.postings(term).items():
            dots[doc_id] = dots.get(doc_id, 0.0) + query_weight * (freq * idf)

    scores = {
        doc_id: dot / doc_norm
        for doc_id, dot in dots.items()
        if (doc_norm := index.doc_norm(doc_id)) > 0.0
    }
    return top_k(scores, k)
