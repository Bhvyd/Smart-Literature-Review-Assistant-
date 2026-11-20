
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from .select import top_k


def _check_same_size(a: Sequence[float], b: Sequence[float]) -> None:
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch: {len(a)} != {len(b)}")


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    _check_same_size(a, b)
    return math.fsum(x * y for x, y in zip(a, b, strict=True))


def l2_norm(vector: Sequence[float]) -> float:
    return math.sqrt(math.fsum(value * value for value in vector))


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    _check_same_size(a, b)
    norm_a = l2_norm(a)
    norm_b = l2_norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot(a, b) / (norm_a * norm_b)


def document_norms(docs: Mapping[str, Sequence[float]]) -> dict[str, float]:
    return {doc_id: l2_norm(vector) for doc_id, vector in docs.items()}


def cosine_search(
    query: Sequence[float],
    docs: Mapping[str, Sequence[float]],
    k: int | None = 10,
    norms: Mapping[str, float] | None = None,
) -> list[tuple[str, float]]:
    query_norm = l2_norm(query)
    if query_norm == 0.0:
        return []

    scores: dict[str, float] = {}
    for doc_id, vector in docs.items():
        doc_norm = norms[doc_id] if norms is not None else l2_norm(vector)
        if doc_norm == 0.0:
            continue
        score = dot(query, vector) / (query_norm * doc_norm)
        if score > 0.0:
            scores[doc_id] = score
    return top_k(scores, k)
