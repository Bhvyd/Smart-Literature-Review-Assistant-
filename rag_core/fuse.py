
from __future__ import annotations

from collections.abc import Mapping, Sequence

from .select import top_k as _top_k

DEFAULT_RRF_K = 60


def min_max_normalise(scores: Mapping[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    low = min(scores.values())
    high = max(scores.values())
    if high == low:
        return {doc_id: 0.0 for doc_id in scores}
    span = high - low
    return {doc_id: (score - low) / span for doc_id, score in scores.items()}


def weighted_fusion(
    sources: Sequence[Mapping[str, float]],
    weights: Sequence[float],
    k: int | None = None,
) -> list[tuple[str, float]]:
    if len(sources) != len(weights):
        raise ValueError(
            f"weights length {len(weights)} does not match sources length {len(sources)}"
        )

    fused: dict[str, float] = {}
    for source, weight in zip(sources, weights, strict=True):
        for doc_id, score in min_max_normalise(source).items():
            fused[doc_id] = fused.get(doc_id, 0.0) + weight * score
    return _top_k(fused, k)


def rrf(
    rankings: Sequence[Sequence[str]],
    k: int = DEFAULT_RRF_K,
    top_k: int | None = None,
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return _top_k(scores, top_k)
