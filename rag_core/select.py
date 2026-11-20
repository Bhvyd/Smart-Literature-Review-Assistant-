
from __future__ import annotations

import heapq
from collections.abc import Iterable, Mapping

Scored = Mapping[str, float] | Iterable[tuple[str, float]]


def _items(scores: Scored) -> list[tuple[str, float]]:
    if isinstance(scores, Mapping):
        return list(scores.items())
    return list(scores)


def top_k(scores: Scored, k: int | None) -> list[tuple[str, float]]:
    items = _items(scores)
    if not items or k is not None and k <= 0:
        return []

    if k is None or k >= len(items):
        return sorted(items, key=lambda item: (-item[1], item[0]))

    return heapq.nsmallest(k, items, key=lambda item: (-item[1], item[0]))
