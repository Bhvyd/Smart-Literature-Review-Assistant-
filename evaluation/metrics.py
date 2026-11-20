
from __future__ import annotations

import math
from collections.abc import Collection, Sequence


def precision_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    hits = sum(1 for doc in ranked[:k] if doc in relevant)
    return hits / k


def recall_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float | None:
    if k <= 0:
        raise ValueError("k must be positive")
    if not relevant:
        return None
    hits = sum(1 for doc in ranked[:k] if doc in relevant)
    return hits / len(relevant)


def reciprocal_rank(ranked: Sequence[str], relevant: Collection[str]) -> float:
    for rank, doc in enumerate(ranked, start=1):
        if doc in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float | None:
    if k <= 0:
        raise ValueError("k must be positive")
    if not relevant:
        return None
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc in enumerate(ranked[:k], start=1)
        if doc in relevant
    )
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return dcg / ideal


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("percentile of an empty sequence")
    if not 0 <= q <= 100:
        raise ValueError("q must be between 0 and 100")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return float(ordered[low])
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def mean(values: Sequence[float]) -> float | None:
    return math.fsum(values) / len(values) if values else None


def best_threshold(
    scored: Sequence[tuple[float, bool]],
) -> dict[str, float] | None:
    positives = sum(1 for _, answerable in scored if answerable)
    if positives == 0 or positives == len(scored):
        return None

    best: dict[str, float] | None = None
    for threshold in sorted({score for score, _ in scored}):
        accepted = [(s, a) for s, a in scored if s >= threshold]
        true_pos = sum(1 for _, a in accepted if a)
        precision = true_pos / len(accepted)
        recall = true_pos / positives
        f1 = 0.0 if true_pos == 0 else 2 * precision * recall / (precision + recall)
        candidate = {"tau": threshold, "f1": f1, "precision": precision, "recall": recall}
        if best is None or f1 >= best["f1"]:
            best = candidate
    return best


def paired_bootstrap(
    a: Sequence[float], b: Sequence[float], *, resamples: int = 5000, seed: int = 0
) -> dict[str, object]:
    import random

    if len(a) != len(b) or not a:
        raise ValueError("need two equal-length, non-empty score lists")
    diffs = [x - y for x, y in zip(a, b)]
    rng = random.Random(seed)
    n = len(diffs)
    means = [math.fsum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples)]
    low, high = round(percentile(means, 2.5), 4), round(percentile(means, 97.5), 4)
    return {
        "mean_diff": round(math.fsum(diffs) / n, 4),
        "ci95": [low, high],
        "excludes_zero": low > 0 or high < 0,
    }
