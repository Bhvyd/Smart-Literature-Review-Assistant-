
import math

import pytest

from rag_core.dense import cosine, cosine_search, dot, l2_norm


def test_dot_and_norm():
    assert dot([1, 2, 3], [4, 5, 6]) == 32
    assert l2_norm([3, 4]) == pytest.approx(5.0)


def test_cosine_known_values():
    assert cosine([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)
    assert cosine([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine([1, 2], [2, 4]) == pytest.approx(1.0)
    assert cosine([1, 2, 3], [4, 5, 6]) == pytest.approx(32 / (math.sqrt(14) * math.sqrt(77)))


def test_zero_vectors_are_zero_not_division_error():
    assert cosine([0, 0], [1, 1]) == 0.0
    assert cosine([0, 0], [0, 0]) == 0.0


def test_dimension_mismatch_raises():
    with pytest.raises(ValueError):
        cosine([1, 2], [1, 2, 3])


def test_cosine_search_ranks_and_breaks_ties_deterministically():
    docs = {"b": [1, 0], "a": [1, 0], "c": [0, 1]}
    hits = cosine_search([1, 0], docs, k=3)
    assert [d for d, _ in hits] == ["a", "b"]


def test_cosine_search_excludes_non_positive_scores():
    docs = {"empty": [0, 0], "real": [1, 0], "opposite": [-1, 0]}
    hits = cosine_search([1, 0], docs, k=5)
    assert [d for d, _ in hits] == ["real"]


def test_cosine_search_respects_k_and_empty_corpus():
    docs = {"a": [1, 0], "b": [0.9, 0.1]}
    assert len(cosine_search([1, 0], docs, k=1)) == 1
    assert cosine_search([1, 0], {}, k=1) == []


def test_zero_query_returns_nothing():
    assert cosine_search([0, 0], {"a": [1, 0]}, k=5) == []
