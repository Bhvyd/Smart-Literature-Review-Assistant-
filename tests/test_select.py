
import pytest

from rag_core.select import top_k


def test_orders_by_score_then_id():
    assert top_k({"a": 1.0, "b": 1.0, "c": 2.0}, k=2) == [("c", 2.0), ("a", 1.0)]


def test_k_none_returns_all_sorted():
    assert top_k({"b": 0.1, "a": 0.2}, k=None) == [("a", 0.2), ("b", 0.1)]


@pytest.mark.parametrize("k", [0, -1])
def test_non_positive_k_returns_empty(k):
    assert top_k({"a": 1.0}, k=k) == []


def test_k_larger_than_corpus():
    assert len(top_k({"a": 1.0}, k=10)) == 1


def test_accepts_pairs_iterable():
    assert top_k([("a", 1.0), ("b", 2.0)], k=1) == [("b", 2.0)]


def test_empty_input():
    assert top_k({}, k=3) == []


def test_ordering_is_independent_of_insertion_order():
    forward = top_k({"a": 1.0, "b": 1.0, "c": 1.0}, k=3)
    backward = top_k({"c": 1.0, "b": 1.0, "a": 1.0}, k=3)
    assert forward == backward == [("a", 1.0), ("b", 1.0), ("c", 1.0)]


def test_bounded_and_unbounded_agree_on_prefix():
    scores = {"a": 3.0, "b": 2.0, "c": 1.0}
    assert top_k(scores, k=2) == top_k(scores, k=None)[:2]
