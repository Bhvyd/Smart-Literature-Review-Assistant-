
import math

import pytest

from evaluation.metrics import (
    best_threshold,
    mean,
    ndcg_at_k,
    percentile,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

RANKED = ["a", "b", "c", "d", "e"]


def test_precision_counts_hits_over_k_even_when_results_are_short():
    assert precision_at_k(RANKED, {"a", "c", "z"}, 3) == pytest.approx(2 / 3)
    assert precision_at_k(["a"], {"a"}, 10) == pytest.approx(0.1)


def test_recall_is_hits_over_all_relevant_and_undefined_without_any():
    assert recall_at_k(RANKED, {"a", "c", "z"}, 3) == pytest.approx(2 / 3)
    assert recall_at_k(RANKED, set(), 3) is None


def test_reciprocal_rank_uses_the_first_relevant_result():
    assert reciprocal_rank(RANKED, {"c", "e"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(RANKED, {"zz"}) == 0.0


def test_ndcg_matches_the_hand_derivation():
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    ideal = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(RANKED, {"b", "d"}, 5) == pytest.approx(dcg / ideal)


def test_a_perfect_ranking_has_ndcg_one_and_no_relevant_items_is_undefined():
    assert ndcg_at_k(["x", "y", "z"], {"x", "y"}, 3) == pytest.approx(1.0)
    assert ndcg_at_k(RANKED, set(), 3) is None


def test_k_must_be_positive():
    with pytest.raises(ValueError):
        precision_at_k(RANKED, {"a"}, 0)


def test_percentiles_interpolate_between_order_statistics():
    values = [10, 20, 30, 40]
    assert percentile(values, 50) == pytest.approx(25.0)
    assert percentile(values, 0) == 10 and percentile(values, 100) == 40
    assert percentile(values, 95) == pytest.approx(38.5)
    with pytest.raises(ValueError):
        percentile([], 50)


def test_mean_of_nothing_is_none_not_zero():
    assert mean([]) is None
    assert mean([1.0, 2.0]) == 1.5


def test_the_threshold_separates_answerable_from_unanswerable_questions():
    scored = [(0.9, True), (0.8, True), (0.7, True), (0.3, False), (0.2, False)]
    best = best_threshold(scored)
    assert best["tau"] == 0.7 and best["f1"] == 1.0


def test_threshold_needs_both_classes():
    assert best_threshold([(0.9, True), (0.8, True)]) is None
    assert best_threshold([]) is None
