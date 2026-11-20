
import pytest

from rag_core.fuse import min_max_normalise, rrf, weighted_fusion


def test_min_max_normalise():
    assert min_max_normalise({"a": 1.0, "b": 0.0}) == {"a": 1.0, "b": 0.0}
    assert min_max_normalise({"a": 5.0, "b": 5.0}) == {"a": 0.0, "b": 0.0}
    assert min_max_normalise({}) == {}


def test_min_max_normalise_scales_linearly():
    out = min_max_normalise({"a": 0.0, "b": 5.0, "c": 10.0})
    assert out["a"] == pytest.approx(0.0)
    assert out["b"] == pytest.approx(0.5)
    assert out["c"] == pytest.approx(1.0)


def test_weighted_fusion_respects_weights():
    left = {"a": 1.0, "b": 0.0}
    right = {"a": 0.0, "b": 1.0}
    assert weighted_fusion([left, right], [1.0, 0.0]) == [("a", 1.0), ("b", 0.0)]
    assert [d for d, _ in weighted_fusion([left, right], [0.5, 0.5])] == ["a", "b"]


def test_weighted_fusion_single_valued_source_contributes_nothing():
    fused = dict(weighted_fusion([{"a": 1.0}, {"b": 1.0}], [0.5, 0.5]))
    assert fused == {"a": 0.0, "b": 0.0}


def test_weighted_fusion_rejects_length_mismatch():
    with pytest.raises(ValueError):
        weighted_fusion([{"a": 1.0}], [0.5, 0.5])


def test_weighted_fusion_with_no_sources_is_empty():
    assert weighted_fusion([], []) == []


def test_rrf_rank_positions():
    hits = rrf([["a", "b", "c"], ["c", "b", "a"]])
    assert [d for d, _ in hits] == ["a", "c", "b"]
    scores = dict(hits)
    assert scores["a"] == pytest.approx(1 / 61 + 1 / 63)
    assert scores["b"] == pytest.approx(2 / 62)
    assert scores["c"] == pytest.approx(1 / 63 + 1 / 61)


def test_rrf_respects_k_and_top_k():
    hits = rrf([["a", "b"]], k=0, top_k=1)
    assert len(hits) == 1
    assert hits[0][0] == "a"


def test_rrf_document_in_one_list_only():
    hits = dict(rrf([["a"], ["b"]], k=60))
    assert hits["a"] == pytest.approx(1 / 61)
    assert hits["b"] == pytest.approx(1 / 61)


def test_rrf_empty_inputs():
    assert rrf([]) == []
    assert rrf([[], []]) == []


def test_rrf_without_top_k_returns_everything():
    assert len(rrf([["a", "b", "c"]])) == 3
