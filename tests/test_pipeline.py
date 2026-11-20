
import pytest

from rag_core.pipeline import Retriever


def test_build_and_lexical_search():
    r = Retriever.build([("d1", "beta beta"), ("d2", "beta gamma")])
    result = r.search("gamma", k=2)
    assert result.mode == "lexical"
    assert result.hits[0].doc_id == "d2"
    assert result.hits[0].rank == 1
    assert result.degraded == []


def test_tfidf_mode():
    r = Retriever.build([("d1", "beta beta"), ("d2", "beta gamma")])
    result = r.search("gamma", k=2, mode="tfidf")
    assert result.mode == "tfidf"
    assert result.hits[0].doc_id == "d2"


def test_dense_mode_requires_query_vector():
    r = Retriever.build([("d1", "alpha")], dense_vectors={"d1": [1.0, 0.0]})
    with pytest.raises(ValueError):
        r.search("alpha", mode="dense")


def test_dense_mode_requires_vectors_to_exist():
    r = Retriever.build([("d1", "alpha")])
    with pytest.raises(ValueError):
        r.search("alpha", mode="dense", query_vector=[1.0, 0.0])


def test_dense_mode_ranks_by_cosine():
    r = Retriever.build(
        [("d1", "alpha"), ("d2", "beta")],
        dense_vectors={"d1": [1.0, 0.0], "d2": [0.0, 1.0]},
    )
    result = r.search("ignored", mode="dense", k=1, query_vector=[1.0, 0.0])
    assert result.hits[0].doc_id == "d1"


def test_hybrid_mode_requires_both_signals():
    r = Retriever.build([("d1", "alpha")])
    with pytest.raises(ValueError):
        r.search("alpha", mode="hybrid", query_vector=[1.0])
    r2 = Retriever.build([("d1", "alpha")], dense_vectors={"d1": [1.0]})
    with pytest.raises(ValueError):
        r2.search("alpha", mode="hybrid")


def test_hybrid_fuses_lexical_and_dense():
    r = Retriever.build(
        [("d1", "alpha"), ("d2", "beta")],
        dense_vectors={"d1": [0.0, 1.0], "d2": [1.0, 0.0]},
    )
    result = r.search("alpha", k=2, mode="hybrid", query_vector=[1.0, 0.0])
    assert {h.doc_id for h in result.hits} == {"d1", "d2"}
    assert result.mode == "hybrid"


def test_unknown_mode_raises():
    r = Retriever.build([("d1", "alpha")])
    with pytest.raises(ValueError):
        r.search("alpha", mode="semantic")


def test_empty_query_returns_no_hits():
    r = Retriever.build([("d1", "alpha")])
    assert r.search("the and of").hits == []


def test_ranks_are_one_based_and_contiguous():
    r = Retriever.build([("d1", "beta beta"), ("d2", "beta gamma"), ("d3", "beta delta")])
    ranks = [hit.rank for hit in r.search("beta", k=3).hits]
    assert ranks == [1, 2, 3]


def test_build_is_deterministic():
    docs = [("d2", "beta gamma"), ("d1", "beta beta")]
    first = Retriever.build(docs).search("beta", k=2)
    second = Retriever.build(docs).search("beta", k=2)
    assert [(h.doc_id, h.score) for h in first.hits] == [
        (h.doc_id, h.score) for h in second.hits
    ]
