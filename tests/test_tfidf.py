
import math

import pytest

from rag_core.index import InvertedIndex
from rag_core.lexical import tfidf_cosine_search

GAMMA_IDF = 1.0 + math.log(1.5)
D2_NORM = math.sqrt(1.0 + GAMMA_IDF**2)


@pytest.fixture
def index():
    idx = InvertedIndex()
    idx.add("d1", "beta beta")
    idx.add("d2", "beta gamma")
    idx.finalize()
    return idx


def test_single_term_query_against_two_term_document(index):
    hits = dict(tfidf_cosine_search(index, "gamma"))
    assert hits["d2"] == pytest.approx(GAMMA_IDF / D2_NORM)
    assert "d1" not in hits


def test_cosine_between_query_and_documents(index):
    hits = dict(tfidf_cosine_search(index, "beta"))
    assert hits["d1"] == pytest.approx(1.0)
    assert hits["d2"] == pytest.approx(1.0 / D2_NORM)


def test_ranking_is_ordered(index):
    assert [doc for doc, _ in tfidf_cosine_search(index, "beta", k=2)] == ["d1", "d2"]


def test_stopword_only_query_returns_empty(index):
    assert tfidf_cosine_search(index, "the and of") == []


def test_document_with_zero_norm_does_not_raise():
    idx = InvertedIndex()
    idx.add("empty", "the of ???")
    idx.add("d1", "beta beta")
    idx.finalize()
    hits = dict(tfidf_cosine_search(idx, "beta"))
    assert hits.get("empty", 0.0) == 0.0
    assert hits["d1"] == pytest.approx(1.0)


def test_multi_term_query_mixes_axes(index):
    query_norm = math.sqrt(1.0 + GAMMA_IDF**2)
    hits = dict(tfidf_cosine_search(index, "beta gamma"))
    expected_d2 = (1.0 / query_norm) * (1.0 / D2_NORM) + (
        GAMMA_IDF / query_norm
    ) * (GAMMA_IDF / D2_NORM)
    assert hits["d2"] == pytest.approx(expected_d2)
    assert hits["d1"] == pytest.approx(1.0 / query_norm)


def test_respects_k(index):
    assert len(tfidf_cosine_search(index, "beta", k=1)) == 1
    assert len(tfidf_cosine_search(index, "beta", k=None)) == 2


def test_all_empty_index_returns_empty():
    idx = InvertedIndex()
    idx.add("e1", "the")
    idx.finalize()
    assert tfidf_cosine_search(idx, "beta") == []
