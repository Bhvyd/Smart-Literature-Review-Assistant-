
import pytest

from rag_core.index import InvertedIndex
from rag_core.lexical import bm25_score, bm25_search


@pytest.fixture
def index():
    idx = InvertedIndex()
    idx.add("d1", "beta beta")
    idx.add("d2", "beta gamma")
    idx.finalize()
    return idx


def test_bm25_hand_computed(index):
    assert bm25_score(index, "beta", "d1") == pytest.approx(0.2604593668485066)
    assert bm25_score(index, "beta", "d2") == pytest.approx(0.1823215567939546)


def test_bm25_ranks_by_term_frequency(index):
    hits = bm25_search(index, "beta", k=2)
    assert [doc for doc, _ in hits] == ["d1", "d2"]


def test_bm25_unseen_term_scores_zero(index):
    assert bm25_score(index, "zeta", "d1") == 0.0
    assert bm25_search(index, "zeta") == []


def test_bm25_stopword_only_query_returns_empty(index):
    assert bm25_search(index, "the and of") == []


def test_bm25_repeated_query_term_counts_twice(index):
    once = bm25_score(index, "beta", "d1")
    twice = bm25_score(index, "beta beta", "d1")
    assert twice == pytest.approx(2 * once)


def test_bm25_respects_k(index):
    assert len(bm25_search(index, "beta", k=1)) == 1
    assert len(bm25_search(index, "beta", k=99)) == 2


def test_bm25_all_empty_index_scores_zero():
    idx = InvertedIndex()
    idx.add("e1", "the")
    idx.add("e2", "of")
    idx.finalize()
    assert bm25_score(idx, "beta", "e1") == 0.0
    assert bm25_search(idx, "beta") == []


def test_bm25_document_without_the_term_scores_zero(index):
    assert bm25_score(index, "gamma", "d1") == 0.0


def test_bm25_accepts_pretokenised_query(index):
    assert bm25_score(index, ["beta"], "d1") == pytest.approx(
        bm25_score(index, "beta", "d1")
    )
