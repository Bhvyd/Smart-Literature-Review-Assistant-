
import math

import pytest

from rag_core.index import InvertedIndex


@pytest.fixture
def index():
    idx = InvertedIndex()
    idx.add("d1", "beta beta")
    idx.add("d2", "beta gamma")
    idx.finalize()
    return idx


def test_doc_stats(index):
    assert index.n_docs == 2
    assert index.avgdl == pytest.approx(2.0)
    assert index.doc_length("d1") == 2
    assert index.df("beta") == 2
    assert index.df("gamma") == 1
    assert index.df("absent") == 0


def test_idf_values(index):
    assert index.idf_bm25("beta") == pytest.approx(0.1823215567939546)
    assert index.idf_bm25("gamma") == pytest.approx(math.log(2.0))
    assert index.idf_tfidf("beta") == pytest.approx(1.0)
    assert index.idf_tfidf("gamma") == pytest.approx(1.0 + math.log(1.5))


def test_document_norms(index):
    assert index.doc_norm("d1") == pytest.approx(2.0)
    assert index.doc_norm("d2") == pytest.approx(math.sqrt(1.0 + (1.0 + math.log(1.5)) ** 2))


def test_postings_and_tf(index):
    assert dict(index.postings("beta")) == {"d1": 2, "d2": 1}
    assert dict(index.postings("gamma")) == {"d2": 1}
    assert dict(index.postings("absent")) == {}


def test_doc_ids_are_listed(index):
    assert index.doc_ids() == ["d1", "d2"]


def test_duplicate_doc_id_raises():
    idx = InvertedIndex()
    idx.add("d1", "alpha")
    with pytest.raises(ValueError):
        idx.add("d1", "beta")


def test_add_after_finalize_raises(index):
    with pytest.raises(RuntimeError):
        index.add("d3", "delta")


def test_queries_before_finalize_raise():
    idx = InvertedIndex()
    idx.add("d1", "alpha")
    with pytest.raises(RuntimeError):
        idx.idf_bm25("alpha")


def test_document_with_no_indexable_tokens_is_safe():
    idx = InvertedIndex()
    idx.add("empty", "the and of ???")
    idx.add("d1", "alpha")
    idx.finalize()
    assert idx.n_docs == 2
    assert idx.doc_length("empty") == 0
    assert idx.doc_norm("empty") == 0.0
    assert idx.avgdl == pytest.approx(0.5)


def test_all_empty_index_finalizes_without_dividing_by_zero():
    idx = InvertedIndex()
    idx.add("e1", "the")
    idx.add("e2", "of")
    idx.finalize()
    assert idx.avgdl == 0.0
    assert idx.n_docs == 2


def test_finalize_is_idempotent():
    idx = InvertedIndex()
    idx.add("d1", "alpha")
    idx.finalize()
    idx.finalize()
    assert idx.n_docs == 1


def test_accepts_pre_tokenised_input():
    idx = InvertedIndex()
    idx.add("d1", ["alpha", "alpha", "beta"])
    idx.finalize()
    assert dict(idx.postings("alpha")) == {"d1": 2}
