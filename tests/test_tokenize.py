
import pytest

from rag_core.tokenize import normalize, tokenize


def test_lowercases_and_drops_stopwords():
    assert tokenize("The Quick Brown Fox") == ["quick", "brown", "fox"]


def test_keeps_hyphenated_and_decimal_tokens():
    assert tokenize("BERT-base and state-of-the-art") == ["bert-base", "state-of-the-art"]
    assert tokenize("version 3.5 of 2402.01788") == ["version", "3.5", "2402.01788"]


def test_folds_diacritics():
    assert tokenize("café naïve") == ["cafe", "naive"]
    assert normalize("Café") == "cafe"


def test_contractions_do_not_leave_fragments():
    assert tokenize("Don't stop") == ["stop"]


def test_repeated_tokens_are_preserved():
    assert tokenize("beta beta gamma") == ["beta", "beta", "gamma"]


@pytest.mark.parametrize("text", ["", "   ", "the and of", "???", "!!!", "—"])
def test_empty_and_stopword_only_queries_yield_no_tokens(text):
    assert tokenize(text) == []


def test_cjk_without_spaces_yields_no_tokens():
    assert tokenize("文献综述") == []


def test_stopwords_can_be_kept():
    assert tokenize("the quick", drop_stopwords=False) == ["the", "quick"]


def test_min_length_is_respected():
    assert tokenize("a b c", drop_stopwords=False) == []
    assert tokenize("ab c", drop_stopwords=False) == ["ab"]


def test_accepts_already_normalised_input():
    assert tokenize("cafe") == ["cafe"]
