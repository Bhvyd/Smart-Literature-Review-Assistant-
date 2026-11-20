
import pytest

from ingest.ids import (
    arxiv_pdf_url,
    arxiv_to_openalex_filter,
    normalise_arxiv_id,
    normalise_doi,
    openalex_work_id,
)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("2402.01788", "2402.01788"),
        ("2402.01788v3", "2402.01788"),
        ("arXiv:2402.01788", "2402.01788"),
        ("arxiv: 2402.01788", "2402.01788"),
        ("https://arxiv.org/abs/2402.01788v2", "2402.01788"),
        ("http://arxiv.org/pdf/2402.01788", "2402.01788"),
        ("https://arxiv.org/pdf/2402.01788.pdf", "2402.01788"),
        (" 2402.01788 ", "2402.01788"),
        ("hep-th/9901001v1", "hep-th/9901001"),
        ("math.GT/0309136", "math.GT/0309136"),
        ("", None),
        (None, None),
        ("not-an-id", None),
        ("2402.017", None),
    ],
)
def test_arxiv_ids_normalise(raw, expected):
    assert normalise_arxiv_id(raw) == expected


def test_arxiv_pdf_url():
    assert arxiv_pdf_url("2402.01788") == "https://arxiv.org/pdf/2402.01788"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("10.18653/v1/N19-1423", "10.18653/v1/n19-1423"),
        ("https://doi.org/10.18653/V1/N19-1423", "10.18653/v1/n19-1423"),
        ("doi:10.1000/ABC", "10.1000/abc"),
        ("", None),
        (None, None),
        ("no-doi-here", None),
    ],
)
def test_dois_normalise(raw, expected):
    assert normalise_doi(raw) == expected


def test_openalex_work_id():
    assert openalex_work_id("https://openalex.org/W2626778328") == "W2626778328"
    assert openalex_work_id("W2626778328") == "W2626778328"
    assert openalex_work_id(None) is None
    assert openalex_work_id("https://openalex.org/") is None


def test_openalex_filter_uses_bare_id():
    assert arxiv_to_openalex_filter("2402.01788") == "ids.arxiv:2402.01788"
