
import httpx
import pymupdf
import pytest

from ingest.cache import Cache
from ingest.http import HttpClient
from ingest.pdf import MIN_CHARS_PER_PAGE, extract_text, fetch_pdf


def _pdf_with(text: str, pages: int = 1) -> bytes:
    doc = pymupdf.open()
    for _ in range(pages):
        page = doc.new_page()
        y = 60.0
        for paragraph in text.split("\n"):
            for start in range(0, max(len(paragraph), 1), 70):
                page.insert_text((40, y), paragraph[start : start + 70], fontsize=9)
                y += 12
                if y > 760:
                    break
            if y > 760:
                break
    return doc.tobytes()


def test_extracts_text_and_marks_ok():
    body = "The Transformer uses self attention. " * 40
    result = extract_text(_pdf_with(body))
    assert result.quality == "ok"
    assert result.pages == 1
    assert "self attention" in result.text
    assert result.alpha_ratio > 0.55


def test_empty_pdf_is_low_quality_not_ok():
    result = extract_text(_pdf_with(""))
    assert result.quality == "low_quality"
    assert result.text == ""


def test_symbol_soup_is_low_quality():
    result = extract_text(_pdf_with("@@@@ #### $$$$ %%%% ^^^^ " * 40))
    assert result.quality == "low_quality"


def test_garbage_bytes_are_failed_not_an_exception():
    result = extract_text(b"this is not a pdf at all")
    assert result.quality == "failed"
    assert result.text == ""
    assert result.pages == 0


def test_text_is_whitespace_collapsed():
    body = "line one\nline two   three\n"
    result = extract_text(_pdf_with(body * 30))
    assert "\n\n" not in result.text
    assert "  " not in result.text


def test_multiple_pages_are_counted():
    result = extract_text(_pdf_with("page body words here " * 30, pages=3))
    assert result.pages == 3
    assert result.quality == "ok"


def test_min_chars_per_page_is_documented():
    assert MIN_CHARS_PER_PAGE == 200


def test_fetch_pdf_returns_none_on_a_404(tmp_path):
    def handler(request):
        return httpx.Response(404)

    client = HttpClient(
        Cache(tmp_path), user_agent="t", transport=httpx.MockTransport(handler), sleep=lambda _: None
    )
    assert fetch_pdf(client, "https://example.org/x.pdf") is None


def test_fetch_pdf_returns_body_on_success(tmp_path):
    def handler(request):
        return httpx.Response(200, content=b"%PDF-1.4 fake")

    client = HttpClient(Cache(tmp_path), user_agent="t", transport=httpx.MockTransport(handler))
    assert fetch_pdf(client, "https://example.org/x.pdf") == b"%PDF-1.4 fake"


def test_the_arxiv_margin_stamp_is_removed_from_extracted_text():
    text = (
        "arXiv:2501.09292v3 [cs.CL] 18 Mar 2025 " + "A real sentence about retrieval and evaluation. " * 12
    )
    result = extract_text(_pdf_with(text))
    assert "arXiv:" not in result.text and "A real sentence about retrieval" in result.text
