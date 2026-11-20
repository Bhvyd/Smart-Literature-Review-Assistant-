
import json
from pathlib import Path

import httpx
import pytest

from ingest.cache import Cache
from ingest.http import HttpClient, RateLimited
from ingest.openalex import (
    lookup_by_arxiv,
    lookup_by_dois,
    parse_work,
    titles_for,
)

FIXTURE = Path(__file__).with_name("openalex_fixture_work.json")


def _client(tmp_path, handler):
    return HttpClient(Cache(tmp_path), user_agent="t", transport=httpx.MockTransport(handler))


def test_parse_work_reads_real_fields():
    work = parse_work(json.loads(FIXTURE.read_text(encoding="utf-8")))
    assert work.openalex_id == "W2626778328"
    assert work.title == "Attention Is All You Need"
    assert work.year == 2017
    assert work.cited_by_count == 136165
    assert work.doi == "10.48550/arxiv.1706.03762"
    assert work.referenced_works
    assert all(w.startswith("W") for w in work.referenced_works)
    assert work.venue and "arXiv" in work.venue


def test_parse_work_tolerates_missing_fields():
    work = parse_work({"id": "https://openalex.org/W1", "display_name": "X"})
    assert work.openalex_id == "W1"
    assert work.cited_by_count is None
    assert work.referenced_works == ()
    assert work.doi is None
    assert work.venue is None


def test_parse_work_rejects_a_payload_without_an_id():
    with pytest.raises(ValueError):
        parse_work({"display_name": "no id"})


def test_lookup_by_arxiv_returns_none_when_not_found(tmp_path):
    def handler(request):
        return httpx.Response(200, content=b'{"meta": {"count": 0}, "results": []}')

    assert lookup_by_arxiv(_client(tmp_path, handler), "2402.01788") is None


def test_lookup_by_arxiv_builds_the_expected_filter(tmp_path):
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        return httpx.Response(200, content=b'{"results": [' + FIXTURE.read_bytes() + b"]}")

    lookup_by_arxiv(_client(tmp_path, handler), "1706.03762")
    assert "filter=ids.arxiv%3A1706.03762" in seen["url"]


def test_lookup_by_arxiv_propagates_rate_limit(tmp_path):
    def handler(request):
        return httpx.Response(429, headers={"Retry-After": "1"})

    client = HttpClient(
        Cache(tmp_path),
        user_agent="t",
        max_retries=1,
        transport=httpx.MockTransport(handler),
        sleep=lambda _: None,
    )
    with pytest.raises(RateLimited):
        lookup_by_arxiv(client, "2402.01788")


def test_lookup_by_dois_maps_normalised_doi_to_work(tmp_path):
    payload = {"results": [json.loads(FIXTURE.read_text(encoding="utf-8"))]}

    def handler(request):
        return httpx.Response(200, content=json.dumps(payload).encode())

    works = lookup_by_dois(_client(tmp_path, handler), ["10.48550/arxiv.1706.03762"])
    assert works["10.48550/arxiv.1706.03762"].openalex_id == "W2626778328"


def test_titles_for_maps_ids_to_titles(tmp_path):
    payload = {
        "results": [
            {"id": "https://openalex.org/W1", "display_name": "One"},
            {"id": "https://openalex.org/W2", "display_name": "Two"},
        ]
    }

    def handler(request):
        return httpx.Response(200, content=json.dumps(payload).encode())

    assert titles_for(_client(tmp_path, handler), ["W1", "W2"]) == {"W1": "One", "W2": "Two"}


def test_titles_for_empty_input_makes_no_request(tmp_path):
    def handler(request):
        pytest.fail("must not call the API for an empty id list")

    assert titles_for(_client(tmp_path, handler), []) == {}
