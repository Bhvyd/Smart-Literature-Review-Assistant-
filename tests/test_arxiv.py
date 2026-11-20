
from pathlib import Path

import httpx
import pytest

from ingest.arxiv import ARXIV_MIN_INTERVAL, parse_feed, search
from ingest.cache import Cache
from ingest.http import HttpClient

FIXTURE = Path(__file__).with_name("arxiv_fixture_feed.xml")


def test_parse_feed_extracts_records():
    records = parse_feed(FIXTURE.read_bytes())
    assert len(records) == 2
    first = records[0]
    assert first.arxiv_id == "2202.09741"
    assert first.title == "Visual Attention Network"
    assert first.pdf_url == "https://arxiv.org/pdf/2202.09741"
    assert first.authors == ("Meng-Hao Guo", "Cheng-Ze Lu", "Zheng-Ning Liu", "Ming-Ming Cheng", "Shi-Min Hu")
    assert first.published.startswith("2022")


def test_parse_feed_strips_version_suffix():
    records = parse_feed(FIXTURE.read_bytes())
    assert all("v" not in r.arxiv_id for r in records)


def test_parse_feed_collapses_whitespace():
    records = parse_feed(FIXTURE.read_bytes())
    assert all("\n" not in r.title and "  " not in r.title for r in records)


def test_parse_feed_ignores_empty_feed():
    empty = b'<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>'
    assert parse_feed(empty) == []


def test_parse_feed_skips_entries_without_a_usable_id():
    payload = b"""<?xml version="1.0"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry><id>http://arxiv.org/abs/bogus</id><title>T</title><summary>S</summary></entry>
    </feed>"""
    assert parse_feed(payload) == []


def test_search_builds_the_right_url(tmp_path):
    calls = []

    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(200, content=FIXTURE.read_bytes())

    client = HttpClient(Cache(tmp_path), user_agent="t", transport=httpx.MockTransport(handler))
    records = search(client, "attention", limit=2, sleep=lambda _: None)
    assert len(records) == 2
    assert "search_query=all%3Aattention" in calls[0]
    assert "max_results=2" in calls[0]


def test_search_hits_the_cache_on_the_second_call(tmp_path):
    calls = []

    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(200, content=FIXTURE.read_bytes())

    client = HttpClient(Cache(tmp_path), user_agent="t", transport=httpx.MockTransport(handler))
    search(client, "attention", limit=2, sleep=lambda _: None)
    search(client, "attention", limit=2, sleep=lambda _: None)
    assert len(calls) == 1


def test_min_interval_is_three_seconds():
    assert ARXIV_MIN_INTERVAL == 3.0
