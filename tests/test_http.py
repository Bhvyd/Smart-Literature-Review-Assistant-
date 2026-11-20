
import httpx
import pytest

from ingest.cache import Cache
from ingest.http import FetchError, HttpClient, RateLimited


def test_cache_roundtrip(tmp_path):
    cache = Cache(tmp_path)
    assert cache.get("https://example.org/a") is None
    cache.put("https://example.org/a", b"body")
    assert cache.get("https://example.org/a") == b"body"


def test_cache_key_is_deterministic_across_instances(tmp_path):
    Cache(tmp_path).put("https://example.org/a", b"x")
    assert Cache(tmp_path).get("https://example.org/a") == b"x"


def test_cache_paths_are_fanned_out(tmp_path):
    cache = Cache(tmp_path)
    path = cache.path_for("https://example.org/a")
    assert path.parent != tmp_path
    assert path.parent.parent == tmp_path


def test_client_caches_and_does_not_refetch(tmp_path):
    calls = []

    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(200, content=b"fresh")

    client = HttpClient(Cache(tmp_path), user_agent="t", transport=httpx.MockTransport(handler))
    assert client.get("https://example.org/a") == b"fresh"
    assert client.get("https://example.org/a") == b"fresh"
    assert len(calls) == 1


def test_client_sends_user_agent(tmp_path):
    seen = {}

    def handler(request):
        seen["ua"] = request.headers.get("user-agent")
        return httpx.Response(200, content=b"ok")

    client = HttpClient(
        Cache(tmp_path), user_agent="litreview/0.1", transport=httpx.MockTransport(handler)
    )
    client.get("https://example.org/a", use_cache=False)
    assert seen["ua"] == "litreview/0.1"


def test_different_params_are_different_cache_entries(tmp_path):
    calls = []

    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(200, content=b"ok")

    client = HttpClient(Cache(tmp_path), user_agent="t", transport=httpx.MockTransport(handler))
    client.get("https://example.org/a", params={"q": "one"})
    client.get("https://example.org/a", params={"q": "two"})
    assert len(calls) == 2


def test_retries_then_succeeds(tmp_path):
    attempts = {"n": 0}

    def handler(request):
        attempts["n"] += 1
        if attempts["n"] < 3:
            return httpx.Response(503)
        return httpx.Response(200, content=b"ok")

    client = HttpClient(
        Cache(tmp_path),
        user_agent="t",
        transport=httpx.MockTransport(handler),
        sleep=lambda _: None,
    )
    assert client.get("https://example.org/a") == b"ok"
    assert attempts["n"] == 3


def test_gives_up_after_max_retries(tmp_path):
    def handler(request):
        return httpx.Response(500)

    client = HttpClient(
        Cache(tmp_path),
        user_agent="t",
        max_retries=2,
        transport=httpx.MockTransport(handler),
        sleep=lambda _: None,
    )
    with pytest.raises(FetchError):
        client.get("https://example.org/a")


def test_404_fails_without_retrying(tmp_path):
    attempts = {"n": 0}

    def handler(request):
        attempts["n"] += 1
        return httpx.Response(404)

    client = HttpClient(
        Cache(tmp_path),
        user_agent="t",
        transport=httpx.MockTransport(handler),
        sleep=lambda _: None,
    )
    with pytest.raises(FetchError):
        client.get("https://example.org/a")
    assert attempts["n"] == 1


def test_429_raises_rate_limited_and_does_not_cache_the_error(tmp_path):
    def handler(request):
        return httpx.Response(429, headers={"Retry-After": "7"})

    cache = Cache(tmp_path)
    client = HttpClient(
        cache,
        user_agent="t",
        max_retries=1,
        transport=httpx.MockTransport(handler),
        sleep=lambda _: None,
    )
    with pytest.raises(RateLimited) as excinfo:
        client.get("https://example.org/a")
    assert excinfo.value.retry_after == 7.0
    assert cache.get("https://example.org/a") is None


def test_transport_error_is_retried_then_raised(tmp_path):
    attempts = {"n": 0}

    def handler(request):
        attempts["n"] += 1
        raise httpx.ConnectError("boom")

    client = HttpClient(
        Cache(tmp_path),
        user_agent="t",
        max_retries=2,
        transport=httpx.MockTransport(handler),
        sleep=lambda _: None,
    )
    with pytest.raises(FetchError):
        client.get("https://example.org/a")
    assert attempts["n"] == 2
