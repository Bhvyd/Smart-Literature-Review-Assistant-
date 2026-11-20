
from __future__ import annotations

import time
from collections.abc import Callable, Mapping

import httpx

from .cache import Cache


class FetchError(Exception):
    pass


class RateLimited(Exception):
    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


def _retry_after(response: httpx.Response) -> float | None:
    raw = response.headers.get("retry-after")
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


class HttpClient:
    def __init__(
        self,
        cache: Cache,
        *,
        user_agent: str,
        timeout: float = 30.0,
        max_retries: int = 3,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.cache = cache
        self.user_agent = user_agent
        self.timeout = timeout
        self.max_retries = max(1, max_retries)
        self.sleep = sleep
        self.network_requests = 0
        self._client = httpx.Client(
            timeout=timeout,
            transport=transport,
            follow_redirects=True,
            headers={"User-Agent": user_agent},
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> HttpClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _resolve(self, url: str, params: Mapping[str, str] | None) -> str:
        if not params:
            return url
        return str(httpx.URL(url).copy_merge_params(params))

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
        use_cache: bool = True,
    ) -> bytes:
        resolved = self._resolve(url, params)

        if use_cache:
            cached = self.cache.get(resolved)
            if cached is not None:
                return cached

        last_error = "unknown error"
        for attempt in range(self.max_retries):
            self.network_requests += 1
            try:
                response = self._client.get(resolved, headers=dict(headers or {}))
            except httpx.HTTPError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            else:
                if response.status_code == 200:
                    body = response.content
                    if use_cache:
                        self.cache.put(resolved, body)
                    return body
                if response.status_code == 429:
                    raise RateLimited(
                        f"rate limited by {httpx.URL(resolved).host}", _retry_after(response)
                    )
                if 400 <= response.status_code < 500:
                    raise FetchError(
                        f"HTTP {response.status_code} for {resolved}"
                    )
                last_error = f"HTTP {response.status_code}"

            if attempt < self.max_retries - 1:
                self.sleep(2.0**attempt)

        raise FetchError(f"giving up on {resolved}: {last_error}")
