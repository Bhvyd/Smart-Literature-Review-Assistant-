
from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from .http import FetchError, HttpClient, RateLimited
from .ids import normalise_doi, openalex_work_id

API_URL = "https://api.openalex.org/works"
MAX_FILTER_IDS = 50


@dataclass(frozen=True)
class OpenAlexWork:
    openalex_id: str
    doi: str | None
    title: str
    year: int | None
    cited_by_count: int | None
    referenced_works: tuple[str, ...]
    venue: str | None


def _venue(payload: dict[str, Any]) -> str | None:
    location = payload.get("primary_location") or {}
    source = location.get("source") or {}
    name = source.get("display_name")
    return name or None


def parse_work(payload: dict[str, Any]) -> OpenAlexWork:
    openalex_id = openalex_work_id(payload.get("id"))
    if openalex_id is None:
        raise ValueError(f"OpenAlex work has no usable id: {payload.get('id')!r}")

    references: list[str] = []
    for raw in payload.get("referenced_works") or []:
        work_id = openalex_work_id(raw)
        if work_id is not None:
            references.append(work_id)

    return OpenAlexWork(
        openalex_id=openalex_id,
        doi=normalise_doi(payload.get("doi")),
        title=(payload.get("display_name") or "").strip(),
        year=payload.get("publication_year"),
        cited_by_count=payload.get("cited_by_count"),
        referenced_works=tuple(references),
        venue=_venue(payload),
    )


def _query(
    client: HttpClient,
    filter_value: str,
    *,
    per_page: int,
    mailto: str | None,
) -> dict[str, Any]:
    params = {"filter": filter_value, "per-page": str(per_page)}
    if mailto:
        params["mailto"] = mailto
    body = client.get(API_URL, params=params)
    return json.loads(body)


def lookup_by_arxiv(
    client: HttpClient, arxiv_id: str, *, mailto: str | None = None
) -> OpenAlexWork | None:
    payload = _query(client, f"ids.arxiv:{arxiv_id}", per_page=1, mailto=mailto)
    results = payload.get("results") or []
    return parse_work(results[0]) if results else None


def lookup_by_dois(
    client: HttpClient,
    dois: Sequence[str],
    *,
    mailto: str | None = None,
) -> dict[str, OpenAlexWork]:
    wanted = [doi for doi in (normalise_doi(d) for d in dois) if doi]
    if not wanted:
        return {}

    found: dict[str, OpenAlexWork] = {}
    for start in range(0, len(wanted), MAX_FILTER_IDS):
        batch = wanted[start : start + MAX_FILTER_IDS]
        payload = _query(
            client, "doi:" + "|".join(batch), per_page=len(batch), mailto=mailto
        )
        for raw in payload.get("results") or []:
            work = parse_work(raw)
            if work.doi:
                found[work.doi] = work
    return found


def titles_for(
    client: HttpClient,
    work_ids: Sequence[str],
    *,
    mailto: str | None = None,
    on_error: Callable[[Exception], None] | None = None,
) -> dict[str, str]:
    wanted = [work_id for work_id in work_ids if work_id]
    if not wanted:
        return {}

    titles: dict[str, str] = {}
    for start in range(0, len(wanted), MAX_FILTER_IDS):
        batch = wanted[start : start + MAX_FILTER_IDS]
        try:
            payload = _query(
                client,
                "openalex_id:" + "|".join(batch),
                per_page=len(batch),
                mailto=mailto,
            )
        except (FetchError, RateLimited) as exc:
            if on_error is None:
                raise
            on_error(exc)
            if isinstance(exc, RateLimited):
                break
            continue
        for raw in payload.get("results") or []:
            work_id = openalex_work_id(raw.get("id"))
            title = (raw.get("display_name") or "").strip()
            if work_id and title:
                titles[work_id] = title
    return titles


def _normal_title(title: str) -> str:
    return " ".join("".join(ch.lower() if ch.isalnum() else " " for ch in title).split())


def lookup_by_title(
    client: HttpClient, title: str, *, mailto: str | None = None
) -> OpenAlexWork | None:
    wanted = _normal_title(title)
    if len(wanted) < 15:
        return None
    params = {"search": title, "per-page": "5"}
    if mailto:
        params["mailto"] = mailto
    payload = json.loads(client.get(API_URL, params=params))
    for raw in payload.get("results") or []:
        if _normal_title(raw.get("display_name") or "") == wanted:
            return parse_work(raw)
    return None
