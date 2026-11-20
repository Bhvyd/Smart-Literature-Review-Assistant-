
from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass

from .http import HttpClient
from .ids import arxiv_pdf_url, normalise_arxiv_id

API_URL = "https://export.arxiv.org/api/query"
ATOM = {"a": "http://www.w3.org/2005/Atom"}

ARXIV_MIN_INTERVAL = 3.0


@dataclass(frozen=True)
class ArxivRecord:
    arxiv_id: str
    title: str
    abstract: str
    authors: tuple[str, ...]
    published: str
    pdf_url: str


def _text(node: ET.Element | None) -> str:
    if node is None or node.text is None:
        return ""
    return " ".join(node.text.split())


def parse_feed(xml_bytes: bytes) -> list[ArxivRecord]:
    root = ET.fromstring(xml_bytes)
    records: list[ArxivRecord] = []

    for entry in root.findall("a:entry", ATOM):
        arxiv_id = normalise_arxiv_id(_text(entry.find("a:id", ATOM)))
        if arxiv_id is None:
            continue

        authors = tuple(
            name
            for author in entry.findall("a:author", ATOM)
            if (name := _text(author.find("a:name", ATOM)))
        )

        records.append(
            ArxivRecord(
                arxiv_id=arxiv_id,
                title=_text(entry.find("a:title", ATOM)),
                abstract=_text(entry.find("a:summary", ATOM)),
                authors=authors,
                published=_text(entry.find("a:published", ATOM)),
                pdf_url=arxiv_pdf_url(arxiv_id),
            )
        )

    return records


_OPERATORS = {"AND", "OR", "ANDNOT"}


def build_query(topic: str) -> str:
    terms: list[str] = []
    for phrase, word in re.findall(r'"([^"]+)"|(\S+)', topic):
        if phrase:
            cleaned = " ".join(phrase.split())
            if cleaned:
                terms.append(f'all:"{cleaned}"')
            continue
        token = re.sub(r"[^\w\-.]", "", word)
        if token and token.upper() not in _OPERATORS:
            terms.append(f"all:{token}")
    if not terms:
        raise ValueError("query has no searchable words")
    return " AND ".join(terms)


def search(
    client: HttpClient,
    query: str,
    limit: int = 100,
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> list[ArxivRecord]:
    body = client.get(
        API_URL,
        params={
            "search_query": build_query(query),
            "start": "0",
            "max_results": str(limit),
            "sortBy": "relevance",
        },
    )
    sleep(ARXIV_MIN_INTERVAL)
    return parse_feed(body)
