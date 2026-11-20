
from __future__ import annotations

import re

_ABS_URL = re.compile(r"^https?://(?:www\.)?arxiv\.org/(?:abs|pdf)/", re.IGNORECASE)
_ARXIV_PREFIX = re.compile(r"^arxiv:\s*", re.IGNORECASE)
_VERSION = re.compile(r"v\d+$", re.IGNORECASE)
_ARXIV_NEW = re.compile(r"^\d{4}\.\d{4,5}$")
_ARXIV_OLD = re.compile(r"^[a-z][a-z\-]*(?:\.[A-Z]{2})?/\d{7}$")
_DOI_PREFIX = re.compile(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", re.IGNORECASE)
_DOI_SHAPE = re.compile(r"^10\.\d{4,9}/\S+$")
_OPENALEX_ID = re.compile(r"(W\d+)$")


def normalise_arxiv_id(raw: str | None) -> str | None:
    if not raw:
        return None
    value = raw.strip()
    value = _ABS_URL.sub("", value)
    value = _ARXIV_PREFIX.sub("", value)
    if value.lower().endswith(".pdf"):
        value = value[:-4]
    value = _VERSION.sub("", value.strip())

    if _ARXIV_NEW.match(value) or _ARXIV_OLD.match(value):
        return value
    return None


def arxiv_pdf_url(arxiv_id: str) -> str:
    return f"https://arxiv.org/pdf/{arxiv_id}"


def normalise_doi(raw: str | None) -> str | None:
    if not raw:
        return None
    value = _DOI_PREFIX.sub("", raw.strip()).strip().rstrip(".")
    value = value.lower()
    return value if _DOI_SHAPE.match(value) else None


def openalex_work_id(raw: str | None) -> str | None:
    if not raw:
        return None
    match = _OPENALEX_ID.search(raw.strip())
    return match.group(1) if match else None


def arxiv_to_openalex_filter(arxiv_id: str) -> str:
    return f"ids.arxiv:{arxiv_id}"
