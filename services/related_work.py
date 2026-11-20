
from __future__ import annotations

import re
from typing import Any

from services.llm import UNTRUSTED_TEXT_RULE, LLMClient
from services.retrieval import paper_details

MAX_PAPERS = 12
MIN_PAPERS = 2
ABSTRACT_CHARS = 900
LABEL = "AI-assisted draft - verify every sentence against its source before use"

SYSTEM_PROMPT = (
    "You write the related-work paragraph of a research paper.\n"
    "Write ONE paragraph of 150 to 250 words, no headings, no lists.\n"
    "Cite papers in the text as [Author, Year] using ONLY the keys given in square "
    "brackets before each paper. Put the key right after the claim it supports.\n"
    "State only what the supplied abstracts state. Do not invent results, numbers or "
    "relationships between papers that the abstracts do not show.\n"
    "Group related contributions and move roughly from earlier to later work."
) + UNTRUSTED_TEXT_RULE

_KEY = re.compile(r"\[([^\[\]]+)\]")


class RelatedWorkError(ValueError):
    pass


def author_year_key(authors: list[str], year: int | None) -> str:
    surname = authors[0].split()[-1] if authors else "Unknown"
    return f"{surname}, {year}" if year else f"{surname}, n.d."


def unique_keys(rows: list[dict[str, Any]]) -> list[str]:
    base = [author_year_key(row["authors"], row["year"]) for row in rows]
    totals = {key: base.count(key) for key in base}
    used: dict[str, int] = {}
    keys: list[str] = []
    for key in base:
        if totals[key] == 1:
            keys.append(key)
            continue
        position = used.get(key, 0)
        used[key] = position + 1
        keys.append(f"{key}{chr(ord('a') + position)}")
    return keys


def _load(conn: Any, collection_id: str, paper_ids: list[str]) -> list[dict[str, Any]]:
    placeholders = ",".join("?" for _ in paper_ids)
    rows = conn.execute(
        f"""
        SELECT id, title, year, citation_count FROM papers
        WHERE collection_id=? AND id IN ({placeholders})
        """,
        [collection_id, *paper_ids],
    ).fetchall()
    found = {row["id"]: row for row in rows}
    missing = [paper_id for paper_id in paper_ids if paper_id not in found]
    if missing:
        raise RelatedWorkError(f"papers not in this collection: {', '.join(missing[:3])}")
    details = paper_details(conn, paper_ids)
    ordered = sorted(
        (
            {
                "paper_id": pid,
                "title": found[pid]["title"],
                "year": found[pid]["year"],
                "citation_count": found[pid]["citation_count"] or 0,
                "abstract": details[pid]["abstract"] or "",
                "authors": details[pid]["authors"],
                "arxiv_id": details[pid]["arxiv_id"],
                "source_url": details[pid]["source_url"],
            }
            for pid in dict.fromkeys(paper_ids)
        ),
        key=lambda row: (row["year"] is None, row["year"] or 0, -row["citation_count"], row["paper_id"]),
    )
    return ordered


def draft_related_work(
    conn: Any, collection_id: str, paper_ids: list[str], *, llm: LLMClient
) -> dict[str, Any]:
    unique_ids = list(dict.fromkeys(paper_ids))
    if not MIN_PAPERS <= len(unique_ids) <= MAX_PAPERS:
        raise RelatedWorkError(
            f"choose between {MIN_PAPERS} and {MAX_PAPERS} papers; got {len(unique_ids)}"
        )
    rows = _load(conn, collection_id, unique_ids)
    keys = unique_keys(rows)

    blocks = []
    for key, row in zip(keys, rows):
        abstract = row["abstract"][:ABSTRACT_CHARS] or "(no abstract stored)"
        blocks.append(f"[{key}] {row['title']} ({row['year']}). {abstract}")
    response = llm.complete(SYSTEM_PROMPT, "\n\n".join(blocks), max_output_tokens=1800)

    paragraph = response.text.strip()
    known = set(keys)
    cited_keys: set[str] = set()
    unknown: list[str] = []
    for match in _KEY.finditer(paragraph):
        for part in (piece.strip() for piece in match.group(1).split(";")):
            if part in known:
                cited_keys.add(part)
            elif part:
                unknown.append(part)

    return {
        "paragraph": paragraph,
        "citations": [
            {
                "key": key,
                "paper_id": row["paper_id"],
                "title": row["title"],
                "year": row["year"],
                "url": row["source_url"],
            }
            for key, row in zip(keys, rows)
        ],
        "unknown_citations": sorted(set(unknown)),
        "uncited_papers": [key for key in keys if key not in cited_keys],
        "word_count": len(paragraph.split()),
        "ordered_by": "year, then citation count",
        "ai_assisted_draft": True,
        "label": LABEL,
        "model": response.model,
        "cached": response.cached,
    }
