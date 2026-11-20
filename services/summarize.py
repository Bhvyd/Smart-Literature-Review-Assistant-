
from __future__ import annotations

import re
from typing import Any

from services.llm import UNTRUSTED_TEXT_RULE, LLMClient
from services.retrieval import paper_detail

SYSTEM_PROMPT = (
    "You summarise ONE research paper using ONLY the text supplied.\n"
    "Output exactly three lines and nothing else:\n"
    "Problem: <what the paper addresses>\n"
    "Method: <what it does>\n"
    "Result: <what it found or claims>\n"
    "Use at most 90 words in total. If the text does not state something, write "
    "'not stated' for that line. Do not add numbers, names, venues or claims that "
    "are not in the text."
) + UNTRUSTED_TEXT_RULE

MAX_SECTION_CHARS = 1800
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


class PaperNotFound(LookupError):
    pass


def _numbers(text: str) -> set[str]:
    return {match.group(0).replace(",", ".") for match in _NUMBER.finditer(text)}


def unsupported_numbers(summary: str, source: str) -> list[str]:
    allowed = _numbers(source)
    return sorted(number for number in _numbers(summary) if number not in allowed)


def build_source(conn: Any, paper_id: str) -> tuple[str, list[str], dict[str, Any]]:
    detail = paper_detail(conn, paper_id)
    if detail is None:
        raise PaperNotFound(paper_id)

    parts = [f"Title: {detail['title']}"]
    sources = ["title"]
    if detail["abstract"]:
        parts.append(f"Abstract: {detail['abstract']}")
        sources.append("abstract")

    if detail["has_fulltext"]:
        rows = conn.execute(
            "SELECT seq, text FROM chunks WHERE paper_id=? ORDER BY seq", (paper_id,)
        ).fetchall()
        picks: list[tuple[str, Any]] = []
        if len(rows) > 1:
            picks.append(("introduction", rows[1]))
        if len(rows) > 2:
            picks.append(("conclusion", rows[-1]))
        for label, row in picks:
            parts.append(f"{label.capitalize()} excerpt: {row['text'][:MAX_SECTION_CHARS]}")
            sources.append(f"{label} (chunk {row['seq']})")

    return "\n\n".join(parts), sources, detail


def paper_summary(conn: Any, paper_id: str, *, llm: LLMClient) -> dict[str, Any]:
    source_text, sources, detail = build_source(conn, paper_id)
    response = llm.complete(SYSTEM_PROMPT, source_text, max_output_tokens=1200)
    return {
        "paper_id": paper_id,
        "title": detail["title"],
        "summary": response.text,
        "model": response.model,
        "cached": response.cached,
        "sources": sources,
        "unsupported_numbers": unsupported_numbers(response.text, source_text),
        "ai_generated": True,
    }
