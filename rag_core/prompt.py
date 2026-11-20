
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

DEFAULT_MAX_CHARS = 6000

ANSWER_RULE = (
    "Answer using only the context above, citing sources as [n]. "
    "If the context does not answer the question, say so."
)


@dataclass(frozen=True)
class Passage:
    chunk_id: str
    paper_id: str
    text: str
    title: str = ""


@dataclass(frozen=True)
class PromptBundle:
    text: str
    citations: dict[int, str]
    papers: dict[int, str]
    used: int


def _slot_line(slot: int, passage: Passage) -> str:
    return f"[{slot}] ({passage.title})" if passage.title else f"[{slot}]"


def _render(header: str, blocks: Sequence[str], tail: str) -> str:
    return header + "\n\n".join(blocks) + tail


def build_prompt(
    query: str,
    passages: Sequence[Passage],
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> PromptBundle:
    items = list(passages)
    if not items:
        raise ValueError("build_prompt requires at least one passage")

    header = "Context:\n"
    tail = f"\n\nQuestion: {query}\n\n{ANSWER_RULE}"

    blocks: list[str] = []
    citations: dict[int, str] = {}
    papers: dict[int, str] = {}

    for slot, passage in enumerate(items, start=1):
        line = _slot_line(slot, passage)
        block = f"{line}\n{passage.text}"

        if len(_render(header, [*blocks, block], tail)) <= max_chars:
            blocks.append(block)
            citations[slot] = passage.chunk_id
            papers[slot] = passage.paper_id
            continue

        if not blocks:
            available = max_chars - len(header) - len(tail) - len(line) - 1
            blocks.append(f"{line}\n{passage.text[: max(0, available)]}")
            citations[slot] = passage.chunk_id
            papers[slot] = passage.paper_id
        break

    return PromptBundle(
        text=_render(header, blocks, tail),
        citations=citations,
        papers=papers,
        used=len(blocks),
    )
