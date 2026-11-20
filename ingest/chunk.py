
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    seq: int
    text: str
    n_tokens: int


_ARXIV_REF = re.compile(r"arxiv preprint|arxiv:\d{4}\.\d{4,5}|arxiv\.org/abs", re.IGNORECASE)
_BIB_MARKER = re.compile(
    r"\bIn Proceedings of\b|\bProceedings of the\b|\bpp\.\s*\d+|\bJournal of\b|\bvol\.\s*\d+"
    r"|\bTransactions (?:on|of)\b|\bIn Proc\.|\bURL https?://|\bdoi:\s*10\.",
    re.IGNORECASE,
)


def looks_like_references(text: str) -> bool:
    arxiv = len(_ARXIV_REF.findall(text))
    bib = len(_BIB_MARKER.findall(text))
    return arxiv >= 3 or bib >= 4 or arxiv + bib >= 4


def word_count(text: str) -> int:
    return len(text.split())


def chunk_text(
    text: str,
    *,
    target: int = 350,
    overlap: int = 50,
    min_tokens: int = 40,
    count_tokens: Callable[[str], int] | None = None,
    max_tokens: int | None = None,
) -> list[Chunk]:
    counter = count_tokens or word_count
    words = text.split()
    if not words:
        return []

    target = max(1, target)
    overlap = max(0, overlap)

    tokens_per_word = counter(text) / len(words)
    if tokens_per_word <= 0:
        tokens_per_word = 1.0

    window = max(1, int(round(target / tokens_per_word)))
    overlap_words = max(0, int(round(overlap / tokens_per_word)))
    overlap_words = min(overlap_words, window - 1)
    step = max(1, window - overlap_words)

    chunks: list[Chunk] = []
    start = 0
    while start < len(words):
        piece = " ".join(words[start : start + window])
        chunks.append(Chunk(seq=len(chunks), text=piece, n_tokens=counter(piece)))
        if start + window >= len(words):
            break
        start += step

    if max_tokens is not None:
        chunks = _enforce_max_tokens(chunks, counter, max_tokens)

    if len(chunks) > 1 and chunks[-1].n_tokens < min_tokens:
        tail = chunks.pop()
        previous = chunks[-1]
        merged = f"{previous.text} {tail.text}"
        if max_tokens is None or counter(merged) <= max_tokens:
            chunks[-1] = Chunk(seq=previous.seq, text=merged, n_tokens=counter(merged))
        else:
            chunks.append(tail)

    return chunks


def _enforce_max_tokens(
    chunks: list[Chunk], counter: Callable[[str], int], max_tokens: int
) -> list[Chunk]:
    out: list[Chunk] = []
    for chunk in chunks:
        out.extend(_halve(chunk.text.split(), counter, max_tokens))
    return [Chunk(seq=index, text=c.text, n_tokens=c.n_tokens) for index, c in enumerate(out)]


def _halve(words: list[str], counter: Callable[[str], int], max_tokens: int) -> list[Chunk]:
    piece = " ".join(words)
    if len(words) <= 1 or counter(piece) <= max_tokens:
        return [Chunk(seq=0, text=piece, n_tokens=counter(piece))]
    middle = len(words) // 2
    return _halve(words[:middle], counter, max_tokens) + _halve(words[middle:], counter, max_tokens)
