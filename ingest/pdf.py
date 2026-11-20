
from __future__ import annotations

import re
from dataclasses import dataclass

import pymupdf

from .http import FetchError, HttpClient

pymupdf.TOOLS.mupdf_display_errors(False)

MIN_CHARS_PER_PAGE = 200
MAX_PDF_BYTES = 60 * 1024 * 1024
_REF_HEADING = re.compile(
    r"^[ \t]*(?:\d+\.?[ \t]*|[A-Z]\.[ \t]+)?(?:references|bibliography|literature cited)[ \t]*:?[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)
_LINE_HYPHEN = re.compile(r"(?<=[a-z])-\n(?=[a-z])")


@dataclass(frozen=True)
class ExtractedText:
    text: str
    pages: int
    alpha_ratio: float
    quality: str
    references_removed: bool = False
    front_matter_chars: int = 0


def _collapse(text: str) -> str:
    return " ".join(text.split())


def _alpha_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for char in text if char.isalpha() or char.isspace()) / len(text)


def drop_references(pages: list[str]) -> tuple[list[str], bool]:
    start_page = int(len(pages) * 0.4)
    for number in range(start_page, len(pages)):
        match = _REF_HEADING.search(pages[number])
        if match:
            kept = pages[:number] + [pages[number][: match.start()]]
            return kept, True
    return pages, False


FRONT_MATTER_WINDOW = 2500
MAX_LEADING_PLAIN_LINES = 15
FALLBACK_WINDOW = 1200
_ABSTRACT_HEADING = re.compile(
    r"^[ \t]*(?:(?:\d+|[IVX]+)\.?[ \t]+)?abstract\b"
    r"(?=[ \t]*(?:$|[:.]|[\u2013\u2014]|-(?:[ \t]|(?-i:[A-Z])|$)))",
    re.IGNORECASE | re.MULTILINE,
)
_EMAIL = re.compile(
    r"(?:[A-Za-z0-9._%+-]+\}?|\{[^}\n]*\})@(?:[A-Za-z0-9-]+\.)+(?:[a-z]{2,}|[A-Z]{2,})\b"
)
_AFFILIATION = re.compile(
    r"\b(?:universit(?:y|\u00e4t|\u00e9)|institute|laborator(?:y|ies)|department|dept\.|school of|college|"
    r"corporation|gmbh|faculty|academy)\b|\b(?:Inc|Ltd)\.",
    re.IGNORECASE,
)
_AFFILIATION_FILLER = {"of", "and", "for", "the", "in", "at", "de", "la", "du", "von", "van", "der", "e", "y", "&"}
_FOOTNOTE_MARK = r"(?:[*\u2020\u2021\u2217\u22c6\u00a7\u00b6#\u2660-\u2667\u2662]+|[\u00b9\u00b2\u00b3\u2070-\u2079]+|\d(?=\D))[ \t]*"
_FOOTNOTE_LINE = re.compile(
    r"^[ \t]*\(?(?:(?:" + _FOOTNOTE_MARK + r")+(?:equal contribution|equally contributed|these authors contributed equally|"
    r"corresponding author|correspondence|work (?:was )?done|all e-?mails?|orcid)\b"
    r"|these authors contributed equally|correspondence(?: to)?(?=[ \t]*:)"
    r"|(?:equal contribution|corresponding authors?|orcid(?:\(s\))?)\b(?=[ \t]*(?:[.:;,]|$)))"
    r"|\b\d{4}-\d{4}-\d{4}-\d{3}[\dX]\b",
    re.IGNORECASE,
)
MAX_FOOTNOTE_LINE = 400


def _is_prose_line(line: str) -> bool:
    words = line.split()
    return len(words) >= 6 and sum(w[:1].islower() for w in words) / len(words) >= 0.6


def _is_affiliation_line(line: str) -> bool:
    if len(line) > 100 or not _AFFILIATION.search(line):
        return False
    if line[0].islower() and not re.match(r"[a-z][A-Z0-9]", line):
        return False
    words = [w for w in re.findall(r"[^\W\d_]+", line) if w.lower() not in _AFFILIATION_FILLER]
    return bool(words) and sum(w[0].islower() for w in words) / len(words) <= 0.34


def _front_matter_cut(first: str) -> int:
    heading = _ABSTRACT_HEADING.search(first, 0, FRONT_MATTER_WINDOW)
    if heading:
        return heading.start()
    offset, last_marker_end, emails, affiliations, gap = 0, 0, 0, 0, 0
    for line in first.splitlines(keepends=True):
        start, offset = offset, offset + len(line)
        stripped = line.strip()
        if start >= FALLBACK_WINDOW:
            break
        if not stripped:
            continue
        if _EMAIL.search(stripped):
            emails += 1
        elif _is_affiliation_line(stripped):
            affiliations += 1
        elif _is_prose_line(stripped) or (
            last_marker_end and stripped[0].islower() and not re.match(r"[a-z][A-Z0-9]", stripped)
        ):
            break
        else:
            gap += 1
            if gap >= (3 if last_marker_end else MAX_LEADING_PLAIN_LINES):
                break
            continue
        gap = 0
        last_marker_end = offset
    return last_marker_end if (emails or affiliations >= 2) else 0


def _strip_footnotes(body: str) -> str:
    kept = []
    for line in body.splitlines(keepends=True):
        short = len(line) <= MAX_FOOTNOTE_LINE
        if short and (_FOOTNOTE_LINE.search(line.strip()) or _EMAIL.search(line)):
            continue
        kept.append(line if short else _EMAIL.sub("", line))
    return "".join(kept)


def drop_front_matter(pages: list[str]) -> tuple[list[str], int]:
    if not pages:
        return pages, 0
    first = pages[0]
    cleaned = _strip_footnotes(first[_front_matter_cut(first) :])
    return [cleaned] + pages[1:], len(first) - len(cleaned)


def extract_text(pdf_bytes: bytes, *, min_alpha_ratio: float = 0.55) -> ExtractedText:
    if len(pdf_bytes) > MAX_PDF_BYTES:
        return ExtractedText(text="", pages=0, alpha_ratio=0.0, quality="failed")
    document = None
    try:
        document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        pages = document.page_count
        if pages > MAX_PAGES:
            return ExtractedText(text="", pages=pages, alpha_ratio=0.0, quality="failed")
        raw_pages = [_LINE_HYPHEN.sub("", page.get_text()) for page in document]
    except Exception:
        return ExtractedText(text="", pages=0, alpha_ratio=0.0, quality="failed")
    finally:
        if document is not None:
            document.close()

    raw_pages, removed = drop_references(raw_pages)
    raw_pages = [_STAMP_LINE.sub(" ", page) for page in raw_pages]
    raw_pages, front_removed = drop_front_matter(raw_pages)
    text = "\n\n".join(_collapse(page) for page in raw_pages)

    ratio = _alpha_ratio(text)
    too_short = len(text.strip()) < MIN_CHARS_PER_PAGE * max(1, pages)
    if too_short or ratio < min_alpha_ratio:
        return ExtractedText(
            text=text, pages=pages, alpha_ratio=ratio, quality="low_quality",
            references_removed=removed, front_matter_chars=front_removed,
        )

    return ExtractedText(
        text=text, pages=pages, alpha_ratio=ratio, quality="ok",
        references_removed=removed, front_matter_chars=front_removed,
    )


def fetch_pdf(client: HttpClient, url: str) -> bytes | None:
    try:
        return client.get(url)
    except FetchError:
        return None


_DOI = re.compile(r"\b(10\.\d{4,9}/[^\s\"<>)\]]+)", re.IGNORECASE)
_ARXIV_STAMP = re.compile(r"arXiv:(\d{4}\.\d{4,5})(?:v\d+)?", re.IGNORECASE)
_ARXIV_OWN_STAMP = re.compile(
    r"arXiv:(\d{4}\.\d{4,5})v\d+\s*\[[a-z\-]+(?:\.[A-Za-z\-]+)?\]", re.IGNORECASE
)
_STAMP_LINE = re.compile(
    r"arXiv:\d{4}\.\d{4,5}v\d+\s*\[[^\]]+\]\s*\d{1,2}\s+[A-Z][a-z]{2,8}\s+\d{4}"
)
MAX_PAGES = 500
_ABSTRACT_START = re.compile(r"\babstract\b[\s:.—\-]*", re.IGNORECASE)
_ABSTRACT_END = re.compile(
    r"^[ \t]*(?:(?:1\.?|I\.)?[ \t]*introduction[ \t]*$|keywords?[ \t]*[:\u2014\-]|index terms|CCS concepts)",
    re.IGNORECASE | re.MULTILINE,
)
_NOT_A_TITLE = re.compile(r"^(?:arxiv|doi|https?://|www\.|preprint|under review|published)", re.IGNORECASE)
_PLACEHOLDER_TITLES = {"untitled", "title", "microsoft word", "document", "pdf"}
_PLACEHOLDER_FRAGMENTS = ("paper title", "use style", "title here", "your title", "untitled", "microsoft word")
MIN_ABSTRACT_CHARS = 200
MAX_ABSTRACT_CHARS = 2500
MIN_TITLE_CHARS = 8


@dataclass(frozen=True)
class PdfMetadata:
    title: str | None
    authors: tuple[str, ...]
    abstract: str | None
    doi: str | None
    arxiv_id: str | None


def _usable_title(title: str) -> bool:
    lowered = title.lower().strip()
    if len(title) < MIN_TITLE_CHARS or lowered in _PLACEHOLDER_TITLES:
        return False
    if _ARXIV_STAMP.match(title) or _NOT_A_TITLE.match(title):
        return False
    if re.search(r"\.(?:docx?|pdf|tex|odt|rtf|txt)$", lowered):
        return False
    return not any(fragment in lowered for fragment in _PLACEHOLDER_FRAGMENTS)


def _title_from_layout(page) -> str | None:
    try:
        blocks = page.get_text("dict")["blocks"]
    except Exception:
        return None
    best_size = 0.0
    lines: list[tuple[float, float, str]] = []
    for block in blocks:
        for line in block.get("lines", []):
            spans = [s for s in line.get("spans", []) if s.get("text", "").strip()]
            if not spans:
                continue
            direction = line.get("dir", (1.0, 0.0))
            if abs(direction[0]) < 0.9:
                continue
            size = max(s["size"] for s in spans)
            text = " ".join(s["text"].strip() for s in spans)
            if _ARXIV_STAMP.match(text) or _NOT_A_TITLE.match(text):
                continue
            lines.append((size, line["bbox"][1], text))
            if len(text) >= MIN_TITLE_CHARS:
                best_size = max(best_size, size)
    if best_size == 0.0:
        return None
    title_lines = [
        (y, text) for size, y, text in lines if abs(size - best_size) < 0.5 and text.strip()
    ]
    title = _collapse(" ".join(text for _, text in sorted(title_lines)))
    return title if len(title) >= MIN_TITLE_CHARS else None


def _abstract_from(first_page_text: str) -> str | None:
    match = _ABSTRACT_START.search(first_page_text)
    if not match:
        return None
    tail = first_page_text[match.end() : match.end() + MAX_ABSTRACT_CHARS + 400]
    end = _ABSTRACT_END.search(tail)
    abstract = _collapse(tail[: end.start()] if end else tail[:MAX_ABSTRACT_CHARS])
    abstract = abstract[:MAX_ABSTRACT_CHARS]
    return abstract if len(abstract) >= MIN_ABSTRACT_CHARS else None


def extract_metadata(pdf_bytes: bytes) -> PdfMetadata:
    empty = PdfMetadata(title=None, authors=(), abstract=None, doi=None, arxiv_id=None)
    document = None
    try:
        document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        if document.page_count == 0:
            return empty
        info = document.metadata or {}
        first = _LINE_HYPHEN.sub("", document[0].get_text())
        head = first + "\n" + (document[1].get_text() if document.page_count > 1 else "")
        title = (info.get("title") or "").strip()
        if not _usable_title(title):
            title = _title_from_layout(document[0]) or ""
        authors = tuple(
            name.strip() for name in re.split(r"[;,]| and ", info.get("author") or "") if name.strip()
        )
        doi_match = _DOI.search(first)
        arxiv_match = _ARXIV_OWN_STAMP.search(first)
        return PdfMetadata(
            title=title or None,
            authors=authors,
            abstract=_abstract_from(first),
            doi=doi_match.group(1).rstrip(".,;") if doi_match else None,
            arxiv_id=arxiv_match.group(1) if arxiv_match else None,
        )
    except Exception:
        return empty
    finally:
        if document is not None:
            document.close()


PARSE_TIMEOUT_SECONDS = 30.0


def _parse_worker(connection, pdf_bytes: bytes) -> None:
    try:
        connection.send(("ok", extract_metadata(pdf_bytes), extract_text(pdf_bytes)))
    except BaseException as exc:
        connection.send(("error", f"{type(exc).__name__}: {exc}", None))
    finally:
        connection.close()


def parse_isolated(
    pdf_bytes: bytes,
    *,
    timeout: float = PARSE_TIMEOUT_SECONDS,
    target=_parse_worker,
) -> tuple[PdfMetadata, ExtractedText, str | None]:
    import multiprocessing

    empty_meta = PdfMetadata(title=None, authors=(), abstract=None, doi=None, arxiv_id=None)
    failed = ExtractedText(text="", pages=0, alpha_ratio=0.0, quality="failed")
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=target, args=(sender, pdf_bytes), daemon=True)
    process.start()
    sender.close()
    try:
        if not receiver.poll(timeout):
            return empty_meta, failed, f"parsing took longer than {timeout:g} s and was stopped"
        try:
            kind, first, second = receiver.recv()
        except EOFError:
            return empty_meta, failed, "the PDF parser crashed on this file"
        if kind != "ok":
            return empty_meta, failed, f"the PDF could not be parsed ({first})"
        return first, second, None
    finally:
        receiver.close()
        if process.is_alive():
            process.kill()
        process.join(5)
