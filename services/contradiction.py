
from __future__ import annotations

import itertools
import json
import re
from typing import Any

from services.llm import UNTRUSTED_TEXT_RULE, LLMClient, LLMUnavailable
from services.map import select_vectors
from services.retrieval import paper_details

VERDICTS = ("agree", "conflict", "superseded")
NO_VERDICT = "no_comparable_claims"
MAX_PAPERS = 8
MIN_PAPERS = 2
DEFAULT_PAIRS = 5
MAX_PAIRS = 10
MIN_SENTENCE_CHARS = 25
BODY_SENTENCES = 8
MIN_BODY_CHARS = 60
MAX_BODY_CHARS = 400
JUDGE_TOKENS = 800
CONFIRM_TOKENS = 500

SYSTEM_PROMPT = (
    "You compare two research papers using ONLY the numbered sentences supplied. The sentences come "
    "from each paper's abstract and from its own full text (results, analysis, conclusions).\n"
    "Look for ONE sentence from paper A and ONE from paper B that make claims about the SAME "
    "specific question (the same method, task or measured quantity: for example whether an approach "
    "helps, whether a metric is reliable, which of two options performs better). Each chosen "
    "sentence must be a finding of that paper's own work. Classify it as:\n"
    '- "result": states an outcome, measurement or conclusion of the paper\'s own experiments or analysis.\n'
    '- "background": general context, motivation, a goal, a method description or a statement about '
    "the field or about earlier work.\n"
    '- "opinion": views, impressions or reports from other people (participants, interviewees, other authors).\n'
    "Background and opinion sentences are never evidence for a verdict. Then decide:\n"
    '- "conflict": the two sentences cannot both be true of the same method under the same conditions. '
    "If both could be true (a different variant, setup, input size, dataset or metric), it is not a conflict.\n"
    '- "agree": both sentences, each on its own, report the same finding about the same method or question.\n'
    '- "superseded": the later paper reports a result that replaces or corrects the earlier one.\n'
    '- "none": no pair of result sentences clearly addresses the same question.\n'
    "These are NOT conflicts or agreements, answer none for them: working on different tasks; one paper "
    "proposing a method while the other criticises a general approach; using the same dataset, "
    "benchmark or topic; both being about retrieval-augmented generation; differing numbers on "
    "different benchmarks; two different methods that each report working well; a motivating "
    "observation next to a method's result; a variant or a differently named method is not the "
    "same method.\n"
    "Judge only what the two chosen sentences state; your reason must restate what each one says "
    "and add nothing to it.\n"
    "Do not infer anything the sentences do not state. Reply with ONLY a JSON object:\n"
    '{"verdict": "agree|conflict|superseded|none", "a": <sentence number in A or null>, '
    '"b": <sentence number in B or null>, "kind_a": "result|background|opinion", '
    '"kind_b": "result|background|opinion", "reason": "<one short sentence>"}'
) + UNTRUSTED_TEXT_RULE

CONFIRM_PROMPT = (
    "Two sentences from two different research papers are given. Judge ONLY these two sentences. "
    'First classify each as "result" (states an outcome, measurement or conclusion of that paper\'s '
    'own experiments or analysis) or "background" (context, motivation, goals, method description, '
    'views of other people, or statements about the field or earlier work). Then decide how they relate:\n'
    '- "conflict": they cannot both be true of the same method under the same conditions.\n'
    '- "agree": both report the same finding about the same method or question. Two different '
    "methods that each work well, or a motivating observation beside a method's result, are not agreement.\n"
    '- "compatible": both could be true (a different variant, setup, dataset, metric or input size), '
    "or neither supports the other.\n"
    '- "unrelated": they are about different subjects.\n'
    'Reply with ONLY a JSON object: {"subject_a": "<what A is about, at most 8 words>", '
    '"subject_b": "<what B is about, at most 8 words>", "kind_a": "result|background", '
    '"kind_b": "result|background", "relation": "conflict|agree|compatible|unrelated"}'
) + UNTRUSTED_TEXT_RULE

_ABBREVIATIONS = ("Fig", "Figs", "Ref", "Refs", "Tab", "Eq", "Eqs", "Sec", "Secs", "vs", "al", "approx", "cf",
                  "e.g", "i.e", "No", "Dr", "vol", "Vol", "pp")
_SENTENCE_SPLIT = re.compile(
    r"(?<=[.!?])" + "".join(rf"(?<!\b{re.escape(a)}\.)" for a in _ABBREVIATIONS) + r"(?<!\s\d\.)\s+(?=[A-Z0-9(\[])"
)
_ABBREVIATION_STOP = re.compile(r"\b(?:" + "|".join(re.escape(a) for a in _ABBREVIATIONS) + r")\.")
_FINDING_CUE = re.compile(
    r"outperform|improv|better|worse|higher|lower|achiev|surpass|fail|limited|reliab|"
    r"did not|does not|do not|no (?:significant|clear|improvement)|significant|confirm|demonstrat|"
    r"reveal|indicate|suggest|found|find|shows? that|showed|shown|result|conclu|observ|reduc|"
    r"aligned|agree|consistent|superior|inferior|degrad|decrease|increase|trade-?off",
    re.I,
)
_NOT_OWN_CLAIM = re.compile(
    r"\bet al\b|\[\d[\d,;\s\u2013-]*\]|(?-i:\[[A-Z](?:\s*[,;]\s*[A-Z])*\])|"
    r"\((?:[A-Z][A-Za-z-]+,? )+\d{4}[a-z]?\)|arxiv|proceedings|\bpp\.|\bdoi\b|https?://|@|"
    r"\bTables? (?:\d+|[IVX]+)\b|\bFig(?:ure|\.)?s? ?(?:\d+|[IVX]+)\b",
    re.I,
)
_DEBRIS = re.compile(
    r"thank|grateful|supported by|\bgrants?\b|acknowledg|funded|funding|fellowship|scholarship|sponsored|"
    r"springer nature|latex template|ieee transactions|published as a conference|under review|"
    r"preprint submitted|\u00a9|copyright|\bvol\. ?\d",
    re.I,
)
_CAPS_RUN = re.compile(r"(?:\b[A-Z]{3,}\b ){4,}")
_OWN_WORK = re.compile(r"\b(?:we|our|ours|this (?:paper|work|study|article|thesis)|the present (?:paper|work|study))\b", re.I)
_PRIOR_WORK = re.compile(
    r"\b(?:existing|previous(?:ly)?|prior|earlier|literature|related work|recent(?:ly)? (?:work|studies)|"
    r"(?:have|has) been (?:proposed|studied|shown|used))\b",
    re.I,
)
_LOWERCASE_AFTER_STOP = re.compile(r"[.!?] [a-z]")


class ContradictionError(ValueError):
    pass


def split_sentences(text: str) -> list[str]:
    pieces = [piece.strip() for piece in _SENTENCE_SPLIT.split(" ".join((text or "").split()))]
    return [piece for piece in pieces if len(piece) >= MIN_SENTENCE_CHARS]


def body_sentences(chunk_texts: list[str], limit: int = BODY_SENTENCES) -> list[str]:
    seen: set[str] = set()
    scored: list[tuple[int, int, str]] = []
    position = 0
    for text in chunk_texts:
        for piece in _SENTENCE_SPLIT.split(" ".join((text or "").split())):
            piece = piece.strip()
            if not (MIN_BODY_CHARS <= len(piece) <= MAX_BODY_CHARS):
                continue
            if not (piece[0].isupper() and piece[-1] in ".!?") or piece in seen:
                continue
            letters = sum(ch.isalpha() for ch in piece)
            digits = sum(ch.isdigit() for ch in piece)
            if letters < 0.75 * len(piece.replace(" ", "")) or digits > 0.1 * len(piece):
                continue
            if _NOT_OWN_CLAIM.search(piece) or _DEBRIS.search(piece) or _CAPS_RUN.search(piece):
                continue
            if _LOWERCASE_AFTER_STOP.search(_ABBREVIATION_STOP.sub(" ", piece)):
                continue
            own = bool(_OWN_WORK.search(piece))
            if _PRIOR_WORK.search(piece) and not own:
                continue
            seen.add(piece)
            position += 1
            cues = len(_FINDING_CUE.findall(piece))
            if cues:
                scored.append((cues + (2 if own else 0), position, piece))
    best = sorted(scored, key=lambda item: (-item[0], -item[1]))[:limit]
    return [sentence for _, _, sentence in sorted(best, key=lambda item: item[1])]


def parse_verdict(reply: str) -> dict[str, Any] | None:
    match = re.search(r"\{.*\}", reply, flags=re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def _index(value: Any, length: int, label: str) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        return None
    match = re.fullmatch(rf"(?:{label}\s*)?([0-9]+)\.?", str(value).strip(), flags=re.I)
    if not match:
        return None
    number = int(match.group(1))
    return number - 1 if 1 <= number <= length else None


def _word(value: Any) -> str:
    return value.strip().lower() if isinstance(value, str) else ""


def _confirm(llm: LLMClient, quote_a: str, quote_b: str) -> tuple[str, bool]:
    response = llm.complete(CONFIRM_PROMPT, f"A: {quote_a}\nB: {quote_b}", max_output_tokens=CONFIRM_TOKENS)
    data = parse_verdict(response.text) or {}
    return _word(data.get("relation")), _word(data.get("kind_a")) == "result" == _word(data.get("kind_b"))


def judge(
    llm: LLMClient, paper_a: dict[str, Any], paper_b: dict[str, Any]
) -> dict[str, Any]:
    abstract_a, abstract_b = split_sentences(paper_a["abstract"]), split_sentences(paper_b["abstract"])
    sentences_a = abstract_a + [x for x in paper_a.get("body", []) if x not in abstract_a]
    sentences_b = abstract_b + [x for x in paper_b.get("body", []) if x not in abstract_b]
    base = {"paper_a": _ref(paper_a), "paper_b": _ref(paper_b), "quote_a": None, "quote_b": None, "reason": None,
            "source_a": None, "source_b": None}
    if not sentences_a or not sentences_b:
        return {**base, "verdict": NO_VERDICT, "note": "a paper has no usable sentences"}

    def block(label: str, paper: dict[str, Any], sentences: list[str]) -> str:
        lines = "\n".join(f"{label}{i}. {s}" for i, s in enumerate(sentences, start=1))
        return f"Paper {label} ({paper['year'] or 'year unknown'}): {paper['title']}\n{lines}"

    user = block("A", paper_a, sentences_a) + "\n\n" + block("B", paper_b, sentences_b)
    response = llm.complete(SYSTEM_PROMPT, user, max_output_tokens=JUDGE_TOKENS)
    data = parse_verdict(response.text)
    result = {**base, "model": response.model, "cached": response.cached}
    if data is None:
        return {**result, "verdict": NO_VERDICT, "note": "the model's reply was not valid JSON"}

    verdict = _word(data.get("verdict"))
    if verdict == "none":
        return {**result, "verdict": NO_VERDICT, "note": "the model found no pair of sentences addressing the same question"}
    index_a, index_b = _index(data.get("a"), len(sentences_a), "A"), _index(data.get("b"), len(sentences_b), "B")
    if not verdict:
        return {**result, "verdict": NO_VERDICT, "note": "the model gave no verdict"}
    if verdict not in VERDICTS or index_a is None or index_b is None:
        return {**result, "verdict": NO_VERDICT,
                "note": "the model named no valid sentence in both papers, so no verdict is shown"}
    if _word(data.get("kind_a")) != "result" or _word(data.get("kind_b")) != "result":
        return {**result, "verdict": NO_VERDICT,
                "note": "a chosen sentence is not a finding of the paper's own work (background or opinion), "
                "so no verdict is shown"}
    if verdict == "superseded":
        years = (paper_a["year"], paper_b["year"])
        if None in years or years[0] == years[1]:
            return {**result, "verdict": NO_VERDICT,
                    "note": "'superseded' needs two different known years; none is claimed"}

    relation, both_results = _confirm(llm, sentences_a[index_a], sentences_b[index_b])
    if relation != ("conflict" if verdict == "superseded" else verdict):
        return {**result, "verdict": NO_VERDICT,
                "note": f"a second reading of the two sentences alone judged them '{relation or 'unreadable'}', "
                "so no verdict is shown"}
    if not both_results:
        return {**result, "verdict": NO_VERDICT,
                "note": "a second reading did not find a finding in both sentences, so no verdict is shown"}

    reason = str(data.get("reason") or "").strip()
    return {
        **result,
        "verdict": verdict,
        "quote_a": sentences_a[index_a],
        "quote_b": sentences_b[index_b],
        "source_a": "abstract" if index_a < len(abstract_a) else "full text",
        "source_b": "abstract" if index_b < len(abstract_b) else "full text",
        "reason": reason[:300] or None,
    }


def _ref(paper: dict[str, Any]) -> dict[str, Any]:
    return {"paper_id": paper["paper_id"], "title": paper["title"], "year": paper["year"], "url": paper["source_url"]}


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def choose_pairs(
    conn: Any, collection_id: str, paper_ids: list[str], max_pairs: int
) -> list[tuple[str, str]]:
    ids, vectors, _, _ = select_vectors(conn, collection_id)
    vector_of = dict(zip(ids, vectors))
    scored = []
    for a, b in itertools.combinations(sorted(paper_ids), 2):
        similarity = _cosine(vector_of[a], vector_of[b]) if a in vector_of and b in vector_of else -1.0
        scored.append((-similarity, a, b))
    scored.sort()
    return [(a, b) for _, a, b in scored[:max_pairs]]


def load_papers(conn: Any, collection_id: str, paper_ids: list[str]) -> dict[str, dict[str, Any]]:
    placeholders = ",".join("?" for _ in paper_ids)
    rows = conn.execute(
        f"SELECT id, title, year FROM papers WHERE collection_id=? AND id IN ({placeholders})",
        [collection_id, *paper_ids],
    ).fetchall()
    meta = {row["id"]: row for row in rows}
    missing = [pid for pid in paper_ids if pid not in meta]
    if missing:
        raise ContradictionError(f"papers not in this collection: {', '.join(missing[:3])}")
    details = paper_details(conn, paper_ids)
    papers = {}
    for pid in paper_ids:
        chunks = [r["text"] for r in conn.execute("SELECT text FROM chunks WHERE paper_id=? AND seq>0 ORDER BY seq", (pid,))]
        papers[pid] = {"paper_id": pid, "title": meta[pid]["title"], "year": meta[pid]["year"],
                       "abstract": details[pid]["abstract"] or "", "source_url": details[pid]["source_url"],
                       "body": body_sentences(chunks)}
    return papers


def compare_papers(
    conn: Any,
    collection_id: str,
    paper_ids: list[str],
    *,
    llm: LLMClient,
    max_pairs: int = DEFAULT_PAIRS,
) -> dict[str, Any]:
    unique = list(dict.fromkeys(paper_ids))
    if not MIN_PAPERS <= len(unique) <= MAX_PAPERS:
        raise ContradictionError(f"choose between {MIN_PAPERS} and {MAX_PAPERS} papers; got {len(unique)}")
    max_pairs = max(1, min(max_pairs, MAX_PAIRS))

    papers = load_papers(conn, collection_id, unique)

    comparisons: list[dict[str, Any]] = []
    unavailable: dict[str, str] | None = None
    incomplete: dict[str, Any] | None = None
    for a, b in choose_pairs(conn, collection_id, unique, max_pairs):
        try:
            comparisons.append(judge(llm, papers[a], papers[b]))
        except LLMUnavailable as exc:
            unavailable = {"code": exc.code, "message": exc.message}
            incomplete = {"paper_a": _ref(papers[a]), "paper_b": _ref(papers[b])}
            break

    note = (
        "Only the abstract and up to %d result-bearing full-text sentences per paper are read, so a "
        "claim elsewhere in a paper is invisible here. A verdict is a lead to check, not a finding." % BODY_SENTENCES
    )
    if incomplete:
        note += (f' The run stopped while comparing "{incomplete["paper_a"]["title"]}" with '
                 f'"{incomplete["paper_b"]["title"]}"; that pair has no result.')
    return {
        "comparisons": comparisons,
        "n_pairs_requested": max_pairs,
        "n_pairs_possible": len(unique) * (len(unique) - 1) // 2,
        "unavailable": unavailable,
        "incomplete_pair": incomplete,
        "method": "pairs chosen by map similarity; each verdict rests on one sentence quoted verbatim "
        "from each paper's abstract or a result-bearing sentence of its full text (the model picks "
        "sentence numbers, it does not write the quotes); a second reading of just the two quotes must "
        "confirm the verdict before it is shown",
        "note": note,
        "ai_generated": True,
    }
