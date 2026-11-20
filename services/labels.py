
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from itertools import combinations
from typing import Any

from rag_core.tokenize import STOPWORDS, tokenize
from services.llm import UNTRUSTED_TEXT_RULE, LLMClient, LLMUnavailable

TITLES_PER_CLUSTER = 5
TERMS_PER_CLUSTER = 6
MAX_LABEL_WORDS = 4
MAX_ABSTRACT_CHARS = 1500
SMOOTHING = 0.05
MIN_CLUSTER_DOCS_FOR_TERM = 2

_BASE_PROMPT = (
    "You name clusters of research papers so that a reader can tell them apart at a glance.\n"
    "You are given every cluster at once. For each you see its most central paper titles and "
    "its distinctive terms (words frequent in that cluster and rare in the others).\n"
    f"Give each cluster a name of 2 to {MAX_LABEL_WORDS} words that says what is specific to "
    "THAT cluster: its sub-topic, task, technique or application. The name must read as a "
    "natural noun phrase a researcher would recognise, not a list of keywords.\n"
    "Ground every name in the evidence shown: it must be supported by most of that cluster's "
    "titles, build it from words that appear in those titles or in the distinctive terms, and "
    "do not introduce a concept that is absent from them.\n"
    "Names must not be synonyms or near-paraphrases of each other; each must say what is "
    "specific to its own cluster.\n"
    "Reply with exactly one line per cluster, formatted `<id>: <name>`, no quotes, no other text."
)

TOPIC_RULE = (
    '\nAll papers in the whole collection are about "{topic}", so do NOT use those words in a name.'
)

SYSTEM_PROMPT = _BASE_PROMPT + UNTRUSTED_TEXT_RULE

GENERIC_WORDS = frozenset(
    """
    paper papers propose proposed proposes present presents presented approach approaches method
    methods model models novel new based using use used uses show shows shown results result
    performance work study studies existing recent state art however significantly demonstrate
    demonstrates demonstrated experiments experimental experiment extensive across different
    various several improve improves improved improving improvement improvements effective
    effectiveness technique techniques framework frameworks system systems analysis advances
    advancements towards toward overview survey review challenges challenge problem problems
    task tasks datasets dataset benchmark benchmarks achieve achieves achieved
    offer offers introduce introduces underlying allowing allows many consistently address
    addresses enable enables enabling make makes making well high highly large number key main
    also due thus within leading lead leads often can may specific particular particularly
    including include includes provide provides provided demonstrating first second third one
    two three four five time input inputs enhance enhances enhanced evaluated individual
    minimizing minimize moreover furthermore outperform outperforms outperforming competitive
    i.e e.g characterized limitations limitation gaps complex
    remains remain remained nature prior available components component generates generate
    generated generating still combining need needs needed given recently increasingly widely
    significant significantly substantial substantially potential current common commonly
    require requires required requiring due despite while although whereas thereby hence
    investigate investigates investigated explore explores explored develop develops developed
    design designed designs identify identifies identified observe observed observation
    https http www github com org net io arxiv code open source release released publicly
    implementation implementations appendix figure table section way ways average forward
    diverse quality overall strong better best simple simply
    """.split()
)

_SUFFIXES = (
    "ization", "isation", "ations", "ation", "izing", "ising", "ized", "ised", "ative",
    "ings", "ing", "ers", "er", "ed", "es", "s", "ly",
)

_URL = re.compile(
    r"(?:https?://|www\.)\S+|\b[\w-]+(?:\.[\w-]+)*\.(?:com|org|net|io|ai|edu|gov|dev)\b(?:/\S*)?",
    re.IGNORECASE,
)
_SPACES = re.compile(r"\s+")


def system_prompt(topic: str | None) -> str:
    if not topic or not topic.strip():
        return SYSTEM_PROMPT
    return _BASE_PROMPT + TOPIC_RULE.format(topic=topic.strip()) + UNTRUSTED_TEXT_RULE


def central_ids(points: list[dict[str, Any]]) -> dict[int, list[str]]:
    members: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for point in points:
        if point["cluster"] >= 0:
            members[point["cluster"]].append(point)

    result: dict[int, list[str]] = {}
    for cluster, group in members.items():
        cx = sum(p["x"] for p in group) / len(group)
        cy = sum(p["y"] for p in group) / len(group)
        group.sort(key=lambda p: ((p["x"] - cx) ** 2 + (p["y"] - cy) ** 2, p["paper_id"]))
        result[cluster] = [p["paper_id"] for p in group]
    return result


def central_titles(points: list[dict[str, Any]], titles: dict[str, str]) -> dict[int, list[str]]:
    return {
        cluster: [titles[pid] for pid in ids if pid in titles][:TITLES_PER_CLUSTER]
        for cluster, ids in central_ids(points).items()
    }


def _stem(word: str) -> str:
    if word.endswith("ies") and len(word) > 5:
        return word[:-3] + "y"
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            word = word[: -len(suffix)]
            break
    return word[:-1] if word.endswith("e") and len(word) > 4 else word


def _root(word: str) -> str:
    return word[:5] if len(word) >= 6 else word


def topic_roots(topic: str | None) -> frozenset[str]:
    words = tokenize(topic or "")
    roots = {_root(word) for word in words}
    if len(words) > 1:
        roots.add("".join(word[0] for word in words))
    return frozenset(roots)


def _document_terms(text: str, banned: frozenset[str]) -> set[str]:
    raw = tokenize(_URL.sub(" ", text).replace("-", " "), drop_stopwords=False)

    def keep(word: str) -> bool:
        return (
            len(word) >= 3
            and word not in STOPWORDS
            and word not in GENERIC_WORDS
            and _root(word) not in banned
            and not word.replace(".", "").isdigit()
        )

    terms = {word for word in raw if keep(word)}
    terms.update(f"{a} {b}" for a, b in zip(raw, raw[1:]) if keep(a) and keep(b))
    return terms


def distinctive_terms(
    points: list[dict[str, Any]],
    titles: dict[str, str],
    abstracts: dict[str, str | None] | None = None,
    *,
    topic: str | None = None,
    top: int = TERMS_PER_CLUSTER,
) -> dict[int, list[str]]:
    members = central_ids(points)
    if not members:
        return {}
    banned = topic_roots(topic)
    abstracts = abstracts or {}

    documents: dict[int, list[set[str]]] = {}
    totals: Counter[str] = Counter()
    for cluster, ids in members.items():
        docs = []
        for paper_id in ids:
            text = f"{titles.get(paper_id, '')}. {(abstracts.get(paper_id) or '')[:MAX_ABSTRACT_CHARS]}"
            terms = _document_terms(text, banned)
            docs.append(terms)
            totals.update(terms)
        documents[cluster] = docs

    clustered = sum(len(docs) for docs in documents.values())
    candidates: list[tuple[float, int, str]] = []
    for cluster, docs in documents.items():
        local: Counter[str] = Counter()
        for terms in docs:
            local.update(terms)
        outside = clustered - len(docs)
        for term, count in local.items():
            if count < min(MIN_CLUSTER_DOCS_FOR_TERM, len(docs)):
                continue
            inside_share = count / len(docs)
            outside_share = (totals[term] - count) / outside if outside else 0.0
            lift = inside_share * math.log((inside_share + SMOOTHING) / (outside_share + SMOOTHING))
            if lift > 0:
                candidates.append((lift, cluster, term))

    candidates.sort(key=lambda item: (-item[0], item[1], -item[2].count(" "), item[2]))
    result: dict[int, list[str]] = {cluster: [] for cluster in members}
    claimed: set[str] = set()
    for _, cluster, term in candidates:
        if len(result[cluster]) >= top:
            continue
        stems = {_stem(word) for word in term.split()}
        if stems & claimed:
            continue
        result[cluster].append(term)
        claimed |= stems
    return result


def _content_stems(name: str, banned: frozenset[str]) -> set[str]:
    return {
        _stem(word)
        for word in tokenize(name)
        if word not in GENERIC_WORDS and _root(word) not in banned
    }


def _normalised(name: str) -> str:
    return _SPACES.sub(" ", " ".join(re.findall(r"\w+", name.casefold()))).strip()


def uninformative_names(names: dict[int, str], *, topic: str | None = None) -> list[int]:
    banned = topic_roots(topic)
    return sorted(c for c, name in names.items() if not _content_stems(name, banned))


def ungrounded_names(
    names: dict[int, str],
    evidence: dict[int, list[str]],
    *,
    topic: str | None = None,
) -> dict[int, list[str]]:
    banned = topic_roots(topic)
    result: dict[int, list[str]] = {}
    for cluster, name in names.items():
        seen = {_root(_stem(word)) for text in evidence.get(cluster, []) for word in tokenize(text.replace("-", " "))}
        missing = [
            word
            for word in tokenize(name)
            if word not in GENERIC_WORDS and _root(word) not in banned and _root(_stem(word)) not in seen
        ]
        if missing:
            result[cluster] = missing
    return result


def name_collisions(
    names: dict[int, str], *, topic: str | None = None
) -> list[tuple[int, int]]:
    banned = topic_roots(topic)
    stems = {cluster: _content_stems(name, banned) for cluster, name in names.items()}
    collisions = []
    for (a, sa), (b, sb) in combinations(sorted(stems.items()), 2):
        if _normalised(names[a]) == _normalised(names[b]):
            collisions.append((a, b))
            continue
        shared = sa & sb
        if not shared:
            continue
        union = sa | sb
        if len(shared) >= 2 or sa <= sb or sb <= sa or len(shared) / len(union) >= 1 / 3:
            collisions.append((a, b))
    return collisions


_LINE = re.compile(r"^\W*(?:cluster\s*)?(\d+)\W*?\s*[:|\-–—]\s*(.+)$", re.IGNORECASE)


def _clean_label(raw: str) -> str:
    first_line = raw.strip().splitlines()[0] if raw.strip() else ""
    words = re.sub(r"[\"'`*.:;]+", "", first_line).split()
    return " ".join(words[:MAX_LABEL_WORDS])


def parse_names(text: str, wanted: set[int]) -> dict[int, str]:
    found: dict[int, str] = {}
    body = re.sub(r"^```\w*\s*|\s*```$", "", text.strip())
    try:
        decoded = json.loads(body)
    except ValueError:
        decoded = None
    if isinstance(decoded, dict):
        for key, value in decoded.items():
            if str(key).isdigit() and isinstance(value, str):
                found[int(key)] = _clean_label(value)
    else:
        for line in body.splitlines():
            match = _LINE.match(line.strip())
            if match:
                found[int(match.group(1))] = _clean_label(match.group(2))
    return {cluster: name for cluster, name in found.items() if cluster in wanted and name}


def _describe(cluster: int, size: int, titles: list[str], terms: list[str]) -> str:
    lines = [f"Cluster {cluster} ({size} papers):", "  Central titles:"]
    lines += [f"  - {title}" for title in titles]
    lines.append("  Distinctive terms: " + (", ".join(terms) if terms else "(none found)"))
    return "\n".join(lines)


def _problems(
    labels: dict[int, str],
    wanted: set[int],
    topic: str | None,
    evidence: dict[int, list[str]],
) -> tuple[list[int], list[tuple[int, int]], list[int], dict[int, list[str]]]:
    return (
        sorted(wanted - set(labels)),
        name_collisions(labels, topic=topic),
        uninformative_names(labels, topic=topic),
        ungrounded_names(labels, evidence, topic=topic),
    )


def cluster_labels(
    points: list[dict[str, Any]],
    titles: dict[str, str],
    *,
    llm: LLMClient,
    topic: str | None = None,
    abstracts: dict[str, str | None] | None = None,
) -> dict[str, Any]:
    shown = central_titles(points, titles)
    terms = distinctive_terms(points, titles, abstracts, topic=topic)
    result: dict[str, Any] = {
        "labels": {},
        "unavailable": None,
        "terms": terms,
        "distinct": True,
        "collisions": [],
        "unnamed": [],
        "uninformative": [],
        "ungrounded": {},
    }
    if not shown:
        return result

    sizes = Counter(point["cluster"] for point in points if point["cluster"] >= 0)
    wanted = set(shown)
    prompt = "Clusters:\n\n" + "\n\n".join(
        _describe(cluster, sizes[cluster], shown[cluster], terms.get(cluster, []))
        for cluster in sorted(shown)
    )
    system = system_prompt(topic)

    try:
        first = llm.complete(system, prompt, max_output_tokens=600)
    except LLMUnavailable as exc:
        result["unavailable"] = {"code": exc.code, "message": exc.message}
        result["unnamed"] = sorted(wanted)
        result["distinct"] = False
        return result
    labels = parse_names(first.text, wanted)

    evidence = {cluster: shown[cluster] + terms.get(cluster, []) for cluster in shown}
    unnamed, collisions, empty, invented = _problems(labels, wanted, topic, evidence)
    if unnamed or collisions or empty or invented:
        complaints = []
        if unnamed:
            complaints.append("no name was given for cluster " + ", ".join(map(str, unnamed)))
        if collisions:
            complaints.append(
                "these names are too similar to tell the clusters apart: "
                + "; ".join(f'"{labels[a]}" (cluster {a}) and "{labels[b]}" (cluster {b})' for a, b in collisions)
            )
        if empty:
            complaints.append(
                "these names say nothing specific (filler or the collection topic only): "
                + "; ".join(f'"{labels[c]}" (cluster {c})' for c in empty)
            )
        if invented:
            complaints.append(
                "these name words appear in none of the titles or terms shown for the cluster, "
                "so they are not supported: "
                + "; ".join(f'"{labels[c]}" (cluster {c}): {", ".join(words)}' for c, words in invented.items())
            )
        retry_prompt = (
            f"{prompt}\n\nYour previous answer:\n{first.text.strip()}\n\n"
            f"Problems: {'. '.join(complaints)}. "
            "Give every cluster a name that says what is specific to it, supported by its own "
            "titles and distinctive terms, and not a synonym or near-paraphrase of another "
            "cluster's name. Reply in the same `<id>: <name>` format, all clusters."
        )
        try:
            second = llm.complete(system, retry_prompt, max_output_tokens=600)
        except LLMUnavailable:
            second = None
        if second is not None:
            labels = {**labels, **parse_names(second.text, wanted)}
            unnamed, collisions, empty, invented = _problems(labels, wanted, topic, evidence)

    result["labels"] = dict(sorted(labels.items()))
    result["unnamed"] = unnamed
    result["collisions"] = [list(pair) for pair in collisions]
    result["uninformative"] = empty
    result["ungrounded"] = invented
    result["distinct"] = not (unnamed or collisions or empty)
    return result
