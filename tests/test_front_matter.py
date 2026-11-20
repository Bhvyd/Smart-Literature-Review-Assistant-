
import pymupdf

from ingest import pdf
from ingest.chunk import chunk_text
from ingest.pdf import extract_text

ABSTRACT = (
    "Retrieval augmented generation grounds a language model in retrieved passages. "
    "We study how chunk boundaries change answer quality on academic texts."
)


def _page(*lines: str) -> str:
    return "\n".join(lines) + "\n"


def _clean(page: str) -> str:
    pages, _ = pdf.drop_front_matter([page, "Second page body text."])
    return pages[0]


def test_text_before_an_early_abstract_heading_is_dropped():
    page = _page(
        "Evaluating Chunking Strategies for RAG",
        "Valentin Kreileder",
        "Computer Science",
        "Deggendorf Institute of Technology",
        "kreileder@gmx.net",
        "Abstract",
        ABSTRACT,
        "1 Introduction",
    )
    out = _clean(page)
    assert "@" not in out and "Deggendorf" not in out and "Kreileder" not in out
    assert out.startswith("Abstract") and ABSTRACT in out and "1 Introduction" in out


def test_abstract_heading_variants_are_recognised():
    for heading in ("ABSTRACT", "Abstract.", "Abstract—" + ABSTRACT[:20], "Abstract:"):
        out = _clean(_page("Title", "A. Author", "a@b.edu", heading, ABSTRACT))
        assert "a@b.edu" not in out and ABSTRACT in out, heading


def test_a_body_line_that_merely_starts_with_the_word_abstract_is_not_a_heading():
    page = _page(
        "A Survey of Program Analysis",
        "Abstract interpretation is a technique for sound static analysis of programs.",
        ABSTRACT,
    )
    assert _clean(page) == page


def test_papers_without_an_abstract_heading_lose_the_author_block_only():
    page = _page(
        "Enhancing Critical Thinking with AI",
        "XUYANG ZHU, Stanford University, USA",
        "SEJOON CHANG, Stanford University, USA",
        "Fig. 1. Two layers of hallucination in question answering relevant to the history subject here.",
        ABSTRACT,
    )
    out = _clean(page)
    assert "Stanford" not in out
    assert "Fig. 1." in out and ABSTRACT in out


def test_a_title_page_with_no_author_block_is_left_alone():
    page = _page("A Plain Title", ABSTRACT, "More body text that talks about the University of Chunking.")
    assert _clean(page) == page


def test_one_affiliation_word_in_a_title_is_not_enough_to_cut():
    page = _page("Institute of Things: a Study", ABSTRACT)
    assert _clean(page) == page


def test_footnote_lines_with_emails_and_equal_contribution_are_dropped_from_page_one():
    page = _page(
        "Abstract",
        ABSTRACT,
        "*Equal contribution. Correspondence to: Di He <dihe@pku.edu.cn>.",
        "ORCID: 0000-0002-0059-8458",
        "1 Introduction",
    )
    out = _clean(page)
    assert "@" not in out and "ORCID" not in out and "Equal contribution" not in out
    assert ABSTRACT in out and "1 Introduction" in out


def test_other_pages_are_untouched():
    pages, _ = pdf.drop_front_matter(["Abstract\n" + ABSTRACT, "Contact: someone@example.org about this."])
    assert pages[1] == "Contact: someone@example.org about this."


def _real_pdf(lines: list[str]) -> bytes:
    document = pymupdf.open()
    page = document.new_page()
    y = 60.0
    for line in lines:
        page.insert_text((40, y), line, fontsize=9)
        y += 12
    return document.tobytes()


def test_first_chunk_of_an_extracted_pdf_has_no_email_or_affiliation():
    body = "The evaluation uses academic texts and measures answer quality. " * 30
    data = _real_pdf(
        [
            "Evaluating Chunking Strategies for RAG on Academic Texts",
            "Valentin Kreileder",
            "Deggendorf Institute of Technology",
            "kreileder@gmx.net",
            "Abstract",
            *[ABSTRACT[i : i + 80] for i in range(0, len(ABSTRACT), 80)],
        ]
    )
    document = pymupdf.open(stream=data, filetype="pdf")
    second = document.new_page()
    for n, start in enumerate(range(0, len(body), 90)):
        second.insert_text((40, 40 + 11 * n), body[start : start + 90], fontsize=8)
    result = extract_text(document.tobytes())
    first = chunk_text(result.text, target=60, overlap=5)[0].text
    assert "@" not in first and "Deggendorf" not in first
    assert "Retrieval augmented generation" in first
    assert result.front_matter_chars > 0


def test_heading_rule_alone_drops_what_precedes_the_heading():
    for heading in ("Abstract", "ABSTRACT", "abstract", "1. Abstract", "I. ABSTRACT", "Abstract-Retrieval"):
        out = _clean(_page("A Title", "Some Name", "More Words", heading, ABSTRACT))
        assert "Some Name" not in out and "A Title" not in out, heading
        assert ABSTRACT in out, heading


def test_a_heading_beyond_the_window_is_ignored():
    padding = "".join(f"Line {i} of ordinary page text.\n" for i in range(120))
    assert len(padding) > pdf.FRONT_MATTER_WINDOW
    page = padding + _page("Abstract", ABSTRACT)
    assert _clean(page) == page


def test_abstract_level_hyphen_compound_is_not_a_heading():
    page = _page("A Title", "Abstract-level features are cheap to compute and widely reused.", ABSTRACT)
    assert _clean(page) == page


def test_an_email_alone_triggers_the_fallback_cut():
    out = _clean(_page("A Title", "Some Name", "some.name@example.org", ABSTRACT))
    assert "some.name" not in out and "Some Name" not in out and ABSTRACT in out


def test_prose_after_the_author_block_stops_the_cut():
    sentence = "We show that retrieval from small models works well in practice for students."
    page = _page("A Title", "Some Name", "Example Institute of Technology", "x@example.org", sentence,
                 "Example University Department of Things appears in this sentence about a thing.")
    out = _clean(page)
    assert out.startswith(sentence) and "Example University Department" in out


def test_narrow_column_abstract_with_a_university_line_keeps_the_abstract():
    page = _page(
        "Efficient Retrieval for Small Models", "Jane Roe", "Example Lab", "jane@example.org",
        "We study retrieval for", "students at a university", "who read long papers and",
        "need short answers fast.",
    )
    out = _clean(page)
    assert out.startswith("We study retrieval for") and "need short answers fast." in out
    assert "jane@example.org" not in out


def test_a_sentence_with_an_affiliation_word_is_not_an_affiliation_line():
    page = _page("A Title", "Some Name", "x@example.org", "We study retrieval at a university lab",
                 "who read long papers and", "need short answers fast.")
    assert _clean(page).startswith("We study retrieval at a university lab")


def test_many_short_lines_then_a_late_email_removes_nothing_from_the_top():
    lines = [f"short line {i}" for i in range(80)] + ["late@example.org"]
    page = _page(*lines)
    out = _clean(page)
    assert "short line 0" in out and "short line 79" in out


def test_three_plain_lines_after_the_author_block_end_the_run():
    page = _page("A Title", "Some Name", "x@example.org", "Plain Line One", "Plain Line Two", "Plain Line Three",
                 "Example University Department Hall")
    out = _clean(page)
    assert out.startswith("Plain Line One") and "Example University Department Hall" in out


def test_metric_names_and_sentence_runs_are_not_email_addresses():
    body = [
        "NDCG@10 and P@1 improved over the baseline.",
        "recall@10.Moreover the model improves.",
        "Correspondence to the ground truth was measured on a held-out split, and we found that",
        "Work done at scale requires care; we show it.",
        "Equal contribution of each head is not required by the model here.",
    ]
    page = _page("Abstract", ABSTRACT, *body)
    out = _clean(page)
    for line in body[:4]:
        assert line in out, line


def test_marked_footnotes_are_dropped_but_unmarked_phrases_in_prose_are_not():
    page = _page("Abstract", ABSTRACT, "∗Work done during an internship at Example Corp.",
                 "1Correspondence to: the authors.", "†Equal contribution.", "Equal contribution of heads is cheap.")
    out = _clean(page)
    assert "internship" not in out and "the authors" not in out and "†Equal" not in out
    assert "Equal contribution of heads is cheap." in out


def test_an_email_inside_a_long_merged_line_is_redacted_not_kept_or_dropped():
    long_line = "Retrieval quality depends on chunking. " * 12 + "Contact jane.roe@example.org for data. " + "More text here. " * 5
    assert len(long_line) > pdf.MAX_FOOTNOTE_LINE
    out = _clean(_page("Abstract", ABSTRACT, long_line))
    assert "@" not in out and "Retrieval quality depends on chunking." in out and "More text here." in out


def test_no_pages_is_a_no_op():
    assert pdf.drop_front_matter([]) == ([], 0)


def test_braced_email_lists_are_recognised():
    out = _clean(_page("A Title", "Some Name", "{alpha, beta}@cs.example.edu", ABSTRACT))
    assert "@" not in out and ABSTRACT in out


def test_stacked_markers_parenthesised_and_bare_correspondence_footnotes_are_dropped():
    page = _page("Abstract", ABSTRACT, "1∗These authors contributed equally to this work.",
                 "(Corresponding author: Won-Yong Shin.)", "Correspondence to:", "∗All emails @google.com.",
                 "g.j.t.leus}@tudelft.nl.", "1 Introduction")
    out = _clean(page)
    assert out == "Abstract\n" + ABSTRACT + "\n1 Introduction\n"
