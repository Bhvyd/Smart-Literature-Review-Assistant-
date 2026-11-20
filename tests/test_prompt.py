
import pytest

from rag_core.prompt import Passage, build_prompt


def _p(n: int, text: str = "body") -> Passage:
    return Passage(chunk_id=f"c{n}", paper_id=f"p{n}", text=text, title=f"T{n}")


def test_slots_are_numbered_in_order():
    bundle = build_prompt("why?", [_p(1), _p(2)])
    assert "[1]" in bundle.text and "[2]" in bundle.text
    assert bundle.citations == {1: "c1", 2: "c2"}
    assert bundle.papers == {1: "p1", 2: "p2"}
    assert bundle.used == 2


def test_query_appears_once():
    bundle = build_prompt("what is lora?", [_p(1)])
    assert bundle.text.count("what is lora?") == 1


def test_answer_rule_is_present():
    bundle = build_prompt("q", [_p(1)])
    assert "citing sources as [n]" in bundle.text
    assert "does not answer" in bundle.text


def test_truncation_drops_later_passages():
    single = build_prompt("q", [_p(1, "x" * 50)])
    budget = len(single.text) + 10
    bundle = build_prompt("q", [_p(1, "x" * 50), _p(2, "y" * 50)], max_chars=budget)
    assert bundle.used == 1
    assert 2 not in bundle.citations
    assert "y" * 10 not in bundle.text


def test_single_oversized_passage_is_still_included():
    bundle = build_prompt("q", [_p(1, "z" * 500)], max_chars=80)
    assert bundle.used == 1
    assert 1 in bundle.citations


def test_empty_passages_raise():
    with pytest.raises(ValueError):
        build_prompt("q", [])


def test_passages_without_titles_still_number():
    bundle = build_prompt("q", [Passage(chunk_id="c", paper_id="p", text="t")])
    assert "[1]" in bundle.text
    assert bundle.citations == {1: "c"}


def test_document_order_is_preserved():
    bundle = build_prompt("q", [_p(1, "first"), _p(2, "second"), _p(3, "third")])
    assert bundle.text.index("first") < bundle.text.index("second") < bundle.text.index("third")
    assert bundle.citations == {1: "c1", 2: "c2", 3: "c3"}


def test_output_respects_budget_when_it_can():
    bundle = build_prompt("q", [_p(i, "body") for i in range(1, 4)], max_chars=250)
    assert len(bundle.text) <= 250
