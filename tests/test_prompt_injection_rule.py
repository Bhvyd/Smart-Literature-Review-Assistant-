
from services import answer, contradiction, labels, related_work, summarize
from services.llm import UNTRUSTED_TEXT_RULE


def test_generation_prompts_mark_paper_text_untrusted() -> None:
    for module in (answer, summarize, related_work, contradiction, labels):
        assert module.SYSTEM_PROMPT.endswith(UNTRUSTED_TEXT_RULE), module.__name__
    assert contradiction.CONFIRM_PROMPT.endswith(UNTRUSTED_TEXT_RULE)
    assert labels.system_prompt(None).endswith(UNTRUSTED_TEXT_RULE)
    assert labels.system_prompt("graph neural networks").endswith(UNTRUSTED_TEXT_RULE)
