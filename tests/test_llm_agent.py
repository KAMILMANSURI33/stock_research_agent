"""Tests for LLMAgent.process() and its prompt budgeting.

None of these load the real model. The empty-article path must return before
initialize() is reached; every other path substitutes a fake model that
tokenizes deterministically, so budgeting is exercised without a 637MB load.
"""
import pytest

from agents.llm_agent import (
    LLMAgent,
    NO_ARTICLES_SUMMARY,
    PromptTooLargeError,
)
from config import ARTICLE_CHAR_BUDGET


class _FakeModel:
    """Stand-in for Llama with a predictable tokenizer.

    chars_per_token mirrors roughly what the real tokenizer does for English
    prose; setting it to 1 makes any prompt overflow, which is how the
    refuse-to-generate path is reached without a huge fixture.
    """

    def __init__(self, seen=None, chars_per_token=4):
        self.seen = seen if seen is not None else {}
        self.chars_per_token = chars_per_token

    def tokenize(self, raw: bytes):
        return [0] * (len(raw) // self.chars_per_token + 1)

    def create_completion(self, prompt, **kwargs):
        self.seen["prompt"] = prompt
        self.seen["kwargs"] = kwargs
        return {"choices": [{"text": "  a real summary  "}]}


def _article(marker: str, size: int = ARTICLE_CHAR_BUDGET) -> str:
    """An article of a known size, tagged so it can be found in a prompt."""
    return f"[{marker}]" + ("x" * (size - len(marker) - 2))


@pytest.mark.asyncio
async def test_empty_articles_produce_no_generated_summary():
    """With nothing to summarise the model must not be consulted at all.

    Given an empty prompt the model does not decline -- it invents figures.
    A captured pre-fix run produced 1009 words of fabricated Apple financials,
    including two contradictory revenue totals for the same year.
    """
    agent = LLMAgent()

    async def fail_initialize():
        raise AssertionError("model must not be loaded for an empty article set")

    agent.initialize = fail_initialize

    result = await agent.process({"symbol": "AAPL", "articles": []})

    assert result["summary"] == NO_ARTICLES_SUMMARY.format(symbol="AAPL")
    assert result["articles_used"] == 0
    assert result["articles_used_indices"] == []
    assert result["prompt_context"] == []
    assert agent.model is None, "model was loaded despite there being no articles"


@pytest.mark.asyncio
async def test_missing_articles_key_is_treated_as_empty():
    agent = LLMAgent()

    async def fail_initialize():
        raise AssertionError("model must not be loaded for an empty article set")

    agent.initialize = fail_initialize

    result = await agent.process({"symbol": "MSFT"})

    assert result["summary"] == NO_ARTICLES_SUMMARY.format(symbol="MSFT")
    assert result["articles_used"] == 0


@pytest.mark.asyncio
async def test_articles_present_still_reach_the_model():
    """The early return must not swallow the normal path."""
    agent = LLMAgent()
    seen = {}
    agent.model = _FakeModel(seen)

    result = await agent.process(
        {"symbol": "AAPL", "articles": ["first body", "second body"]}
    )

    assert result["summary"] == "a real summary"
    assert result["articles_used"] == 2
    assert result["articles_used_indices"] == [0, 1]
    assert "first body" in seen["prompt"]
    assert "second body" in seen["prompt"]


@pytest.mark.asyncio
async def test_six_large_articles_are_truncated_to_fit():
    """The real failure case: 6 full articles overflowed a 2048-token window.

    Live, this produced
    "Requested tokens (10670) exceed context window of 2048" and a 500.
    """
    agent = LLMAgent()
    seen = {}
    agent.model = _FakeModel(seen)
    articles = [_article(f"A{i}") for i in range(6)]

    result = await agent.process({"symbol": "AAPL", "articles": articles})

    assert result["summary"] == "a real summary"
    assert 0 < result["articles_used"] < 6, "expected some articles to be dropped"
    assert agent.count_tokens(seen["prompt"]) <= agent.prompt_token_budget()
    assert seen["kwargs"]["max_tokens"] == 512


def test_prompt_budget_is_never_exceeded():
    """Whatever build_prompt returns must fit, for every article count."""
    agent = LLMAgent()
    agent.model = _FakeModel()
    budget = agent.prompt_token_budget()

    for n in range(1, 13):
        prompt, used, excerpts = agent.build_prompt(
            "AAPL", [_article(f"A{i}") for i in range(n)]
        )
        assert agent.count_tokens(prompt) <= budget, f"{n} articles overflowed"
        assert len(used) <= n
        assert used == sorted(set(used)), "indices must be unique and ordered"
        assert len(excerpts) == len(used)


def test_budget_leaves_room_for_the_full_output():
    """Prompt + output must fit the window, which is what the old code broke."""
    agent = LLMAgent()
    from config import N_CTX, MAX_OUTPUT_TOKENS

    assert agent.prompt_token_budget() + MAX_OUTPUT_TOKENS < N_CTX


def test_refuses_when_even_one_article_cannot_fit():
    """Rather than let llama-cpp raise ValueError partway through generation."""
    agent = LLMAgent()
    agent.model = _FakeModel(chars_per_token=1)  # every char a token

    with pytest.raises(PromptTooLargeError) as exc:
        agent.build_prompt("AAPL", [_article("A0")])

    message = str(exc.value)
    assert "tokens needed" in message
    assert "ARTICLE_CHAR_BUDGET" in message


@pytest.mark.asyncio
async def test_used_indices_match_what_is_actually_in_the_prompt():
    """The exposed indices must name exactly the articles the model saw.

    This is what grounding gets scored against: if a dropped article were
    reported as used, a figure the model invented could be matched back to text
    the model never received and counted as sourced.
    """
    agent = LLMAgent()
    seen = {}
    agent.model = _FakeModel(seen)
    articles = [_article(f"A{i}") for i in range(6)]

    result = await agent.process({"symbol": "AAPL", "articles": articles})

    used = result["articles_used_indices"]
    assert used, "expected at least one article to survive truncation"
    assert len(used) == result["articles_used"]
    assert used == sorted(set(used))

    prompt = seen["prompt"]
    for i in range(len(articles)):
        marker = f"[A{i}]"
        if i in used:
            assert marker in prompt, f"article {i} reported used but absent from prompt"
        else:
            assert marker not in prompt, f"article {i} dropped but present in prompt"


@pytest.mark.asyncio
async def test_prompt_context_is_the_exact_text_the_model_received():
    """Including ARTICLE_CHAR_BUDGET prefix truncation, not the full article.

    A harness scoring grounding against the untruncated article would credit
    text the model never saw, so the excerpt has to be reported verbatim.
    """
    agent = LLMAgent()
    seen = {}
    agent.model = _FakeModel(seen)
    # deliberately longer than the per-article cap so truncation is visible
    oversized = [_article(f"A{i}", size=ARTICLE_CHAR_BUDGET * 2) for i in range(6)]

    result = await agent.process({"symbol": "AAPL", "articles": oversized})

    context = result["prompt_context"]
    indices = result["articles_used_indices"]

    assert len(context) == len(indices) == result["articles_used"]

    for position, article_index in enumerate(indices):
        excerpt = context[position]
        full = oversized[article_index]
        # truncated to the cap, and a genuine prefix of the original
        assert len(excerpt) == ARTICLE_CHAR_BUDGET
        assert full.startswith(excerpt)
        assert excerpt != full, "expected the excerpt to be shorter than the article"
        # and it is what actually went into the prompt
        assert excerpt in seen["prompt"]
