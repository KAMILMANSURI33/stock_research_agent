"""Tests for LLMAgent.process().

None of these load the model: the empty-article path must return before
initialize() is reached, and the populated path stubs the model out.
"""
import pytest

from agents.llm_agent import LLMAgent, NO_ARTICLES_SUMMARY


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

    assert result == NO_ARTICLES_SUMMARY.format(symbol="AAPL")
    assert "AAPL" in result
    assert agent.model is None, "model was loaded despite there being no articles"


@pytest.mark.asyncio
async def test_missing_articles_key_is_treated_as_empty():
    agent = LLMAgent()

    async def fail_initialize():
        raise AssertionError("model must not be loaded for an empty article set")

    agent.initialize = fail_initialize

    result = await agent.process({"symbol": "MSFT"})

    assert result == NO_ARTICLES_SUMMARY.format(symbol="MSFT")


@pytest.mark.asyncio
async def test_articles_present_still_reach_the_model():
    """The early return must not swallow the normal path."""
    agent = LLMAgent()
    seen = {}

    class FakeModel:
        def create_completion(self, prompt, **kwargs):
            seen["prompt"] = prompt
            return {"choices": [{"text": "  a real summary  "}]}

    agent.model = FakeModel()

    result = await agent.process(
        {"symbol": "AAPL", "articles": ["first body", "second body"]}
    )

    assert result == "a real summary"
    assert "first body" in seen["prompt"]
    assert "second body" in seen["prompt"]
