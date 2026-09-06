"""Tests for Orchestrator.process().

These run without the model file: LLMAgent is never initialized, only its
process() coroutine is replaced.
"""
import asyncio
import time

import pytest

from agents.llm_agent import NO_ARTICLES_SUMMARY
from agents.orchestrator import Orchestrator

# Each stubbed sub-agent sleeps this long. Run sequentially the two independent
# agents would take 2x; run concurrently, ~1x.
AGENT_DELAY = 0.3

STOCK_RESULT = {"symbol": "AAPL", "price": 123.45}
NEWS_RESULT = [
    {"title": "Article one", "content": "first article body"},
    {"title": "Article two", "content": "second article body"},
]
SUMMARY_RESULT = "a summary"


def _make_orchestrator():
    """Build an Orchestrator with all three sub-agents stubbed out."""
    orchestrator = Orchestrator()
    llm_calls = []

    async def fake_stock(input_data):
        await asyncio.sleep(AGENT_DELAY)
        return STOCK_RESULT

    async def fake_news(input_data):
        await asyncio.sleep(AGENT_DELAY)
        return NEWS_RESULT

    async def fake_llm(input_data):
        llm_calls.append(input_data)
        return SUMMARY_RESULT

    orchestrator.stock_agent.process = fake_stock
    orchestrator.web_search_agent.process = fake_news
    orchestrator.llm_agent.process = fake_llm
    return orchestrator, llm_calls


@pytest.mark.asyncio
async def test_independent_agents_run_concurrently():
    """StockAgent and WebSearchAgent must overlap, not run back to back."""
    orchestrator, _ = _make_orchestrator()

    start = time.perf_counter()
    await orchestrator.process({"symbol": "AAPL", "days": 1})
    elapsed = time.perf_counter() - start

    sequential = AGENT_DELAY * 2
    # Concurrent execution lands near AGENT_DELAY; sequential lands near 2x.
    # The midpoint separates the two cleanly without being timing-flaky.
    assert elapsed < (AGENT_DELAY + sequential) / 2, (
        f"agents appear to run sequentially: {elapsed:.3f}s elapsed, "
        f"expected close to {AGENT_DELAY:.3f}s, sequential would be {sequential:.3f}s"
    )
    assert elapsed >= AGENT_DELAY, (
        f"{elapsed:.3f}s is faster than a single agent — stubs did not run"
    )


@pytest.mark.asyncio
async def test_results_map_to_correct_keys():
    """gather() returns results positionally — guard against a swapped unpack."""
    orchestrator, _ = _make_orchestrator()

    result = await orchestrator.process({"symbol": "AAPL", "days": 1})

    assert result["stock_data"] == STOCK_RESULT
    assert result["news_articles"] == NEWS_RESULT
    assert result["articles_retrieved"] == len(NEWS_RESULT)
    assert result["summary"] == SUMMARY_RESULT


@pytest.mark.asyncio
async def test_llm_receives_article_content_from_news_agent():
    """The LLM input is derived from the WebSearchAgent result, not the stock one."""
    orchestrator, llm_calls = _make_orchestrator()

    await orchestrator.process({"symbol": "AAPL", "days": 1})

    assert len(llm_calls) == 1
    assert llm_calls[0]["symbol"] == "AAPL"
    assert llm_calls[0]["articles"] == [a["content"] for a in NEWS_RESULT]


@pytest.mark.asyncio
async def test_timestamp_is_passed_through():
    orchestrator, _ = _make_orchestrator()

    result = await orchestrator.process(
        {"symbol": "AAPL", "days": 1, "timestamp": "2026-01-01T00:00:00"}
    )

    assert result["timestamp"] == "2026-01-01T00:00:00"


@pytest.mark.asyncio
async def test_sub_agent_failure_propagates():
    """A failing sub-agent must surface, not be swallowed into a partial result."""
    orchestrator, _ = _make_orchestrator()

    async def failing_stock(input_data):
        raise RuntimeError("upstream stock API is down")

    orchestrator.stock_agent.process = failing_stock

    with pytest.raises(RuntimeError, match="upstream stock API is down"):
        await orchestrator.process({"symbol": "AAPL", "days": 1})


@pytest.mark.asyncio
async def test_empty_news_yields_zero_count_and_no_generated_summary():
    """End to end with the real LLMAgent: no articles, no invented summary.

    LLMAgent is left unstubbed here on purpose -- the empty-article path must
    return before the model is ever loaded, so this needs no model file.
    """
    orchestrator, _ = _make_orchestrator()

    async def no_news(input_data):
        return []

    orchestrator.web_search_agent.process = no_news
    orchestrator.llm_agent = type(orchestrator.llm_agent)()  # fresh, unstubbed

    async def fail_initialize():
        raise AssertionError("model must not be loaded when there are no articles")

    orchestrator.llm_agent.initialize = fail_initialize

    result = await orchestrator.process({"symbol": "AAPL", "days": 1})

    assert result["articles_retrieved"] == 0
    assert result["news_articles"] == []
    assert result["summary"] == NO_ARTICLES_SUMMARY.format(symbol="AAPL")
    assert orchestrator.llm_agent.model is None
