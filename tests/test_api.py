"""Tests for the /analyze endpoint.

Split in two:

* The in-process tests drive the real FastAPI app through TestClient with the
  orchestrator stubbed. They need no running server and no model file.
* test_live_server_analyze talks to an actual server on localhost. It is marked
  `integration` and FAILS when nothing is listening — it never passes silently.
  Deselect it with:  pytest tests/ -m "not integration"
"""
import pytest
import requests
from fastapi.testclient import TestClient

import main

LIVE_SERVER_URL = "http://127.0.0.1:8000/analyze"

STOCK_DATA = {"symbol": "AAPL", "price": 123.45}
NEWS_ARTICLES = [{"title": "Article one", "content": "body"}]
SUMMARY = "a summary"


@pytest.fixture
def client(monkeypatch):
    """TestClient over the real app, with the orchestrator stubbed out."""

    async def fake_initialize():
        return None

    async def fake_cleanup():
        return None

    async def fake_process(input_data):
        return {
            "stock_data": STOCK_DATA,
            "news_articles": NEWS_ARTICLES,
            "articles_retrieved": len(NEWS_ARTICLES),
            "articles_used": len(NEWS_ARTICLES),
            "articles_used_indices": list(range(len(NEWS_ARTICLES))),
            "summary": SUMMARY,
            "timestamp": input_data.get("timestamp"),
        }

    monkeypatch.setattr(main.orchestrator, "initialize", fake_initialize)
    monkeypatch.setattr(main.orchestrator, "cleanup", fake_cleanup)
    monkeypatch.setattr(main.orchestrator, "process", fake_process)

    with TestClient(main.app) as test_client:
        yield test_client


def test_analyze_returns_expected_payload(client):
    response = client.post("/analyze", json={"symbol": "AAPL", "days": 3})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["stock_data"] == STOCK_DATA
    assert body["news_articles"] == NEWS_ARTICLES
    assert body["summary"] == SUMMARY
    assert body["articles_retrieved"] == len(NEWS_ARTICLES)
    assert body["articles_used"] == len(NEWS_ARTICLES)
    assert body["articles_used_indices"] == list(range(len(NEWS_ARTICLES)))
    assert body["timestamp"]


def test_symbol_reaches_the_orchestrator(client, monkeypatch):
    seen = {}

    async def capturing_process(input_data):
        seen.update(input_data)
        return {
            "stock_data": STOCK_DATA,
            "news_articles": NEWS_ARTICLES,
            "articles_retrieved": len(NEWS_ARTICLES),
            "articles_used": len(NEWS_ARTICLES),
            "articles_used_indices": list(range(len(NEWS_ARTICLES))),
            "summary": SUMMARY,
            "timestamp": input_data.get("timestamp"),
        }

    monkeypatch.setattr(main.orchestrator, "process", capturing_process)

    client.post("/analyze", json={"symbol": "MSFT", "days": 5})

    assert seen["symbol"] == "MSFT"
    assert seen["days"] == 5


def test_days_defaults_to_one(client, monkeypatch):
    seen = {}

    async def capturing_process(input_data):
        seen.update(input_data)
        return {
            "stock_data": STOCK_DATA,
            "news_articles": NEWS_ARTICLES,
            "articles_retrieved": len(NEWS_ARTICLES),
            "articles_used": len(NEWS_ARTICLES),
            "articles_used_indices": list(range(len(NEWS_ARTICLES))),
            "summary": SUMMARY,
            "timestamp": input_data.get("timestamp"),
        }

    monkeypatch.setattr(main.orchestrator, "process", capturing_process)

    client.post("/analyze", json={"symbol": "AAPL"})

    assert seen["days"] == 1


def test_zero_days_is_rejected(client):
    response = client.post("/analyze", json={"symbol": "AAPL", "days": 0})

    assert response.status_code == 400
    assert "Days must be greater than 0" in response.text


def test_missing_symbol_is_rejected(client):
    response = client.post("/analyze", json={"days": 1})

    assert response.status_code == 422  # pydantic rejects before the handler


def test_malformed_json_is_rejected(client):
    response = client.post(
        "/analyze",
        content="{not json",
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 422


def test_orchestrator_failure_returns_500(client, monkeypatch):
    async def failing_process(input_data):
        raise RuntimeError("upstream is down")

    monkeypatch.setattr(main.orchestrator, "process", failing_process)

    response = client.post("/analyze", json={"symbol": "AAPL", "days": 1})

    assert response.status_code == 500


def test_sensitive_headers_are_redacted():
    """Credential-bearing headers must never reach the logs in the clear."""
    safe = main._safe_headers(
        {
            "Authorization": "Bearer supersecret",
            "Cookie": "session=abc123",
            "X-API-Key": "key-material",
            "Content-Type": "application/json",
        }
    )

    assert safe["Authorization"] == "<redacted>"
    assert safe["Cookie"] == "<redacted>"
    assert safe["X-API-Key"] == "<redacted>"
    assert safe["Content-Type"] == "application/json"
    assert "supersecret" not in str(safe)
    assert "abc123" not in str(safe)


@pytest.mark.integration
def test_live_server_analyze():
    """Hits a real server. Fails loudly if one is not running."""
    try:
        response = requests.post(
            LIVE_SERVER_URL,
            json={"symbol": "AAPL", "days": 3},
            timeout=120,
        )
    except requests.exceptions.RequestException as exc:
        pytest.fail(f"could not reach {LIVE_SERVER_URL}: {exc}")

    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) >= {
        "stock_data", "news_articles", "articles_retrieved",
        "articles_used", "articles_used_indices", "summary", "timestamp",
    }
    assert body["articles_retrieved"] == len(body["news_articles"])
    assert body["articles_used"] <= body["articles_retrieved"]
    assert len(body["articles_used_indices"]) == body["articles_used"]
    assert all(0 <= i < body["articles_retrieved"] for i in body["articles_used_indices"])
    assert isinstance(body["news_articles"], list)
    assert isinstance(body["summary"], str) and body["summary"].strip()


def test_prompt_context_absent_by_default(client):
    """Production responses must be unchanged by the evaluation opt-in."""
    response = client.post("/analyze", json={"symbol": "AAPL", "days": 1})

    assert response.status_code == 200, response.text
    assert "prompt_context" not in response.json()


def test_prompt_context_returned_when_requested(client, monkeypatch):
    excerpts = ["excerpt one", "excerpt two"]

    async def process_with_context(input_data):
        assert input_data["include_prompt_context"] is True
        return {
            "stock_data": STOCK_DATA,
            "news_articles": NEWS_ARTICLES,
            "articles_retrieved": len(NEWS_ARTICLES),
            "articles_used": len(excerpts),
            "articles_used_indices": list(range(len(excerpts))),
            "prompt_context": excerpts,
            "summary": SUMMARY,
            "timestamp": input_data.get("timestamp"),
        }

    monkeypatch.setattr(main.orchestrator, "process", process_with_context)

    response = client.post(
        "/analyze",
        json={"symbol": "AAPL", "days": 1, "include_prompt_context": True},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["prompt_context"] == excerpts
    assert len(body["prompt_context"]) == body["articles_used"]


def test_include_prompt_context_defaults_to_false(client, monkeypatch):
    seen = {}

    async def capturing_process(input_data):
        seen.update(input_data)
        return {
            "stock_data": STOCK_DATA,
            "news_articles": NEWS_ARTICLES,
            "articles_retrieved": len(NEWS_ARTICLES),
            "articles_used": len(NEWS_ARTICLES),
            "articles_used_indices": list(range(len(NEWS_ARTICLES))),
            "summary": SUMMARY,
            "timestamp": input_data.get("timestamp"),
        }

    monkeypatch.setattr(main.orchestrator, "process", capturing_process)

    client.post("/analyze", json={"symbol": "AAPL", "days": 1})

    assert seen["include_prompt_context"] is False
