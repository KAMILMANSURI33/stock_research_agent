# Stock News Research Agent

A multi-agent system that gathers recent news and market data for a stock ticker and produces a grounded summary, served behind a FastAPI endpoint. Summarization runs on a locally-hosted TinyLlama (GGUF, 4-bit) via `llama-cpp` — no external LLM API calls, no per-request inference cost.

```
POST /analyze  {"symbol": "AAPL", "days": 1}

  Orchestrator
    ├── StockAgent      → market data (Alpha Vantage)
    ├── WebSearchAgent  → article search (News API) + full-text extraction (newspaper3k)
    └── LLMAgent        → summary grounded in the retrieved articles (TinyLlama, llama-cpp)

  → {stock_data, news_articles, summary, timestamp}
```

## Design notes

**One interface for every agent.** `BaseAgent` is an ABC defining `process`, `initialize`, `cleanup`, and state accessors. The orchestrator treats sub-agents uniformly, so adding a new source — filings, earnings transcripts, sentiment — needs no orchestrator changes.

**Async throughout, with retries.** Agents are `async` and the FastAPI layer is non-blocking. News API calls are wrapped in `tenacity` retry with exponential backoff (3 attempts, 4–10s), so transient upstream failures don't fail the request.

**Local inference by choice.** A 1B model at 4-bit quantization runs on consumer hardware. The trade is summary quality against zero API cost and no article content leaving the host — the right call for a tool that may run across a watchlist on a schedule.

**Prompt is structured, not freeform.** The summary prompt directs the model at market-moving events, financial metrics, sentiment, and outlook, rather than asking for an open-ended summary. Configurable in `config.py`.

## Results

<!-- FILL THIS IN. This section is why someone keeps reading. -->

Evaluated on __ articles across __ tickers:

| Metric | Score |
|---|---|
| Grounding (claims traceable to a source article) | __% |
| Ticker / figure accuracy | __% |
| Summary usefulness (1–5 rubric, mean) | __ |
| Median end-to-end latency | __ s |

Failure modes observed: __

## Setup

**1. Dependencies**

```bash
pip install -r requirements.txt
```

**2. Model** (~1.1 GB)

```bash
mkdir models
wget https://huggingface.co/TheBloke/TinyLlama-2-1b-miniguanaco-GGUF/resolve/main/tinyllama-2-1b-miniguanaco.Q4_K_M.gguf -P models/
```

**3. API keys** — create a `.env` in the project root:

```
ALPHAVANTAGE_API_KEY=your_key_here
NEWS_API_KEY=your_key_here
```

**4. Run**

```bash
python main.py
```

Service listens on `http://localhost:8000`; interactive docs at `/docs`.

## Usage

```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "days": 1}'
```

## Configuration

Tunable in `config.py`: `MAX_NEWS_ARTICLES` (default 10), `MAX_TOKENS` (2048), `TEMPERATURE` (0.7), `SUMMARY_MAX_LENGTH`, and the summary prompt template.

## Model

[TheBloke/TinyLlama-2-1b-miniguanaco-GGUF](https://huggingface.co/TheBloke/TinyLlama-2-1b-miniguanaco-GGUF), Q4_K_M quantization, ~1.1 GB, loaded with a 2048-token context on 4 threads.

## Tests

```bash
python -m pytest tests/ -v
```

Requires the model to be downloaded first.

## Limitations

- Summary quality is bounded by the 1B model; coherence degrades as the article set grows.
- Article extraction depends on source sites remaining scrapable by `newspaper3k`.
- Both upstream APIs are rate-limited on free tiers.

## Roadmap

- [ ] Run `StockAgent` and `WebSearchAgent` concurrently with `asyncio.gather` — they are independent but currently execute sequentially
- [ ] Response caching keyed on ticker + date window (`CACHE_DURATION` is already defined in config)
- [ ] Swap larger local models behind the same `LLMAgent` interface and compare quality against latency
