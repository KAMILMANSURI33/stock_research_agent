# Stock News AI Agent

An intelligent agent system for analyzing and summarizing stock-related news using Llama-based LLM and multiple specialized sub-agents.

## Features

- Web search agent for gathering news articles
- Stock data agent for fetching market information
- News processing agent for article extraction and cleaning
- LLM-based summarization using Llama
- API interface for easy integration

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download a Llama model (e.g., Llama-2-7B-Q4) and place it in the `models` directory

3. Create a `.env` file with your API keys:
```
ALPHAVANTAGE_API_KEY=your_key_here
NEWS_API_KEY=your_key_here
```

4. Run the application:
```bash
python main.py
```

## Usage

The system exposes a FastAPI interface that can be accessed at `http://localhost:8000`.

Example API call:
```bash
curl -X POST "http://localhost:8000/analyze" -H "Content-Type: application/json" -d '{"symbol": "AAPL", "days": 1}'
```

## Architecture

The system uses a multi-agent architecture:
- Main Agent: Orchestrates the sub-agents and manages the workflow
- Web Search Agent: Gathers relevant news articles
- Stock Data Agent: Fetches market data and financial information
- News Processing Agent: Extracts and processes article content
- LLM Agent: Generates summaries using Llama model
