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

2. Set up the model:
   - Create a `models` directory in the project root:
   ```bash
   mkdir models
   ```
   - Download the TinyLlama model:
   ```bash
   # Option 1: Using wget
   wget https://huggingface.co/TheBloke/TinyLlama-2-1b-miniguanaco-GGUF/resolve/main/tinyllama-2-1b-miniguanaco.Q4_K_M.gguf -P models/

   # Option 2: Using curl
   curl -L https://huggingface.co/TheBloke/TinyLlama-2-1b-miniguanaco-GGUF/resolve/main/tinyllama-2-1b-miniguanaco.Q4_K_M.gguf --output models/tinyllama-2-1b-miniguanaco.Q4_K_M.gguf
   ```
   
   Note: The model file is approximately 1.1GB. Make sure you have sufficient disk space and a stable internet connection.

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

## Model Information

The project uses the TinyLlama-2-1b-miniguanaco model, which is a lightweight version of Llama optimized for efficiency:
- Model: TinyLlama-2-1b-miniguanaco
- Format: GGUF (Q4_K_M quantization)
- Size: ~1.1GB
- Source: [TheBloke/TinyLlama-2-1b-miniguanaco-GGUF](https://huggingface.co/TheBloke/TinyLlama-2-1b-miniguanaco-GGUF)

This model provides a good balance between performance and resource usage, making it suitable for running on consumer hardware.
