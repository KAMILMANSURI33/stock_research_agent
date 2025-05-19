import os
from dotenv import load_dotenv

# Load environment variables
print("Loading .env file...")
load_dotenv()

# API Keys
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

print(f"ALPHAVANTAGE_API_KEY: {ALPHAVANTAGE_API_KEY}")
print(f"NEWS_API_KEY: {NEWS_API_KEY}")

# LLM Configuration
LLAMA_MODEL_PATH = "models/tinyllama-2-1b-miniguanaco.Q4_K_M.gguf"
MAX_TOKENS = 2048
TEMPERATURE = 0.7

# Agent Configuration
MAX_NEWS_ARTICLES = 10
SUMMARY_MAX_LENGTH = 500
CACHE_DURATION = 3600  # 1 hour in seconds

# API Configuration
API_HOST = "0.0.0.0"
API_PORT = 8000

# Prompts
SUMMARY_PROMPT = """
Analyze the following news articles about {symbol} stock and create a concise summary *make it short and concise*:

{articles}

Focus on:
1. Key market-moving events
2. Important financial metrics
3. Market sentiment
4. Future outlook

Summary:
""" 