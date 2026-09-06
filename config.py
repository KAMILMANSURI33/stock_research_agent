import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

assert ALPHAVANTAGE_API_KEY and NEWS_API_KEY, "Missing API keys in .env"

# LLM Configuration
LLAMA_MODEL_PATH = "models/tinyllama-2-1b-miniguanaco.Q4_K_M.gguf"

# Total context window. TinyLlama reports n_ctx_train = 2048; llama-cpp will
# accept a larger value, but with no rope scaling configured anything past the
# trained length extrapolates and the output degenerates. Measured at n_ctx
# 4096: a 253-token prompt stays coherent, a 3613-token prompt returns
# "l amp". A loud failure beats silent gibberish, so this stays at 2048.
N_CTX = 2048

# Generation budget, deducted from N_CTX. Previously both of these were one
# MAX_TOKENS = 2048 constant used as n_ctx *and* max_tokens, which reserved the
# entire window for output and left no room for the prompt.
MAX_OUTPUT_TOKENS = 512

# Headroom for tokenizer drift and the BOS token, so a prompt measured as
# fitting does not land one token over at generation time.
PROMPT_SAFETY_MARGIN = 128

# Per-article excerpt cap applied before prompt assembly. Replaces the unused
# SUMMARY_MAX_LENGTH, which was defined but referenced nowhere.
ARTICLE_CHAR_BUDGET = 1500

TEMPERATURE = 0.7

# Agent Configuration
MAX_NEWS_ARTICLES = 10
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