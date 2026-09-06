"""Shared test setup.

config.py asserts that both API keys are present at import time, so dummy values
must be in the environment before any module that imports config is loaded.
These are placeholders — no test in this suite makes a real upstream API call.
"""
import os

os.environ.setdefault("ALPHAVANTAGE_API_KEY", "test-alphavantage-key")
os.environ.setdefault("NEWS_API_KEY", "test-news-key")
