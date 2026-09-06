"""
Deterministic quality metrics for generated stock-news summaries.

Everything here is rule-based: no judge model, no API key, no randomness.
Two runs over the same data give the same numbers, which is what makes the
results reportable.

Each function returns a dict so the caller can aggregate however it likes.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Sequence

# ---------------------------------------------------------------------------
# normalisation helpers
# ---------------------------------------------------------------------------

_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lowercase, strip punctuation noise, collapse whitespace."""
    text = text.lower()
    text = text.replace("\u2019", "'").replace("\u2014", " ").replace("\u2013", " ")
    text = re.sub(r"[^\w\s.%$/-]", " ", text)
    return _WS.sub(" ", text).strip()


def tokens(text: str) -> List[str]:
    return normalize(text).split()


def ngrams(seq: Sequence[str], n: int) -> List[tuple]:
    if len(seq) < n:
        return []
    return [tuple(seq[i : i + n]) for i in range(len(seq) - n + 1)]


# ---------------------------------------------------------------------------
# 1. numeric grounding
# ---------------------------------------------------------------------------

# matches 12, 12.5, 1,234, 45%, $3.2, 3.4B
_NUMBER = re.compile(r"\$?\d[\d,]*\.?\d*\s?%?[BMK]?", re.IGNORECASE)

_MULTIPLIER = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}


def _number_variants(raw: str) -> List[str]:
    """
    Generate surface forms a number might legitimately take in the source.
    '$1,234.50' -> {'1234.50', '1,234.50', '1234.5', ...}
    """
    core = raw.strip().lower().lstrip("$").rstrip("%").strip()
    suffix = ""
    if core and core[-1] in _MULTIPLIER:
        suffix = core[-1]
        core = core[:-1].strip()

    variants = {core, core.replace(",", "")}

    plain = core.replace(",", "")
    try:
        value = float(plain)
    except ValueError:
        return [v for v in variants if v]

    # trailing-zero forms: 3.50 -> 3.5, 3.0 -> 3
    if value == int(value):
        variants.add(str(int(value)))
        variants.add(f"{int(value):,}")
    variants.add(f"{value:g}")

    # expanded form for B/M/K suffixes
    if suffix:
        expanded = value * _MULTIPLIER[suffix]
        variants.add(f"{expanded:g}")
        variants.add(f"{int(expanded):,}")

    return [v for v in variants if v]


def numeric_grounding(summary: str, sources: str) -> Dict[str, Any]:
    """
    Every number the model states should be traceable to the source articles.
    Ungrounded numbers are the single clearest hallucination signal in
    financial summaries.
    """
    src = normalize(sources)
    src_nocomma = src.replace(",", "")

    found = [m.group(0) for m in _NUMBER.finditer(summary)]
    # drop bare years and list markers, which are rarely meaningful claims
    found = [f for f in found if not re.fullmatch(r"(19|20)\d{2}", f.strip())]

    grounded, ungrounded = [], []
    for raw in found:
        variants = _number_variants(raw)
        hit = any(v in src or v in src_nocomma for v in variants)
        (grounded if hit else ungrounded).append(raw.strip())

    total = len(found)
    return {
        "numbers_total": total,
        "numbers_grounded": len(grounded),
        "numeric_grounding_rate": (len(grounded) / total) if total else None,
        "ungrounded_numbers": ungrounded,
    }


# ---------------------------------------------------------------------------
# 2. ticker fidelity
# ---------------------------------------------------------------------------

# common all-caps words that look like tickers but aren't
_TICKER_STOPWORDS = {
    "AI", "CEO", "CFO", "COO", "CTO", "EPS", "ETF", "GDP", "IPO", "IRS",
    "SEC", "USA", "US", "UK", "EU", "FDA", "FED", "Q1", "Q2", "Q3", "Q4",
    "YOY", "QOQ", "NYSE", "NASDAQ", "AND", "THE", "FOR", "NEW", "ALL",
}

_TICKER = re.compile(r"\b[A-Z]{2,5}\b")


def ticker_fidelity(summary: str, requested: str, sources: str) -> Dict[str, Any]:
    """
    The summary should be about the ticker that was asked for, and shouldn't
    invent other tickers. Drifting onto a company that appears nowhere in the
    sources is a distinct and serious failure.
    """
    requested = requested.upper()
    src_upper = sources.upper()

    mentioned = {
        t for t in _TICKER.findall(summary)
        if t not in _TICKER_STOPWORDS and t != requested
    }
    invented = sorted(t for t in mentioned if t not in src_upper)

    return {
        "requested_ticker_present": requested in summary.upper(),
        "other_tickers_mentioned": sorted(mentioned),
        "invented_tickers": invented,
        # cleanliness is about not inventing companies. Writing "Apple"
        # rather than "AAPL" is normal prose, not an error.
        "ticker_clean": not invented,
    }


# ---------------------------------------------------------------------------
# 3. extractiveness
# ---------------------------------------------------------------------------

def extractiveness(summary: str, sources: str, n: int = 8) -> Dict[str, Any]:
    """
    Fraction of the summary's n-grams copied verbatim from the sources.

    Near 1.0 means the model is stitching sentences rather than summarising.
    Near 0.0 with poor grounding means it's freewheeling. Neither extreme is
    good; this is a diagnostic, not a score to maximise.
    """
    s_tokens, src_tokens = tokens(summary), tokens(sources)
    s_grams = ngrams(s_tokens, n)
    if not s_grams:
        return {"extractive_rate": None, "summary_tokens": len(s_tokens)}

    src_grams = set(ngrams(src_tokens, n))
    copied = sum(1 for g in s_grams if g in src_grams)
    return {
        "extractive_rate": copied / len(s_grams),
        "summary_tokens": len(s_tokens),
    }


# ---------------------------------------------------------------------------
# 4. degeneracy
# ---------------------------------------------------------------------------

def degeneracy(summary: str, n: int = 5) -> Dict[str, Any]:
    """
    Small quantized models loop. This catches the two common shapes: repeated
    n-grams, and a collapsed vocabulary.
    """
    toks = tokens(summary)
    grams = ngrams(toks, n)
    if not grams:
        return {
            "repeated_ngram_rate": None,
            "distinct_token_ratio": None,
            "degenerate": len(toks) < 20,
        }

    counts = Counter(grams)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    repeat_rate = repeated / len(grams)
    distinct_ratio = len(set(toks)) / len(toks)

    return {
        "repeated_ngram_rate": repeat_rate,
        "distinct_token_ratio": distinct_ratio,
        "degenerate": repeat_rate > 0.15 or distinct_ratio < 0.35 or len(toks) < 20,
    }


# ---------------------------------------------------------------------------
# 5. coverage
# ---------------------------------------------------------------------------

_PROPER = re.compile(r"\b[A-Z][a-zA-Z]{2,}\b")


def _fingerprint(text: str) -> set:
    """Distinctive content of a passage: its figures and its proper nouns."""
    figures = {m.group(0).strip().lower().lstrip("$").rstrip("%")
               for m in _NUMBER.finditer(text)}
    figures = {f for f in figures if f and not re.fullmatch(r"(19|20)\d{2}", f)}
    propers = {p.lower() for p in _PROPER.findall(text)}
    return figures | propers


def article_coverage(summary: str, articles: Iterable[str], n: int = 4) -> Dict[str, Any]:
    """
    How many of the retrieved articles left a trace in the summary.

    Matched on distinctive content (figures and proper nouns) rather than
    verbatim n-grams, so a properly paraphrased article still counts as
    covered. Low coverage across many articles usually means later articles
    are being truncated out of the context window.
    """
    articles = list(articles)
    s_print = _fingerprint(summary)
    if not articles or not s_print:
        return {"articles_total": len(articles), "articles_represented": 0,
                "coverage_rate": None}

    represented = 0
    for a in articles:
        a_print = _fingerprint(a)
        # ignore content the summary could have picked up from any article
        distinctive = a_print - set().union(*[
            _fingerprint(o) for o in articles if o is not a
        ] or [set()])
        target = distinctive or a_print
        if s_print & target:
            represented += 1
    return {
        "articles_total": len(articles),
        "articles_represented": represented,
        "coverage_rate": represented / len(articles),
    }


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------

def score_one(summary: str, articles: Sequence[str], ticker: str) -> Dict[str, Any]:
    """Run every metric over a single (summary, articles) pair."""
    sources = "\n\n".join(articles)
    result: Dict[str, Any] = {"ticker": ticker}
    result.update(numeric_grounding(summary, sources))
    result.update(ticker_fidelity(summary, ticker, sources))
    result.update(extractiveness(summary, sources))
    result.update(degeneracy(summary))
    result.update(article_coverage(summary, articles))
    return result
