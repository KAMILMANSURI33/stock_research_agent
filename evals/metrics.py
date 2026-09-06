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
# 2. entity fidelity
# ---------------------------------------------------------------------------

# Words that are capitalised but are not organisations. This is the LAST
# filter, applied only to candidates that already survived the source check and
# the sentence-initial check, so it stays small and does not need to anticipate
# every acronym a new corpus might contain.
_NON_COMPANY = {
    # role / finance / filing vocabulary
    "CEO", "CFO", "COO", "CTO", "CIO", "EPS", "ETF", "GDP", "IPO", "IRS",
    "SEC", "FDA", "FED", "FOMC", "ISM", "PMI", "CPI", "PPI", "YOY", "QOQ",
    "EBITDA", "PBT", "ROI", "ROE", "KPI", "GAAP", "MOU", "IPO", "AGM",
    "Q1", "Q2", "Q3", "Q4", "FY", "H1", "H2",
    # technology vocabulary
    "AI", "ML", "API", "AWS", "GPU", "CPU", "TPU", "VRAM", "RAM", "OS",
    "PC", "IT", "SDK", "LLM", "SAAS", "IAAS", "PAAS", "HBM", "DRAM", "NAND",
    "5G", "4G", "VR", "AR", "IOT", "EV",
    # places, markets, misc
    "USA", "US", "UK", "EU", "UAE", "NYSE", "NASDAQ", "SP", "DJIA", "FTSE",
    "COVID", "WHO", "OPEC", "NATO", "GMT", "UTC", "ET", "EST", "PST", "PDT",
    "AM", "PM", "CET",
    # calendar
    "JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE", "JULY", "AUGUST",
    "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER",
    "MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY",
    "SUNDAY",
    # capitalised English that survives sentence-initial filtering
    "THE", "AND", "FOR", "NEW", "ALL", "ANALYSTS", "INVESTORS", "EARNINGS",
    "REVENUE", "CONCLUSION", "SUMMARY", "KEY", "FACTS", "METRICS", "OUTLOOK",
    # discourse markers: capitalised because they open a clause, never entities
    "ACCORDING", "HOWEVER", "OVERALL", "ADDITIONALLY", "MOREOVER",
    "FURTHERMORE", "MEANWHILE", "NEVERTHELESS", "THEREFORE", "CONSEQUENTLY",
    "FINALLY", "NOTABLY", "SEPARATELY", "ELSEWHERE", "DESPITE", "ALTHOUGH",
    "WHILE", "SINCE", "DURING", "FOLLOWING", "GIVEN", "THIS", "THAT", "THESE",
    "THOSE", "THERE", "THEIR", "THEY", "WITH", "WITHOUT", "AFTER", "BEFORE",
    "BOTH", "EACH", "MOST", "MANY", "SOME", "OTHER", "OTHERS", "SUCH",
    # generic market nouns that name no organisation on their own
    "INDEX", "AVERAGE", "COMPOSITE", "SHARE", "SHARES", "STOCK", "STOCKS",
    "MARKET", "MARKETS", "QUARTER", "GROWTH", "MARGIN", "MARGINS", "GUIDANCE",
    "PRICE", "TARGET", "SALES", "INCOME", "EQUITY", "MONEY", "BANK",
}

# Title Case (Apple, Nvidia) and ALL CAPS runs (AAPL, IBM) are both how a
# company can appear. Matching only ALL CAPS was what produced false positives
# on acronyms while missing "Apple" entirely.
# Match Title Case *sequences* as single units. Matching word by word split
# "Dow Jones Industrial Average" into four candidates, none of which is an
# organisation on its own, and each of which then failed the source check
# separately. Internal lowercase connectives are allowed so "Standard & Poor's"
# and "Bank of America" survive intact.
_TITLE_WORD = r"[A-Z][a-z]+(?:['\u2019][a-z]+)?"
_TITLE_PHRASE = re.compile(
    # Only "of" and "&" join a name across a lowercase token. Allowing "and"
    # merged "Apple and Bank of America" into a single phantom entity.
    rf"\b{_TITLE_WORD}(?:\s+(?:&|of)\s+{_TITLE_WORD}|\s+{_TITLE_WORD})*"
)
_ALL_CAPS = re.compile(r"\b[A-Z]{2,6}\b")

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_EDGE = ".,;:!?'\"()[]{}*-"


# Applied to every word, not just the last: "Microsoft's Azure" must normalise
# to "Microsoft Azure" or it never matches the sourced form.
_POSSESSIVE = re.compile(r"['\u2019]s\b")


def _strip_possessive(text: str) -> str:
    return _POSSESSIVE.sub("", text).strip()


def _expands_source_acronym(phrase: str, src_upper: str) -> bool:
    """True when the phrase spells out an acronym present in the sources.

    "Earnings Per Share" -> EPS, "Graphics Processing Unit" -> GPU. Writing a
    term out in full is a reference to it, not an invented entity, so flagging
    it as unsourced misreports ordinary prose as hallucination.
    """
    words = [w for w in re.split(r"[\s&]+", phrase) if w]
    if len(words) < 2:
        return False
    initials = "".join(w[0] for w in words).upper()
    if len(initials) < 2:
        return False
    return bool(re.search(rf"\b{re.escape(initials)}\b", src_upper))


def _sentence_initial_only(text: str, candidates: set) -> set:
    """Candidates that only ever appear as the first word of a sentence.

    "Investors will watch..." capitalises Investors for position, not because
    it names anything. A real entity almost always also appears mid-sentence.
    """
    initial, medial = set(), set()
    for sentence in _SENTENCE_SPLIT.split(text):
        words = [w.strip(_EDGE) for w in sentence.strip().split()]
        words = [w for w in words if w]
        if not words:
            continue
        initial.add(words[0])
        medial.update(words[1:])
    return {c for c in candidates if c in initial and c not in medial}


def entity_fidelity(summary: str, requested: str, sources: str) -> Dict[str, Any]:
    """Named entities in the summary that appear nowhere in the source text.

    The honest claim this supports is "the summary named something absent from
    its sources", not "the model invented a ticker": the extractor cannot tell
    a ticker from any other proper noun, and pretending otherwise is what made
    the previous version report acronyms as fabricated companies.
    """
    requested = requested.upper()
    src_upper = sources.upper()

    phrases = {m.group(0).strip() for m in _TITLE_PHRASE.finditer(summary)}
    candidates = {p for p in phrases if p} | set(_ALL_CAPS.findall(summary))

    # Possessives are the same entity: "Microsoft's" must not be counted
    # separately from "Microsoft", nor flagged when "Microsoft" is sourced.
    candidates = {_strip_possessive(c) for c in candidates}
    candidates = {c for c in candidates if c}

    # 1. anything the sources actually contain is sourced, by definition
    unsourced = {c for c in candidates if c.upper() not in src_upper}
    # 1b. an expansion of an acronym in the sources is sourced too:
    #     "Graphics Processing Unit" where the sources say GPU is a spelled-out
    #     reference, not a fabricated organisation.
    unsourced = {c for c in unsourced if not _expands_source_acronym(c, src_upper)}
    # 2. capitalised only because it starts a sentence. Applied to single words
    #    only: a multi-word phrase is not capitalised for position.
    single = {c for c in unsourced if " " not in c}
    unsourced -= _sentence_initial_only(summary, single)
    # 3. last resort: known non-company vocabulary
    # 3. last resort: known non-company vocabulary. A multi-word phrase counts
    #    as filtered only if every one of its words is non-company vocabulary,
    #    so "Bank of America" survives while "However The Company" does not.
    def _all_words_generic(candidate: str) -> bool:
        words = [w for w in re.split(r"[\s&]+", candidate) if w]
        return all(w.upper() in _NON_COMPANY for w in words)

    unsourced = {c for c in unsourced if not _all_words_generic(c)}
    unsourced.discard(requested)

    return {
        "requested_ticker_present": requested in summary.upper(),
        "unsourced_entities": sorted(unsourced),
        "entities_clean": not unsourced,
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
    result.update(entity_fidelity(summary, ticker, sources))
    result.update(extractiveness(summary, sources))
    result.update(degeneracy(summary))
    result.update(article_coverage(summary, articles))
    return result
