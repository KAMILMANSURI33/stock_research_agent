"""
Evaluation harness for the stock news research agent.

Hits the running service over a fixed ticker set, scores every summary with
deterministic metrics, and writes both the raw records and a markdown table.

    python main.py                       # in one terminal
    python evals/run_eval.py              # in another

    python evals/run_eval.py --tickers AAPL MSFT NVDA --days 3 --repeats 2

Outputs:
    evals/results/raw_<timestamp>.json    every request, response and score
    evals/results/summary.md              markdown table for the README
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import score_one  # noqa: E402

DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA", "JPM", "XOM", "PFE", "WMT"]
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def call_api(base_url: str, ticker: str, days: int, timeout: int) -> tuple[dict | None, float, str | None]:
    """Returns (payload, latency_seconds, error)."""
    start = time.perf_counter()
    try:
        response = requests.post(
            f"{base_url}/analyze",
            json={
                "symbol": ticker,
                "days": days,
                # Score against what the model actually received, not what was
                # retrieved. Without this the response omits prompt_context.
                "include_prompt_context": True,
            },
            timeout=timeout,
        )
        latency = time.perf_counter() - start
        if response.status_code != 200:
            return None, latency, f"HTTP {response.status_code}: {response.text[:200]}"
        return response.json(), latency, None
    except requests.exceptions.Timeout:
        return None, time.perf_counter() - start, f"timeout after {timeout}s"
    except requests.exceptions.ConnectionError:
        return None, time.perf_counter() - start, "connection refused - is the service running?"
    except Exception as exc:  # noqa: BLE001
        return None, time.perf_counter() - start, f"{type(exc).__name__}: {exc}"


class MissingPromptContext(RuntimeError):
    """The service did not return prompt_context."""


def extract_articles(payload: dict) -> list[str]:
    """The excerpts the model actually saw, in prompt order.

    Deliberately NOT news_articles. Articles are capped at ARTICLE_CHAR_BUDGET
    and whole articles are dropped to fit the context window, so news_articles
    includes text the model never received. Scoring grounding against it would
    credit a figure to a source the model never read.

    Missing prompt_context is fatal rather than a silent fallback: falling back
    would still produce numbers, just inflated ones, and nothing downstream
    would show that the run was scored against the wrong text.
    """
    if "prompt_context" not in payload:
        raise MissingPromptContext(
            "response has no prompt_context field. The service must support "
            "include_prompt_context and the request must set it to true. "
            "Refusing to fall back to news_articles, which would score "
            "grounding against text the model never received. "
            f"Response keys were: {sorted(payload)}"
        )
    return [a for a in payload["prompt_context"] if a and a.strip()]


def run(base_url: str, tickers: list[str], days: int, repeats: int, timeout: int) -> list[dict]:
    records = []
    total = len(tickers) * repeats

    for i, ticker in enumerate(tickers * repeats, start=1):
        print(f"[{i}/{total}] {ticker} ... ", end="", flush=True)
        payload, latency, error = call_api(base_url, ticker, days, timeout)

        if error:
            print(f"FAILED ({error})")
            records.append({
                "ticker": ticker, "ok": False, "error": error,
                "latency_s": round(latency, 2),
            })
            continue

        summary = payload.get("summary") or ""
        try:
            articles = extract_articles(payload)
        except MissingPromptContext as exc:
            print("FAILED (missing prompt_context)")
            records.append({
                "ticker": ticker, "ok": False, "error": str(exc),
                "latency_s": round(latency, 2),
            })
            continue

        retrieved = payload.get("articles_retrieved")
        used = payload.get("articles_used")

        if not summary.strip():
            print(f"EMPTY SUMMARY ({latency:.1f}s)")
            records.append({
                "ticker": ticker, "ok": False, "error": "empty summary",
                "latency_s": round(latency, 2), "articles_total": len(articles),
            })
            continue

        record = {"ticker": ticker, "ok": True, "latency_s": round(latency, 2)}
        record.update(score_one(summary, articles, ticker))
        record["articles_retrieved"] = retrieved
        record["articles_used"] = used
        record["drop_rate"] = (
            (retrieved - used) / retrieved
            if isinstance(retrieved, int) and isinstance(used, int) and retrieved
            else None
        )
        record["stock_data_source"] = payload.get("stock_data_source")
        record["summary"] = summary
        records.append(record)

        rate = record.get("numeric_grounding_rate")
        rate_str = "n/a" if rate is None else f"{rate:.0%}"
        flag = " DEGENERATE" if record.get("degenerate") else ""
        used_str = (
            f"{used}/{retrieved} articles used"
            if retrieved is not None else f"{len(articles)} articles"
        )
        print(f"ok ({latency:.1f}s, {used_str}, grounding {rate_str}){flag}")

    return records


def _mean(values: list[float]) -> float | None:
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def _median(values: list[float]) -> float | None:
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def _fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def aggregate(records: list[dict]) -> dict:
    ok = [r for r in records if r.get("ok")]
    failed = [r for r in records if not r.get("ok")]

    numbers_total = sum(r.get("numbers_total", 0) for r in ok)
    numbers_grounded = sum(r.get("numbers_grounded", 0) for r in ok)

    return {
        "runs_total": len(records),
        "runs_ok": len(ok),
        "runs_failed": len(failed),
        "articles_total": sum(r.get("articles_total", 0) for r in ok),
        "articles_retrieved_total": sum(r.get("articles_retrieved") or 0 for r in ok),
        "articles_used_total": sum(r.get("articles_used") or 0 for r in ok),
        "drop_rate_mean": _mean([r.get("drop_rate") for r in ok]),
        # pooled, not mean-of-means: a summary with 8 numbers should count
        # more than one with a single number
        "numeric_grounding_rate": (numbers_grounded / numbers_total) if numbers_total else None,
        "numbers_total": numbers_total,
        "ticker_clean_rate": _mean([1.0 if r.get("ticker_clean") else 0.0 for r in ok]),
        "coverage_rate": _mean([r.get("coverage_rate") for r in ok]),
        "extractive_rate": _mean([r.get("extractive_rate") for r in ok]),
        "degenerate_rate": _mean([1.0 if r.get("degenerate") else 0.0 for r in ok]),
        "latency_median_s": _median([r.get("latency_s") for r in records]),
        "latency_p90_s": (
            sorted(r["latency_s"] for r in records)[int(0.9 * len(records))]
            if len(records) >= 10 else None
        ),
        "ungrounded_examples": sorted({
            n for r in ok for n in r.get("ungrounded_numbers", [])
        })[:10],
        "failures": [{"ticker": r["ticker"], "error": r.get("error")} for r in failed],
    }


def to_markdown(agg: dict, days: int) -> str:
    lines = [
        "## Results",
        "",
        f"Evaluated on {agg['articles_total']} articles across "
        f"{agg['runs_total']} runs ({days}-day windows), "
        f"{agg['runs_ok']} completed successfully.",
        "",
        "| Metric | Score |",
        "|---|---|",
        f"| Numeric grounding (figures traceable to a source article) | "
        f"{_fmt_pct(agg['numeric_grounding_rate'])} ({agg['numbers_total']} figures checked) |",
        f"| Ticker fidelity (correct ticker, none invented) | {_fmt_pct(agg['ticker_clean_rate'])} |",
        f"| Article coverage (retrieved articles reflected in summary) | {_fmt_pct(agg['coverage_rate'])} |",
        f"| Extractive rate (8-grams copied verbatim) | {_fmt_pct(agg['extractive_rate'])} |",
        f"| Degenerate outputs (looping or collapsed) | {_fmt_pct(agg['degenerate_rate'])} |",
        f"| Article drop rate (retrieved but not sent to the model) | "
        f"{_fmt_pct(agg['drop_rate_mean'])} "
        f"({agg['articles_used_total']}/{agg['articles_retrieved_total']} used) |",
        f"| Median end-to-end latency | {agg['latency_median_s']:.1f} s |",
    ]
    if agg.get("latency_p90_s"):
        lines.append(f"| p90 latency | {agg['latency_p90_s']:.1f} s |")

    lines += ["", "**Failure modes observed:**", ""]
    if agg["ungrounded_examples"]:
        lines.append(
            "- Ungrounded figures appearing in summaries: "
            + ", ".join(f"`{n}`" for n in agg["ungrounded_examples"])
        )
    if agg["degenerate_rate"]:
        lines.append(
            f"- {_fmt_pct(agg['degenerate_rate'])} of summaries showed repetition "
            "or vocabulary collapse, the characteristic failure of the 1B model "
            "on long article sets."
        )
    if agg["failures"]:
        errors = {f["error"] for f in agg["failures"]}
        lines.append(f"- {len(agg['failures'])} run(s) failed: {'; '.join(sorted(errors))}")
    if agg.get("drop_rate_mean"):
        lines.append(
            f"- {_fmt_pct(agg['drop_rate_mean'])} of retrieved articles were dropped "
            "before generation to fit the 2048-token context window; grounding is "
            "scored only against the excerpts the model actually received."
        )
    if agg["coverage_rate"] is not None and agg["coverage_rate"] < 0.6:
        lines.append(
            "- Low article coverage suggests later articles are being truncated "
            "out of the 2048-token context window."
        )
    if not lines[-1].startswith("-"):
        lines.append("- None recorded.")

    lines += [
        "",
        f"_Reproduce with `python evals/run_eval.py --days {days}`. "
        "All metrics are deterministic and rule-based; no judge model is involved._",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=1,
                        help="runs per ticker; >1 measures run-to-run variance")
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"Evaluating {len(args.tickers)} tickers x {args.repeats} "
          f"against {args.base_url}\n")

    records = run(args.base_url, args.tickers, args.days, args.repeats, args.timeout)

    if not any(r.get("ok") for r in records):
        print("\nEvery run failed. Is the service up? Check: "
              f"curl {args.base_url}/docs")
        raw_path = RESULTS_DIR / f"raw_{stamp}.json"
        raw_path.write_text(json.dumps(records, indent=2))
        print(f"Raw records written to {raw_path}")
        return 1

    agg = aggregate(records)

    raw_path = RESULTS_DIR / f"raw_{stamp}.json"
    raw_path.write_text(json.dumps({"config": vars(args), "records": records,
                                    "aggregate": agg}, indent=2))

    markdown = to_markdown(agg, args.days)
    md_path = RESULTS_DIR / "summary.md"
    md_path.write_text(markdown + "\n")

    print("\n" + markdown)
    print(f"\nRaw records: {raw_path}")
    print(f"Markdown:    {md_path}  <- paste this over the Results section in README.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
