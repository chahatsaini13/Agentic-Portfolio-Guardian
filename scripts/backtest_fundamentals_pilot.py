"""
Fundamentals-only backtest of thesis-flip detection on
historical NSE data (stoicstatic/india-stock-data-nse-1990-2020).

HONEST SCOPING (see ADR for full writeup):
There is no historical NewsAPI archive, so this cannot replay "what news
would the LLM have seen at time T" the way the live agent does. Only
price-derived fundamentals features (volatility, drawdown, VaR/CVaR,
downside deviation, trailing momentum) are computed up to a cutoff date T
and fed to Ollama - this tests the fundamentals-reasoning half of the
Investment Thesis Agent, not the full news-aware system.

There is also no record of what a real investor's original thesis was for
any of these historical holdings. Each case uses a generic synthetic
"momentum continuation" framing instead of a real stated thesis - flagged
explicitly, not hidden.

Usage:
    # Step 1: find candidate cutoff dates for a ticker (both directions)
    python scripts/backtest_fundamentals_pilot.py --data-dir data/nse_historical --suggest TCS

    # Step 2: run the actual backtest on hand-picked cases
    python scripts/backtest_fundamentals_pilot.py --data-dir data/nse_historical \
        --cases TCS:2015-06-01 INFY:2018-03-01 RELIANCE:2016-11-01 \
        --lookback-days 252 --forward-days 126
"""

import argparse
import glob
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import requests

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.agents.portfolio_health_agent import (
    compute_volatility, compute_max_drawdown, compute_var, compute_cvar,
    compute_downside_deviation,
)

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1")

# Outcome classification thresholds on forward cumulative return -
# deliberately wide bands so "FLAT" isn't accidentally swallowing real
# moves. Not tuned against anything - a first-pass, documented as such.
GROWTH_THRESHOLD = 0.15
DECLINE_THRESHOLD = -0.15


def _find_csv(data_dir: str, ticker: str) -> str:
    """stoicstatic's dataset names files by NSE symbol - try a few
    reasonable filename patterns before giving up."""
    candidates = (
        glob.glob(os.path.join(data_dir, f"{ticker}.csv")) +
        glob.glob(os.path.join(data_dir, f"{ticker}*.csv")) +
        glob.glob(os.path.join(data_dir, "**", f"{ticker}.csv"), recursive=True)
    )
    if not candidates:
        raise FileNotFoundError(
            f"No CSV found for ticker '{ticker}' under {data_dir}. "
            f"Check the exact NSE symbol used as the filename in the dataset."
        )
    return candidates[0]


def load_price_series(data_dir: str, ticker: str) -> pd.Series:
    """Loads a stock's CSV and returns a Date-indexed Close price series.
    Handles the common column-naming variants in this dataset rather than
    assuming one exact schema."""
    path = _find_csv(data_dir, ticker)
    df = pd.read_csv(path)

    date_col = next((c for c in df.columns if c.strip().lower() == "date"), None)
    close_col = next((c for c in df.columns if c.strip().lower() in ("close", "last", "vwap")), None)
    if date_col is None or close_col is None:
        raise ValueError(
            f"Could not find Date/Close-like columns in {path}. "
            f"Columns present: {list(df.columns)} - update the detection "
            f"logic above for this file's actual schema."
        )

    df[date_col] = pd.to_datetime(df[date_col])
    df = df[[date_col, close_col]].dropna()
    df = df.sort_values(date_col).drop_duplicates(subset=date_col)
    series = df.set_index(date_col)[close_col].astype(float)
    series.name = ticker
    return series


def suggest_cutoffs(series: pd.Series, lookback_days: int = 252, forward_days: int = 126,
                     threshold: float = GROWTH_THRESHOLD, top_n: int = 5) -> pd.DataFrame:
    """Scans the price history for dates whose forward `forward_days`
    return is strongly positive or negative - candidate cutoffs for
    'clear later decline' / 'clear later growth' cases. Now also checks
    the LOOKBACK window (not just forward) for split artifacts, since a
    candidate can look forward-clean but still have a split hiding in the
    252-day window that compute_features_at_cutoff() will actually use."""
    fwd_return = series.shift(-forward_days) / series - 1
    candidates = fwd_return.dropna()
    strong = candidates[(candidates >= threshold) | (candidates <= -threshold)]

    rows = []
    for date, ret in strong.items():
        lookback_start = date - pd.Timedelta(days=int(lookback_days * 1.5))
        forward_end = date + pd.Timedelta(days=int(forward_days * 1.5))
        if contains_likely_split(series, max(lookback_start, series.index.min()),
                                  min(forward_end, series.index.max())):
            continue  # skip if EITHER lookback or forward window has a split
        rows.append({
            "date": date.date().isoformat(),
            "forward_return": round(float(ret), 4),
            "direction": "GROWTH" if ret > 0 else "DECLINE",
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out

    growth = out[out["direction"] == "GROWTH"].sort_values("forward_return", ascending=False).head(top_n)
    decline = out[out["direction"] == "DECLINE"].sort_values("forward_return").head(top_n)
    return pd.concat([growth, decline]).reset_index(drop=True)

SPLIT_JUMP_THRESHOLD = 0.35  # single-day |return| above this is treated as a
                              # likely stock split/bonus artifact, not a real
                              # one-day price move - this dataset is confirmed
                              # NOT split-adjusted (see ADR)


def contains_likely_split(series: pd.Series, start, end) -> bool:
    """True if any single-day return in [start, end] looks like a split/
    bonus artifact rather than a genuine price move. Confirmed necessary
    after HDFCBANK.csv showed a literal 5:1 split (2519.70 -> 505.10) on
    2011-07-14 with no corporate-action flag in the data - this dataset
    is NOT split-adjusted."""
    window = series[(series.index >= start) & (series.index <= end)]
    if len(window) < 2:
        return False
    daily_returns = window.pct_change().dropna()
    return bool((daily_returns.abs() > SPLIT_JUMP_THRESHOLD).any())

def compute_features_at_cutoff(series: pd.Series, cutoff: pd.Timestamp, lookback_days: int) -> dict:
    """Trailing-window fundamentals features, computed using the SAME
    functions portfolio_health_agent.py uses live - so this pilot is
    testing the same math the real agent runs, not a reimplementation."""
    window = series[series.index <= cutoff].tail(lookback_days)
    if contains_likely_split(series, window.index.min(), window.index.max()):
        raise ValueError(f"Lookback window for {cutoff.date()} contains a likely "
                        f"stock split/bonus artifact (single-day move >{SPLIT_JUMP_THRESHOLD*100:.0f}%). "
                        f"Pick a different cutoff or ticker - see ADR for why this dataset needs this guard.")
    if len(window) < lookback_days * 0.6:
        raise ValueError(f"Not enough trailing history before {cutoff.date()} "
                          f"({len(window)} days, wanted ~{lookback_days}).")

    returns = window.pct_change().dropna()
    trailing_cum_return = float(window.iloc[-1] / window.iloc[0] - 1)

    return {
        "cutoff_date": cutoff.date().isoformat(),
        "lookback_days_actual": len(window),
        "trailing_cumulative_return": round(trailing_cum_return, 4),
        "volatility_annualized": round(float(compute_volatility(returns)), 4),
        "max_drawdown": round(float(compute_max_drawdown(returns)), 4),
        "value_at_risk_95": round(compute_var(returns, 0.95), 4),
        "conditional_var_95": round(compute_cvar(returns, 0.95), 4),
        "downside_deviation": round(float(compute_downside_deviation(returns)), 4),
    }


def classify_actual_outcome(series: pd.Series, cutoff: pd.Timestamp, forward_days: int) -> dict:
    """What actually happened after T - the ground truth this pilot
    checks the LLM verdict against."""
    after = series[series.index > cutoff].head(forward_days)
    if contains_likely_split(series, cutoff, after.index.max()):
        raise ValueError(f"Forward window after {cutoff.date()} contains a likely "
                         f"stock split/bonus artifact. Pick a different cutoff or ticker.")
    if len(after) < forward_days * 0.6:
        raise ValueError(f"Not enough forward history after {cutoff.date()} to classify outcome "
                          f"({len(after)} days, wanted ~{forward_days}). Pick an earlier cutoff.")

    cutoff_price = series[series.index <= cutoff].iloc[-1]
    forward_return = float(after.iloc[-1] / cutoff_price - 1)

    if forward_return >= GROWTH_THRESHOLD:
        outcome = "GROWTH"
    elif forward_return <= DECLINE_THRESHOLD:
        outcome = "DECLINE"
    else:
        outcome = "FLAT"

    return {"forward_return": round(forward_return, 4), "actual_outcome": outcome}


def build_backtest_prompt(ticker: str, features: dict) -> str:
    """Simplified version of investment_thesis_agent.py's build_prompt() -
    same verdict shape (HOLDS/WEAKENING/BROKEN + reasoning), news block
    replaced with an explicit 'not available' note, and the thesis is a
    generic momentum-continuation framing (see module docstring) since no
    real historical investor thesis exists for these holdings."""
    thesis = (
        f"I expect {ticker}'s recent fundamental strength (price momentum, "
        f"volatility, and drawdown profile) to continue supporting the stock "
        f"going forward."
    )

    return f"""You are an investment thesis reviewer. Judge whether the following
fundamentals-based reasoning still holds, using ONLY the data given - no
external knowledge of what actually happened to this stock afterward.

TICKER: {ticker}
THESIS: "{thesis}"

FUNDAMENTALS AS OF {features['cutoff_date']} (trailing {features['lookback_days_actual']}-day window):
{json.dumps({k: v for k, v in features.items() if k not in ('cutoff_date', 'lookback_days_actual')}, indent=2)}

RECENT NEWS: Not available - historical NewsAPI archive does not exist for
this backtest. Base your judgment on the fundamentals trend alone.

Respond with ONLY a valid JSON object, no markdown fences, no commentary:
{{
  "thesis_status": "HOLDS" | "WEAKENING" | "BROKEN",
  "reasoning": "2-4 sentences explaining the verdict, referencing the specific fundamentals given"
}}
"""


def call_ollama(prompt: str) -> str:
    resp = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "options": {"num_predict": 400}},
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["response"]


def parse_verdict(raw: str) -> dict:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").replace("json\n", "", 1)
    try:
        parsed = json.loads(cleaned)
        return {"thesis_status": parsed.get("thesis_status", "PARSE_ERROR"),
                "reasoning": parsed.get("reasoning", "")}
    except json.JSONDecodeError:
        print("[warn] Model did not return clean JSON:\n", raw)
        return {"thesis_status": "PARSE_ERROR", "reasoning": raw.strip()}


def score_direction(verdict: str, actual: str) -> str:
    """HOLDS predicts continued strength (GROWTH); BROKEN predicts decline;
    WEAKENING is genuinely ambiguous and is NOT forced into a binary
    correct/incorrect - reported as its own category so the accuracy
    number isn't inflated by an arbitrary mapping. Document this choice
    in the ADR rather than silently picking a mapping that flatters the
    result."""
    if verdict == "HOLDS":
        return "correct" if actual == "GROWTH" else "incorrect"
    if verdict == "BROKEN":
        return "correct" if actual == "DECLINE" else "incorrect"
    if verdict == "WEAKENING":
        return "ambiguous_not_scored"
    return "unscored_parse_error"


def run_case(data_dir: str, ticker: str, cutoff_str: str, lookback_days: int, forward_days: int) -> dict:
    series = load_price_series(data_dir, ticker)
    cutoff = pd.Timestamp(cutoff_str)

    features = compute_features_at_cutoff(series, cutoff, lookback_days)
    outcome = classify_actual_outcome(series, cutoff, forward_days)

    prompt = build_backtest_prompt(ticker, features)
    raw = call_ollama(prompt)
    verdict = parse_verdict(raw)

    result = {
        "ticker": ticker,
        "cutoff_date": features["cutoff_date"],
        "features": features,
        "llm_verdict": verdict["thesis_status"],
        "llm_reasoning": verdict["reasoning"],
        "actual_forward_return": outcome["forward_return"],
        "actual_outcome": outcome["actual_outcome"],
        "direction_score": score_direction(verdict["thesis_status"], outcome["actual_outcome"]),
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Week 9 fundamentals-only backtest pilot")
    parser.add_argument("--data-dir", default="data/nse_historical")
    parser.add_argument("--suggest", help="Ticker to scan for candidate cutoff dates (both directions)")
    parser.add_argument("--cases", nargs="+", help="TICKER:YYYY-MM-DD pairs to actually run")
    parser.add_argument("--lookback-days", type=int, default=252)
    parser.add_argument("--forward-days", type=int, default=126)
    parser.add_argument("--log-path", default="results/week9_backtest_log.json")
    args = parser.parse_args()

    if args.suggest:
        series = load_price_series(args.data_dir, args.suggest)
        candidates = suggest_cutoffs(series, lookback_days=args.lookback_days, forward_days=args.forward_days)
        print(f"\nCandidate cutoffs for {args.suggest} (forward_days={args.forward_days}):")
        print(candidates.to_string(index=False) if not candidates.empty else "  none found at this threshold")
        return

    if not args.cases:
        parser.error("Pass --suggest TICKER to find cutoffs, or --cases TICKER:DATE ... to run the backtest.")

    results = []
    for case in args.cases:
        ticker, cutoff_str = case.split(":", 1)
        print(f"\n=== {ticker} @ {cutoff_str} ===")
        try:
            result = run_case(args.data_dir, ticker, cutoff_str, args.lookback_days, args.forward_days)
            results.append(result)
            print(json.dumps(result, indent=2))
        except Exception as e:
            print(f"[error] {ticker} @ {cutoff_str}: {e}")
            results.append({"ticker": ticker, "cutoff_date": cutoff_str, "error": str(e)})

    os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
    with open(args.log_path, "w") as f:
        json.dump({"run_at": datetime.now().isoformat(timespec="seconds"),
                   "lookback_days": args.lookback_days, "forward_days": args.forward_days,
                   "results": results}, f, indent=2)
    print(f"\nWrote {len(results)} case(s) -> {args.log_path}")

    scored = [r for r in results if r.get("direction_score") in ("correct", "incorrect")]
    correct = sum(1 for r in scored if r["direction_score"] == "correct")
    print(f"\nDirectionally scored: {correct}/{len(scored)} correct "
          f"(WEAKENING and errors excluded from this ratio - see log for full breakdown)")


if __name__ == "__main__":
    main()