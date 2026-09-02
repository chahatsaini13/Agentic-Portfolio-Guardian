"""
Early Warning Agent - Portfolio Guardian 

Fourth agent in the pipeline. For every holding in the portfolio, this:
  1. Loads holdings via load_portfolio() (same loader Portfolio Health
     and Market Intelligence agents use).
  2. Pulls recent news per holding - reuses market_intelligence_agent.py's
     fetch_company_name()/fetch_news() as-is, same request shape as every
     other agent's news fetch (see investment_thesis_agent.py's original).
  3. Runs each article through redflag_detector.redflag_score() (rule-based
     placeholder now, Member 2's contrastive anomaly detector later - see
     that module's docstring) and classify_flag_type() for a category label.
  4. For every article that clears the red-flag threshold, sends it +
     portfolio context to local Ollama and asks for a severity/urgency
     score (LOW/MEDIUM/HIGH) with brief reasoning.
  5. Outputs one structured alert per flagged article:
     {ticker, flag_type, headline, severity, reasoning}.

No LangGraph here yet, same reasoning as ADR 0001 - build the agent logic
standalone first, wrap it as a graph node later.

Usage:
    python -m src.agents.early_warning_agent --file data/sample_portfolio.csv --limit 3
"""

import argparse
import json
import os

import requests
from dotenv import load_dotenv

from src.agents.portfolio_health_agent import load_portfolio
from src.agents.market_intelligence_agent import fetch_company_name, fetch_news
from src.agents.redflag_detector import redflag_score, is_red_flag, classify_flag_type

load_dotenv()

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1")


def detect_flags(news: list) -> list:
    """Score every article, keep only the ones over threshold, tag each
    with its flag_type. Annotate-in-place style, same as
    market_intelligence_agent.py's tag_news_sentiment()."""
    flagged = []
    for article in news:
        text = f"{article.get('title') or ''}. {article.get('description') or ''}".strip()
        score = redflag_score(text)
        if is_red_flag(text):
            flagged.append({
                **article,
                "redflag_score": score,
                "flag_type": classify_flag_type(text),
            })
    return flagged


def build_severity_prompt(ticker: str, company: str, article: dict) -> str:
    """One flagged article in, severity verdict out. Deliberately narrow -
    one article per call, not the whole batch - so a bad/ambiguous article
    can't drag down the severity call for the others."""
    return f"""You are a risk analyst flagging potential red-flag events for a retail
investor's portfolio holding. You are given ONE news item that a keyword
filter has already flagged as a possible red flag (management change,
litigation, credit downgrade, fraud, or similar). Judge how severe/urgent
this is for someone holding this stock - not whether to buy/sell.

TICKER: {ticker}
COMPANY: {company}
SUSPECTED FLAG TYPE: {article.get('flag_type')}

HEADLINE: {article.get('title')}
DESCRIPTION: {article.get('description')}
SOURCE: {article.get('source')}
PUBLISHED: {article.get('published_at')}

Respond with ONLY a valid JSON object, no markdown fences, no commentary
before or after, in exactly this shape:
{{
  "severity": "LOW" | "MEDIUM" | "HIGH",
  "reasoning": "1-3 sentences explaining the severity call"
}}
"""


def call_ollama(prompt: str) -> str:
    """Same call shape as every other agent's call_ollama()."""
    resp = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["response"]


def parse_severity_output(raw: str) -> dict:
    """Same strip-then-parse-then-fail-loudly approach as every other
    agent's parse function. Falls back to MEDIUM rather than silently
    dropping the alert - an unparsed severity is still a flagged event,
    better to surface it at a middling severity than to lose it."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json\n", "", 1)

    try:
        parsed = json.loads(cleaned)
        return {
            "severity": parsed.get("severity", "MEDIUM"),
            "reasoning": parsed.get("reasoning", ""),
        }
    except json.JSONDecodeError:
        print("[warn] Model did not return clean JSON. Raw output below:\n")
        print(raw)
        return {"severity": "MEDIUM", "reasoning": raw.strip()}


def score_severity(ticker: str, company: str, article: dict) -> dict:
    prompt = build_severity_prompt(ticker, company, article)
    try:
        raw_output = call_ollama(prompt)
        return parse_severity_output(raw_output)
    except requests.exceptions.ConnectionError as e:
        print(f"  [warn] could not reach Ollama: {e}")
        return {"severity": "MEDIUM", "reasoning": "Severity unavailable - could not reach local Ollama server."}


def run_for_holding(ticker: str) -> list:
    """One holding, start to finish. Returns a list of structured alerts
    (empty list if nothing was flagged) - this is the shape the
    Orchestrator will eventually consume alongside the other 3 agents."""
    print(f"  [1/4] Resolving company name for {ticker} ...")
    company = fetch_company_name(ticker)

    print(f"  [2/4] Fetching news for {company} ...")
    news = fetch_news(company)

    print(f"  [3/4] Running red-flag detection on {len(news)} article(s) ...")
    flagged = detect_flags(news)
    print(f"        {len(flagged)} article(s) flagged.")

    alerts = []
    for i, article in enumerate(flagged, 1):
        print(f"  [4/4] Scoring severity for flag {i}/{len(flagged)} via Ollama ({OLLAMA_MODEL}) ...")
        severity = score_severity(ticker, company, article)
        alerts.append({
            "ticker": ticker,
            "flag_type": article.get("flag_type"),
            "headline": article.get("title"),
            "severity": severity["severity"],
            "reasoning": severity["reasoning"],
        })

    return alerts


def run(filepath: str, limit: int = None) -> list:
    holdings = load_portfolio(filepath)
    if limit:
        holdings = holdings[:limit]

    all_alerts = []
    for h in holdings:
        print(f"\n=== {h['ticker']} ===")
        all_alerts.extend(run_for_holding(h["ticker"]))

    return all_alerts


def main():
    parser = argparse.ArgumentParser(description="Early Warning Agent")
    parser.add_argument("--file", default="data/sample_portfolio.csv",
                         help="Path to portfolio CSV (same format as the other agents)")
    parser.add_argument("--limit", type=int, default=3,
                         help="Only process the first N holdings (default 3, for a quick test run)")
    args = parser.parse_args()

    alerts = run(args.file, limit=args.limit)

    print("\n" + "=" * 60)
    print("EARLY WARNING ALERTS")
    print("=" * 60)
    print(json.dumps(alerts, indent=2))


if __name__ == "__main__":
    main()