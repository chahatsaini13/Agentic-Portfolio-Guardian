"""
Market Intelligence Agent - Portfolio Guardian 

Third agent in the pipeline. For every holding in the portfolio, this:
  1. Loads holdings via load_portfolio() (same loader Portfolio Health
     Agent uses - one CSV, one contract, no duplicate parsing logic).
  2. Pulls recent news per holding (NewsAPI), same request shape as
     investment_thesis_agent.py's fetch_news() - just without a thesis
     to keyword-filter on, since this agent isn't checking a thesis,
     it's summarizing "what's going on" for a holding.
  3. Tags each article's sentiment via sentiment_tagger.tag_sentiment()
     (placeholder now, Member 2's contrastive model later - see that
     module's docstring).
  4. Sends the holding's news + sentiment tags to local Ollama and asks
     for a plain-language summary of what's relevant/changed.

No LangGraph here yet, same reasoning as ADR 0001 - build the agent logic
standalone first, wrap it as a graph node later.

Usage:
    python -m src.agents.market_intelligence_agent --file data/sample_portfolio.csv --limit 3
"""

import argparse
import json
import os

import requests
import yfinance as yf
from dotenv import load_dotenv

from src.agents.portfolio_health_agent import load_portfolio
from src.agents.news_filter import filter_news_by_relevance
from src.agents.sentiment_tagger import tag_sentiment

# filter_news_by_relevance() takes any two texts to embed and compare -
# it was written for (thesis, news_text) in Week 3, but nothing about it
# is thesis-specific. Here the "anchor" is just the company name, so it's
# filtering for "is this article actually about this company" rather than
# "is this article relevant to a specific thesis". Company-name-only
# anchors tend to score lower than a full thesis sentence would (less
# text for the embedding to latch onto), so this threshold is looser than
# news_filter.py's DEFAULT_RELEVANCE_THRESHOLD (0.5) - retune once you've
# eyeballed real output, same as Week 3.
NEWS_RELEVANCE_THRESHOLD = 0.3

load_dotenv()

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1")
NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY")


def fetch_company_name(ticker: str) -> str:
    """yfinance long/short name for a ticker, falls back to the ticker
    itself if the lookup fails - one bad ticker shouldn't kill the run."""
    try:
        info = yf.Ticker(ticker).info
        return info.get("longName") or info.get("shortName") or ticker.split(".")[0]
    except Exception as e:
        print(f"[warn] could not fetch company name for {ticker}: {e}")
        return ticker.split(".")[0]


def fetch_news(company: str, page_size: int = 5) -> list:
    """Pull recent headlines for a company. Same NewsAPI request shape as
    investment_thesis_agent.py's fetch_news() - no thesis keywords here
    since this agent summarizes general developments, not thesis-fit."""
    if not NEWSAPI_KEY:
        print("[warn] NEWSAPI_KEY not set - skipping news for this holding.")
        return []

    resp = requests.get(
        "https://newsapi.org/v2/everything",
        params={
            "q": company,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": page_size,
            "apiKey": NEWSAPI_KEY,
        },
        timeout=10,
    )

    if resp.status_code != 200:
        print(f"[warn] NewsAPI returned {resp.status_code}: {resp.text[:200]}")
        return []

    articles = resp.json().get("articles", [])
    articles = [
        a for a in articles
        if company.lower().split()[0] in (
            (a.get("title") or "") + " " + (a.get("description") or "")
        ).lower()
    ]

    return [
        {
            "title": a.get("title"),
            "description": a.get("description"),
            "source": (a.get("source") or {}).get("name"),
            "published_at": a.get("publishedAt"),
        }
        for a in articles
    ]


def tag_news_sentiment(news: list) -> list:
    """Attach a sentiment tag to each article, same annotate-in-place
    style as news_filter.py's relevance scores."""
    tagged = []
    for article in news:
        text = f"{article.get('title') or ''}. {article.get('description') or ''}".strip()
        tagged.append({**article, "sentiment": tag_sentiment(text)})
    return tagged


def aggregate_sentiment(tagged_news: list) -> str:
    """Majority vote across article-level tags. No news, or a tie ->
    neutral - safer default than guessing a direction with no evidence."""
    if not tagged_news:
        return "neutral"

    counts = {"positive": 0, "negative": 0, "neutral": 0}
    for article in tagged_news:
        counts[article["sentiment"]] += 1

    top_count = max(counts.values())
    leaders = [label for label, c in counts.items() if c == top_count]
    return leaders[0] if len(leaders) == 1 else "neutral"


def build_prompt(ticker: str, company: str, tagged_news: list) -> str:
    """One prompt: news + sentiment tags in, plain-language summary out."""
    news_block = "\n".join(
        f"- [{n['published_at']}] ({n['sentiment'].upper()}) {n['title']} "
        f"({n['source']}): {n['description']}"
        for n in tagged_news
    ) or "No recent news found."

    return f"""You are a market intelligence summarizer for a retail investor's
portfolio holding. You are given recent news headlines for a company, each
already tagged with a sentiment label. Summarize what's relevant or has
changed recently, in plain language a non-expert can follow. Do not give
buy/sell advice - just report what's happening and why it might matter.

TICKER: {ticker}
COMPANY: {company}

RECENT NEWS (with sentiment tags):
{news_block}

Respond with ONLY a valid JSON object, no markdown fences, no commentary
before or after, in exactly this shape:
{{
  "summary": "2-4 sentences in plain language on what's relevant/changed"
}}
"""


def call_ollama(prompt: str) -> str:
    resp = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["response"]


def parse_summary_output(raw: str) -> str:
    """Same strip-then-parse approach as investment_thesis_agent.py's
    parse_model_output() - fail loudly, not silently, if it's not JSON."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json\n", "", 1)

    try:
        return json.loads(cleaned)["summary"]
    except (json.JSONDecodeError, KeyError):
        print("[warn] Model did not return clean JSON. Raw output below:\n")
        print(raw)
        return raw.strip()


def run_for_holding(ticker: str) -> dict:
    """One holding, start to finish. Returns the structured per-holding
    output - this is the shape the Orchestrator will eventually consume."""
    print(f"  [1/5] Resolving company name for {ticker} ...")
    company = fetch_company_name(ticker)

    print(f"  [2/5] Fetching news for {company} ...")
    news = fetch_news(company)

    print(f"  [3/5] Filtering for relevance to {company} ...")
    relevant_news = filter_news_by_relevance(company, news, threshold=NEWS_RELEVANCE_THRESHOLD)
    # relevance_score was only needed for the filter step's debug output -
    # drop it here so it doesn't clutter the final per-holding JSON.
    relevant_news = [{k: v for k, v in a.items() if k != "relevance_score"} for a in relevant_news]

    print(f"  [4/5] Tagging sentiment on {len(relevant_news)} article(s) ...")
    tagged_news = tag_news_sentiment(relevant_news)
    overall_sentiment = aggregate_sentiment(tagged_news)

    print(f"  [5/5] Summarizing via Ollama ({OLLAMA_MODEL}) ...")
    prompt = build_prompt(ticker, company, tagged_news)
    try:
        raw_output = call_ollama(prompt)
        summary = parse_summary_output(raw_output)
    except requests.exceptions.ConnectionError as e:
        print(f"  [warn] could not reach Ollama: {e}")
        summary = "Summary unavailable - could not reach local Ollama server."

    return {
        "ticker": ticker,
        "news_items": tagged_news,
        "overall_sentiment": overall_sentiment,
        "summary": summary,
    }


def run(filepath: str, limit: int = None) -> list:
    holdings = load_portfolio(filepath)
    if limit:
        holdings = holdings[:limit]

    results = []
    for h in holdings:
        print(f"\n=== {h['ticker']} ===")
        results.append(run_for_holding(h["ticker"]))

    return results


def main():
    parser = argparse.ArgumentParser(description="Market Intelligence Agent")
    parser.add_argument("--file", default="data/sample_portfolio.csv",
                         help="Path to portfolio CSV (same format as Portfolio Health Agent)")
    parser.add_argument("--limit", type=int, default=3,
                         help="Only process the first N holdings (default 3, for a quick test run)")
    args = parser.parse_args()

    results = run(args.file, limit=args.limit)

    print("\n" + "=" * 60)
    print("MARKET INTELLIGENCE RESULTS")
    print("=" * 60)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()