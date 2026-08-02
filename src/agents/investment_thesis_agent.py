"""
Investment Thesis Agent - Portfolio Guardian (prototype)

Given a stock ticker + a one-line investment thesis, this pulls current
price/fundamentals (yfinance) and recent news (NewsAPI), hands all of it
to a local Ollama model, and asks it to judge whether the thesis still
holds. No orchestration framework here on purpose - this is the single
"tool" that will later get wired into a LangGraph agent.

Usage:
    python investment_thesis_agent.py --ticker TATAMOTORS.NS \
        --thesis "I bought this because EV adoption will drive demand"
"""

import argparse
import json
import os
import sys
import re
import datetime

import requests
import yfinance as yf
from dotenv import load_dotenv
from src.agents.news_filter import filter_news_by_relevance, DEFAULT_RELEVANCE_THRESHOLD

load_dotenv()  # reads .env in the current directory, sets values into os.environ

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1")  # change to whatever you've pulled
NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY")  # set this in your .env file, see .env.example


def fetch_fundamentals(ticker: str) -> dict:
    """Pull current price + a handful of fundamentals from yfinance."""
    stock = yf.Ticker(ticker)
    info = stock.info  # yfinance quirk: first access is slow, it's fetching + caching
    if not info:
        raise ValueError(f"Invalid Yahoo Finance ticker: {ticker}")

    if not info or info.get("regularMarketPrice") is None:
        print(f"[warn] yfinance returned little/no data for '{ticker}'. "
              f"Check if the Yahoo Finance ticker is correct.")

    return {
        "ticker": ticker,
        "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
        "pe_ratio": info.get("trailingPE"),
        "market_cap": info.get("marketCap"),
        "52_week_high": info.get("fiftyTwoWeekHigh"),
        "52_week_low": info.get("fiftyTwoWeekLow"),
        "revenue_growth": info.get("revenueGrowth"),
        "profit_margins": info.get("profitMargins"),
        "sector": info.get("sector"),
        "long_name": info.get("longName") or info.get("shortName"),
    }

def extract_keywords(text: str) -> str:
    stop_words = {
        "i", "bought", "this", "because", "will", "drive",
        "the", "a", "an", "and", "to", "for", "of"
    }

    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())

    keywords = [
        word for word in words
        if word not in stop_words
    ]

    return " ".join(keywords)

def fetch_news(company: str, thesis: str, page_size: int = 5) -> list:
    """Pull recent headlines relevant to the company and investment thesis.
    Returns a list of {title, description, source, published_at} dicts."""
    if not NEWSAPI_KEY:
        print("[warn] NEWSAPI_KEY not set - skipping news, agent will reason on fundamentals only.")
        return []
    
    keywords = extract_keywords(thesis)
    resp = requests.get(
        "https://newsapi.org/v2/everything",
        params={
            "q": f'{company}',
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


def build_prompt(ticker: str, thesis: str, fundamentals: dict, news: list) -> str:
    """One prompt, everything the model needs to reason, nothing it has to guess."""
    news_block = "\n".join(
        f"- [{n['published_at']}] {n['title']} ({n['source']}): {n['description']}"
        for n in news
    ) or "No recent news found."

    return f"""You are an investment thesis reviewer. A retail investor wrote down WHY
they bought a stock. Your job is to check if that reasoning still holds given
current data and recent news - not to give generic stock advice.

TICKER: {ticker}
ORIGINAL THESIS: "{thesis}"

CURRENT FUNDAMENTALS:
{json.dumps(fundamentals, indent=2)}

RECENT NEWS:
{news_block}

Evaluate whether the ORIGINAL THESIS still holds. Consider only things that
actually bear on the thesis's logic - don't just summarize the stock.

Respond with ONLY a valid JSON object, no markdown fences, no commentary
before or after, in exactly this shape:
{{
  "ticker": "{ticker}",
  "thesis_status": "HOLDS" | "WEAKENING" | "BROKEN",
  "reasoning": "2-4 sentences explaining the verdict, referencing specific data/news points",
  "key_changes": ["short bullet of a relevant change", "another one if applicable"]
}}
"""


def call_ollama(prompt: str) -> str:
    """Send the prompt to a locally running Ollama model, non-streamed."""
    resp = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["response"]


def parse_model_output(raw: str) -> dict:
    """Models occasionally wrap JSON in ```json fences or add a stray sentence.
    Strip the obvious junk before parsing, and fail loudly (not silently) if
    it's still not valid JSON - better to see the raw text than a crash."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json\n", "", 1)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        print("[warn] Model did not return clean JSON. Raw output below:\n")
        print(raw)
        return {
            "ticker": None,
            "thesis_status": "PARSE_ERROR",
            "reasoning": raw,
            "key_changes": [],
        }


def log_result(ticker, thesis, threshold, fetched_count, kept_count, result, log_path="results/week3_log.json"):
    """Append this run's result to a JSON log file, for batch testing tonight."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    entry = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "ticker": ticker,
        "thesis": thesis,
        "threshold": threshold,
        "articles_fetched": fetched_count,
        "articles_kept": kept_count,
        "thesis_status": result.get("thesis_status"),
    }
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            log = json.load(f)
    else:
        log = []
    log.append(entry)
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)


def run(ticker: str, thesis: str, threshold: float = DEFAULT_RELEVANCE_THRESHOLD) -> dict:
    print(f"[1/4] Fetching fundamentals for {ticker} ...")
    fundamentals = fetch_fundamentals(ticker)

    print(f"[2/4] Fetching news ...")
    company_query = fundamentals.get("long_name") or ticker.split(".")[0]
    news = fetch_news(company_query, thesis)
    fetched_count = len(news)
    print(f"  fetched {fetched_count} article(s) before filtering")

    print(f"[2.5/4] Filtering news by relevance ...")
    news = filter_news_by_relevance(thesis, news, threshold=threshold)
    kept_count = len(news)
    print(f"  kept {kept_count} of the fetched articles above threshold {threshold}")

    print(f"[3/4] Building prompt and calling Ollama ({OLLAMA_MODEL}) ...")
    prompt = build_prompt(ticker, thesis, fundamentals, news)
    raw_output = call_ollama(prompt)

    print(f"[4/4] Parsing model response ...")
    result = parse_model_output(raw_output)
    log_result(ticker, thesis, threshold, fetched_count, kept_count, result)
    return result


def main():
    parser = argparse.ArgumentParser(description="Investment Thesis Agent")
    parser.add_argument("--ticker", required=True, help="e.g. TATAMOTORS.NS or RELIANCE.BO")
    parser.add_argument("--thesis", required=True, help="Your one-line investment thesis")
    parser.add_argument("--threshold", type=float, default=DEFAULT_RELEVANCE_THRESHOLD,
                     help="Minimum relevance score to keep an article (default: 0.35)")
    args = parser.parse_args()

    result = run(args.ticker, args.thesis, args.threshold)

    print("\n" + "=" * 60)
    print("THESIS CHECK RESULT")
    print("=" * 60)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.ConnectionError as e:
        print(f"\n[error] Could not connect: {e}")
        print("If this is Ollama, run `ollama serve` in another terminal and check `ollama list`.")
        sys.exit(1)