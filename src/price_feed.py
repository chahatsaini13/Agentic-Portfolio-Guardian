"""
Price Feed - Portfolio Guardian

Lightweight, LLM-free current-price lookup for the dashboard's live price
strip. Uses history() instead of fast_info - more reliable across
yfinance versions and NSE tickers, especially outside market hours.
"""

import yfinance as yf


def fetch_live_prices(tickers: list) -> dict:
    """Returns {ticker: {"price": float|None, "change_pct": float|None}}."""
    prices = {}
    for ticker in tickers:
        try:
            hist = yf.Ticker(ticker).history(period="2d")
            if hist.empty or "Close" not in hist.columns:
                print(f"[price_feed] no history data for {ticker}")
                prices[ticker] = {"price": None, "change_pct": None}
                continue

            closes = hist["Close"].dropna()
            if len(closes) == 0:
                prices[ticker] = {"price": None, "change_pct": None}
                continue

            last = float(closes.iloc[-1])
            prev = float(closes.iloc[-2]) if len(closes) >= 2 else None
            change_pct = ((last - prev) / prev * 100) if prev else None
            prices[ticker] = {"price": last, "change_pct": change_pct}
        except Exception as e:
            print(f"[price_feed] could not fetch live price for {ticker}: {e}")
            prices[ticker] = {"price": None, "change_pct": None}
    return prices