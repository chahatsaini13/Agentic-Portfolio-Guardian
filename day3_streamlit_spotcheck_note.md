# Day 3 Spot-Check Note — Streamlit App Validation

**Date:** 2026-09-06/07
**Validated by:** Navneet
**Environment:** Local Windows machine, `app.py` (Chahat's Streamlit dashboard), full 4-agent orchestrator run

## Setup notes (for reproducibility)

Two local environment issues had to be fixed before the app would run — neither is a code bug, both are one-time local setup gaps:

1. `.env` had no `OLLAMA_MODEL` set, so agents defaulted to `llama3.1`, which wasn't pulled on this machine (only `llama3:latest` was). This produced a 404 from Ollama's `/api/generate`. Fixed by setting `OLLAMA_MODEL=llama3` in `.env`.
2. Ollama's first response on a cold model took longer than the hardcoded `timeout=120` in each agent's `call_ollama()`, causing a `ReadTimeout`. Bumped to `timeout=300` in all four agent files (`investment_thesis_agent.py`, `portfolio_health_agent.py`, `market_intelligence_agent.py`, `early_warning_agent.py`).

`NEWSAPI_KEY` in `.env` was left as the placeholder value (`your_actual_key_here`) — every NewsAPI call returned `401 apiKeyInvalid`. This was **not fixed**, and turned out to be a useful accidental test case (see below).

## What was run

Full portfolio (`data/sample_portfolio.csv`, 8 holdings: TCS.NS, INFY.NS, HDFCBANK.NS, ICICIBANK.NS, TMPV.NS, MARUTI.NS, HINDUNILVR.NS, ITC.NS), via `app.py` → `orchestrator.run()`, with "Fetch live prices" and "Fetch market metadata" unchecked (matches ADR 0010's Day 2 validation flags).

## Results

- **Portfolio Health**: Health Score 85.0, Diversification Score 89.16, Largest Holding HDFCBANK.NS (20%), Largest Sector Banking (35%) — populated correctly from real `sample_portfolio.csv` weights.
- **Thesis Status breakdown**: Holds 7, Weakening 1 (no errors this run).
- **Market Sentiment breakdown**: Neutral 8 (all 8 holdings) — expected, since NewsAPI returned 0 articles for every ticker (401 invalid key), and `aggregate_sentiment()`'s documented no-news-defaults-to-neutral behavior kicked in correctly for all of them.
- **Exposure Alerts**: fired correctly (single_stock, sector-level) based on real portfolio weights.
- **Red-flag alerts**: 0 across all holdings — consistent with 0 articles fetched (nothing to flag).

### Per-holding spot-check: TCS.NS
- `thesis_status`: **HOLDS**
- `reasoning`: cites real fundamentals (revenue growth 13.9%, PE ratio 16.75) fetched from yfinance, explicitly states "No recent news has changed the fundamentals, maintaining the thesis's logic."
- `market_sentiment`: neutral, summary correctly states no recent news was available.
- `redflag_alerts`: none.

### Per-holding spot-check: HDFCBANK.NS
- `thesis_status`: **HOLDS**
- `reasoning`: cites real revenue growth (0.166) tied to the retail-credit-growth thesis.
- `market_sentiment`: neutral, "no updates to report."
- `redflag_alerts`: none.

## Key finding — relevant to the open hallucination-risk item

`OPEN_ITEMS.md` and ADR 0002/0003 flag a known limitation: *"when both fundamentals and news fetches fail, the LLM was observed to fabricate plausible-sounding but fictitious supporting figures."*

This run is a **different, narrower scenario** than that documented failure mode: fundamentals were available (real yfinance data), only news was missing (NewsAPI 401). In this specific case, the model did **not** hallucinate — it grounded its reasoning in the real fundamentals it did have and explicitly acknowledged the absence of news rather than inventing any. This is a positive data point, but it does not resolve the open item, since the open item's failure mode (fundamentals AND news both missing) was not exercised here. Recommend keeping the open item open, but adding a note that the "fundamentals-only, no-news" sub-case appears well-behaved based on this spot-check.

## LOTO hard-negative inconsistency (ADR 0011) — checkability against current data

The three theses that showed weaker hard-negative accuracy in ADR 0011's LOTO breakdown (`t6_adanigreen_renewables`, `t7_itc_fmcg`, `t8_dlf_realestate`) map to sectors that are **not present** in `data/sample_portfolio.csv` in a directly comparable way:
- Adani Green / renewables: not held.
- DLF / real estate: not held.
- ITC / FMCG: **is** held (`ITC.NS`), but this run's `thesis_status` for ITC.NS could not be meaningfully cross-checked against the relevance-model weakness, because the relevance filter never had any candidate articles to score in the first place (NewsAPI 401 → 0 fetched). The LOTO finding is about the model's ability to separate *hard-negative news it does see*; with zero articles fetched, there was nothing for the relevance model to succeed or fail on in this run.

**Conclusion**: as anticipated in the task brief, this is not checkable today with the current portfolio + working NewsAPI key. Flagged as a "note for later" — a real check would need (a) a valid NEWSAPI_KEY, and (b) ideally a holding in one of the three weaker-thesis sectors, to see whether a same-company/thesis-irrelevant article slips through the filter in production the way ADR 0011 predicts it might.

## Follow-ups to log in OPEN_ITEMS.md

- [ ] Local dev setup gap: `.env.example` should document that `OLLAMA_MODEL` must match a locally-pulled model name, and that Ollama's `timeout` may need to exceed 120s on a cold/CPU-only model. Consider bumping the default timeout constants in the repo itself rather than relying on local edits.
- [ ] Confirmed (not new, but re-validated): fundamentals-only-no-news case does not appear to trigger the hallucination risk documented in ADR 0002/0003 — only the fully-empty-fundamentals-and-news case remains an open concern.
- [ ] LOTO hard-negative weak-sector spot-check (Adani Green / ITC / DLF) remains unchecked against live data — needs a working NEWSAPI_KEY and ideally portfolio holdings in those sectors.
