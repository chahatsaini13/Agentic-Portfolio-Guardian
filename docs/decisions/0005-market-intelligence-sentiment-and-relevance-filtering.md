# ADR 0005: Market Intelligence Agent - sentiment placeholder and relevance filter reuse

## Status
Accepted - Member 1's portion (agent pipeline, filtering, VADER placeholder)
is complete and validated below. Member 2's portion (contrastive sentiment
model, accuracy comparison) was completed in ADR 0006 - see that file for
the swap-in confirmation and validation numbers.

## Context
Week 5 (Market Intelligence Agent) needs to tag sentiment on recent news
per holding, per ROADMAP.md's "Contrastive Sentiment Tagging" plan: Member
2 fine-tunes sentence embeddings contrastively (SimCSE-style, using
Financial PhraseBank as labeled anchor pairs) so similar-sentiment
financial text clusters together in embedding space.

That model isn't ready yet, and the agent, prompt-building, and JSON
output shape all needed to be built and tested tonight regardless -
same reasoning as ADR 0001 (build the standalone pipeline first, swap
the real model in later without touching anything else).

Separately, the agent needed a news-relevance step. Week 3's
`news_filter.py` / `relevance_scorer.py` already solve "score how
relevant a piece of text is to an anchor text" - nothing about that
function is thesis-specific, it just embeds two strings and compares
them. Market Intelligence has no thesis to anchor on, so the question
was whether to write a new filtering module or reuse the existing one
with a different anchor.

## Decision

**1. Sentiment tagging placeholder: VADER, not a downloaded transformer.**
`sentiment_tagger.py` uses `vaderSentiment` (rule-based, lexicon-driven,
no model download) behind the same swappable-contract pattern as
`relevance_scorer.py`:

    tag_sentiment(text: str) -> str   # "positive" | "negative" | "neutral"

Chosen over an off-the-shelf HF sentiment pipeline (e.g.
`distilbert-base-uncased-finetuned-sst-2-english`) purely for tonight's
time budget - VADER scores instantly with no download, and the contract
means swapping in Member 2's model later is a body-only change in
`_get_analyzer()` / `tag_sentiment()`, same as `relevance_scorer.py`'s
`_get_model()` swap path.

**2. Relevance filtering: reuse `filter_news_by_relevance()`, anchor on
company name instead of thesis.**
Rather than writing a second filtering module, Market Intelligence Agent
calls the same Week 3 function:

```python
filter_news_by_relevance(company, news, threshold=NEWS_RELEVANCE_THRESHOLD)
```

`NEWS_RELEVANCE_THRESHOLD = 0.3` - set lower than Week 3's `0.28`→ tuned
value is close by coincidence, not derived the same way. This is a
first-pass guess for a **weaker anchor**: a bare company name gives the
embedding model far less to latch onto than a full thesis sentence did
in Week 3, so score bands are expected to sit lower across the board.
Not yet validated against enough real runs to trust the exact number -
flagged as a retuning candidate below, same as Week 3's threshold was
before its own retuning pass (ADR 0002).

**Pipeline (Week 5):**
```
load_portfolio() -> fetch company name (yfinance) -> fetch news (NewsAPI,
company-name query only, no thesis) -> filter_news_by_relevance(company,
news) -> tag_sentiment() per kept article -> majority-vote overall
sentiment -> build prompt (news + sentiment tags) -> Ollama -> parse JSON
-> {ticker, news_items, overall_sentiment, summary}
```

## Validation

Tested end-to-end on 3 holdings from `data/sample_portfolio.csv`
(TCS.NS, INFY.NS, HDFCBANK.NS), `threshold=0.3`:

| Ticker | Fetched | Relevance scores | Kept | Overall sentiment |
|---|---|---|---|---|
| TCS.NS | 2 | 0.5069, 0.3887 | 2 | neutral |
| INFY.NS | 1 | 0.2154 | 0 | neutral (no news) |
| HDFCBANK.NS | 2 | 0.6335, 0.5966 | 2 | negative |

**Filter worked correctly once (INFY.NS):** a multi-company IT-sector
roundup article ("HCLTech, TCS, Infosys, other IT stocks rally...")
scored 0.2154 and was correctly dropped - it wasn't specifically about
Infosys.

**Filter false-positive observed (TCS.NS):** an article titled "Stocks
to watch: Wipro, Hero MotoCorp, HDFC Bank, Tata Motors PV & more" - whose
description is entirely about HCLTech and Wacker Chemie, not TCS at all
- scored 0.3887 and was kept. This is a real miss: the embedding model
is picking up on general IT-sector/stocks-roundup semantic similarity
rather than "is this article actually about TCS." **Do not quote 0.3887
as a validated relevance signal** - it demonstrates the filter's current
weak spot, not a case it handled well.

**Known limitation, not fixed by this filter (HDFCBANK.NS):** both kept
articles were about a US securities class-action against "HDFC Bank
Limited (NYSE: HDB)" - the NYSE-listed ADR, not the NSE-listed shares
this portfolio actually holds. Company name matches genuinely (same
underlying company), so relevance scoring correctly calls it relevant -
but it isn't distinguishing listing venue. This is a cross-listing
ambiguity in the fetch/filter design, not something a better relevance
threshold can fix; would need an explicit exchange/ticker check or a
prompt-level instruction to Ollama to flag venue-specific caveats.

## Consequences
- `NEWS_RELEVANCE_THRESHOLD = 0.3` is a first-pass estimate on 3 tickers'
  worth of real scores (5 articles total) - too small a sample to trust
  the way Week 3's 0.28 was validated (~13 articles, three visible score
  bands). Expect this to move once more runs are logged.
- The TCS false-positive suggests the threshold may need to go up
  (~0.4?) to catch sector-roundup noise, but that risks cutting genuinely
  relevant borderline articles too - same precision/recall tension noted
  in ADR 0002, not yet resolved here with enough data to pick a side.
- Cross-listing (NSE vs. ADR/NYSE) news bleed-through is a known,
  unresolved limitation - flagged for a future week rather than blocking
  Week 5 completion.
- `vaderSentiment` needs to be added to `requirements.txt` (not yet
  committed as of this ADR).
- Sentiment tags are from a generic English-language lexicon model, not
  validated against Financial PhraseBank yet - treat as a rough signal
  only until Member 2's model is integrated and the comparison below is
  run.

---
