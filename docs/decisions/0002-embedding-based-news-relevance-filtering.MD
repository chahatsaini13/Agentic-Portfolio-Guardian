# ADR 0002: Replace keyword-only news filtering with embedding-similarity filtering

## Status
Accepted

## Context
The Week 2 prototype relied on keyword overlap (via `extract_keywords()`)
baked directly into the NewsAPI query string, plus a simple substring
check against the company name. This has two problems:

1. It conflates two different concerns - "find news about this company"
   and "find news relevant to this specific thesis" - inside a single
   NewsAPI query string. Packing thesis keywords directly into the `q`
   parameter over-constrains the search: NewsAPI's `q` matching on a long
   space-separated string effectively requires most/all terms to
   co-occur, which is a much stricter bar than "relevant to the thesis."
2. Keyword overlap doesn't capture semantic relevance. An article can be
   highly relevant to a thesis without sharing any exact keywords with it
   (e.g. "resale value concerns" is relevant to an "EV adoption will
   drive demand" thesis without repeating any of its words), and can
   share keywords without being relevant.

## Decision
Split the two concerns into separate pipeline stages:

- **Fetching**: query NewsAPI on the company name only, so it returns a
  broader set of candidate articles about the company (not narrowed by
  thesis wording).
- **Filtering**: score every fetched article's relevance to the specific
  thesis using sentence embeddings (`all-MiniLM-L6-v2`, generic
  pretrained model for now) and cosine similarity, keeping only articles
  at or above a threshold (default `0.35`, tuned by eyeballing real
  output rather than derived analytically).

This is implemented as two small modules (`relevance_scorer.py`,
`news_filter.py`) with a single function contract -
`score_relevance(thesis, news_text) -> float` - so that Member 2's
contrastively fine-tuned encoder (trained on `data/contrastive/`,
merged separately) can be swapped in later by changing one model
reference, with no changes needed to the filtering logic or its callers.

**Pipeline (Week 3):**
```
yfinance fundamentals -> NewsAPI (company name only) -> embed thesis + each
article -> keep articles above similarity threshold -> build prompt ->
Ollama -> parse JSON -> log run to results/week3_log.json
```

## Validation
10 runs were logged tonight across 8 tickers/theses
(`results/week3_log.json`), split across two query strategies:

**Before the fix** (NewsAPI query included thesis keywords):
Most tickers returned 0 articles - the over-constrained query rarely
found matches. Only 2 of 6 runs returned any candidates at all
(`TMCV.NS`: 1 fetched -> 1 kept; `INFY.NS`: 2 fetched -> 0 kept).

**After the fix** (NewsAPI query on company name only):
| Ticker | Thesis (short) | Fetched | Kept | Verdict |
|---|---|---|---|---|
| ITC.NS | FMCG demand recovery | 1 | 0 | WEAKENING |
| ZOMATO.NS | Food delivery market share | 4 | 0 | WEAKENING |
| HDFCBANK.NS | Retail credit growth | 1 | 0 | BROKEN |
| RELIANCE.NS | Jio expansion | 0 | 0 | WEAKENING |

The Zomato case is the clearest evidence the filter is doing real work:
4 genuine candidate articles were fetched, and the embedding filter
judged all 4 insufficiently relevant to the specific thesis (rather than
simply passing through everything NewsAPI returned, as the Week 2
keyword-substring check effectively did).

## Consequences
- Fetching and relevance-scoring are now decoupled, which is a cleaner
  separation of concerns and makes each stage independently testable.
- The 0.35 threshold is a starting point from a small number of real
  runs, not a principled number - expect to retune once more data is
  available, and again once Member 2's contrastively trained model
  replaces the MiniLM placeholder (a model trained specifically to
  separate relevant/irrelevant pairs should give better score
  separation, likely shifting the useful threshold range).
- **Known limitation**: NewsAPI's free tier still returns 0 articles for
  a meaningful fraction of tickers even after the query fix (e.g.
  `RELIANCE.NS` in this test run) - likely a combination of recency
  restrictions and company-name matching on NewsAPI's side
  (`fundamentals["long_name"]` may not match how NewsAPI indexes a
  company). Not resolved tonight; flagged for follow-up.
- **Known limitation**: when both fundamentals and news fetches fail
  (observed once, on an invalid ticker symbol), the LLM was observed to
  fabricate plausible-sounding but fictitious supporting figures rather
  than surfacing an error - the current pipeline has no check for
  "insufficient data to reason from." Flagged as a candidate for the
  self-reflection / output-validation enhancement noted under Week 7.
- Small-sample overfitting risk in Member 2's eventual trained model
  (6 train / 2 val theses) remains an open follow-up, previously flagged
  during branch review.
