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
  at or above a threshold.

This is implemented as two small modules (`relevance_scorer.py`,
`news_filter.py`) with a single function contract -
`score_relevance(thesis, news_text) -> float` - so that Member 2's
contrastively fine-tuned encoder (trained on `data/contrastive/`,
merged separately) can be swapped in later by changing one model
reference, with no changes needed to the filtering logic or its callers.

**Pipeline (Week 3):**
```
yfinance fundamentals -> NewsAPI -> embed thesis + each article ->
keep articles above similarity threshold -> build prompt -> Ollama ->
parse JSON -> log run to results/week3_log.json
```

## Validation

### Attempt 1: company-name-only query (Aug 2)
The first implementation queried NewsAPI on company name only, per the
"separate concerns" reasoning above, at `threshold=0.35`. 14 runs across
10 tickers were logged:

| Ticker | Fetched | Kept | Verdict |
|---|---|---|---|
| TMCV.NS | 1 | 1 | WEAKENING |
| RELIANCE.NS | 0 | 0 | WEAKENING |
| INFY.NS | 2 | 0 | BROKEN |
| HDFCBANK.NS | 0 → 1 (rerun) | 0 | BROKEN |
| TCS.NS | 0 | 0 | HOLDS |
| SUNPHARMA.NS | 0 | 0 | HOLDS |
| ADANIENT.NS | 0 | 0 | WEAKENING |
| ZOMATO.NS | 0 → 4 (rerun) | 0 | WEAKENING / BROKEN |
| PAYTM.NS | 0 | 0 | BROKEN |
| ITC.NS | 0 → 1 (rerun) | 0 | WEAKENING |

**13 of 14 runs fetched 0 or 1 article.** Even the one run with a decent
candidate pool (`ZOMATO.NS`, 4 fetched) had all 4 filtered out at
`threshold=0.35`. This wasn't the filter doing careful discrimination -
there simply wasn't enough of a candidate pool for the filter to work
with in the first place. Company-name-only queries against NewsAPI's
free tier returned too few results, most likely because
`fundamentals["long_name"]` (e.g. "HDFC Bank Limited") doesn't match how
NewsAPI indexes company mentions, combined with free-tier recency/volume
restrictions.

**Consequence observed:** with 0 kept articles, the LLM had nothing to
reason from besides the thesis text and fundamentals - and on at least
one occasion (an invalid ticker symbol, separately), it was observed to
generate plausible-sounding but fabricated supporting figures rather
than surfacing a "no data" signal. This is a known limitation, flagged
below.

### Attempt 2: restore thesis keywords in the query (Aug 4)
Re-inspecting `fetch_news()`, the `keywords = extract_keywords(thesis)`
variable was computed but never actually used in the query string - the
request only sent `"q": f'{company}'`. Restoring
`"q": f'{company} {keywords}'` (the Week 2 behavior) and raising
`page_size` from 5 to 15 immediately changed the fetch picture:

| Ticker | Thesis (short) | Fetched | Kept @ 0.35 |
|---|---|---|---|
| TMCV.NS | EV adoption | 1 | 1 |
| RELIANCE.NS | Jio expansion | 1 | 1 |
| INFY.NS | AI services | 6 | 3 |
| HDFCBANK.NS | Retail credit growth | 5 | 0 |
| SUNPHARMA.NS | US generic approvals | 0 | 0 |
| ZOMATO.NS | Quick commerce | 1 | 0 |

This is the opposite of what the original "separate concerns" reasoning
predicted: including thesis keywords in the query, rather than
over-constraining results, gave the fetch stage more to work with -
likely because NewsAPI's `q` matching is closer to "these terms appear
somewhere across the article" than a strict AND of every word, and the
thesis keywords helped surface more specific, on-topic articles rather
than generic company coverage. **The original decision to separate
fetching and filtering by dropping thesis context from the query was
revised**: thesis keywords stay in the query (recall), and the embedding
filter still runs afterward (precision) - the two stages are
complementary, not query-vs-filter alternatives.

### Threshold retuning (Aug 4)
With `INFY.NS`, `HDFCBANK.NS`, and `ZOMATO.NS` now returning multiple
candidate articles, per-article debug scores were logged for the first
time, revealing three bands:

```
0.42 - 0.55  clearly on-topic (e.g. RELIANCE Jio-earnings article,
             INFY AI-revenue articles)
0.28 - 0.33  genuinely relevant but not headline-obvious (e.g. HDFC Bank
             Q1 earnings call; Zomato CEO responding to Rapido/Swiggy
             competition - directly bears on the thesis, but doesn't
             repeat thesis wording)
0.19 - 0.26  generic noise (market-wide Sensex roundups, multi-company
             "Q1 Results Live" listicles, unrelated HR-policy news)
```

`threshold=0.35` sat inside the "genuinely relevant" band, discarding
articles that should have been kept (both the HDFC Bank earnings-call
article and the Zomato competitor article scored below 0.35 and were
dropped). **Threshold revised from 0.35 to 0.28**, which sits in the gap
between band 2 and band 3.

### Before/after comparison at the revised threshold
Same ticker, same thesis, same fetch - only the threshold changed
(`0.0` = unfiltered baseline, all fetched articles passed through;
`0.28` = revised filter).

**HDFCBANK.NS** - "I bought this because retail credit growth will drive margins"

| | Unfiltered (0 kept threshold, 5 articles) | Filtered (0.28, 1 kept) |
|---|---|---|
| Articles in prompt | Earnings call + ICICI comparison + Sensex market roundup + 2x multi-company "Q1 Results Live" listicles | Earnings call only |
| Verdict | **BROKEN** | **WEAKENING** |
| Key changes cited | "Decline in HDFC Bank's stock" (traceable to the Sensex-wide roundup, not an HDFC-specific signal) | "No explicit guidance on future revenue growth" (traceable to the earnings call) |

Feeding in market-wide noise alongside the genuinely relevant article
pushed the verdict to a harsher, less defensible conclusion (BROKEN)
than the filtered version (WEAKENING), and the unfiltered version's
supporting claim traces back to an article that isn't actually about
HDFC Bank specifically. This is concrete evidence that filtering changes
not just article count but the resulting verdict's accuracy.

**ZOMATO.NS** - only 1 article was fetched in this run regardless of
threshold, so filtering had no differentiating effect here. Noted as an
honest limitation: filtering's benefit is only visible when the fetch
stage returns a non-trivial candidate pool: on a low-recall run there's
nothing to filter.

## Consequences
- Fetching (broad recall, now keyword-inclusive) and relevance-scoring
  (precision) are complementary pipeline stages, not a strict either/or
  split - revising the original "company-name-only" fetch decision was
  necessary once real NewsAPI recall was measured.
- Threshold is `0.28`, set from observed score bands across ~13 articles
  with real fetched content, not from a single eyeballed guess. Still
  expected to move again once Member 2's contrastively trained model
  (trained specifically to separate relevant/irrelevant pairs) replaces
  the generic MiniLM placeholder - a model trained for this task should
  give better score separation, likely raising the useful threshold and
  making the cutoff less sensitive to exact placement.
- **Known limitation**: NewsAPI's free tier still returns 0 articles for
  a meaningful fraction of tickers (e.g. `SUNPHARMA.NS` in this test
  run) even after the query fix - not resolved, flagged for follow-up.
  Confirmed as an external rate-limiting/coverage issue, not a code bug
  (repeated calls for the same ticker return different counts).
- **Known limitation**: when both fundamentals and news fetches fail,
  the LLM was observed to fabricate plausible-sounding but fictitious
  supporting figures rather than surfacing an error - the pipeline has
  no check yet for "insufficient data to reason from." Flagged as a
  candidate for a self-reflection / output-validation enhancement
  (Week 7 or later).
- Ollama's default generation length occasionally truncated the JSON
  response once prompts grew longer (more kept articles -> longer
  reasoning) - fixed by setting `num_predict: 500` explicitly in the
  `/api/generate` call.
- Small-sample overfitting risk in Member 2's eventual trained model
  (6 train / 2 val theses) remains an open follow-up, previously flagged
  during branch review.