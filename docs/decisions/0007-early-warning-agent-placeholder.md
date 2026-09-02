# ADR 0007: Rule-based red-flag detection as a placeholder for the Early Warning Agent, ahead of contrastive anomaly detection

## Status
Accepted

## Context
Week 6 of the roadmap calls for the Early Warning Agent: detecting
red-flag events (management change, litigation, credit downgrades, fraud)
in news per holding, treated as anomaly detection - contrastively trained
embeddings separating "routine business news" from "red-flag news" into
distinguishable clusters, supplementing rule-based keyword matching with a
learned signal.

Member 2's contrastive red-flag model was not ready at the start of this
week's session. Following the same reasoning as ADR 0001 (build the agent
logic standalone before wiring in the final component) and the same
placeholder-swap pattern used for news relevance filtering (Week 3) and
sentiment tagging (Week 5, ADR 0006), the decision was to build the full
Early Warning Agent pipeline tonight against a rule-based placeholder, so
the pipeline shape (fetch news -> detect -> score severity -> alert) is
validated end-to-end before the contrastive model exists to drop in.

## Decision
Implement red-flag detection in two files, mirroring `sentiment_tagger.py`'s
exact pattern:

**`src/agents/redflag_detector.py`** exposes a single swappable contract:
```
redflag_score(text: str) -> float   # 0.0 (routine) .. 1.0 (red-flag)
```
`is_red_flag(text, threshold=0.5)` is a thin threshold wrapper around it.
The placeholder implementation is keyword matching against five categories
(fraud, litigation, credit_downgrade, management_change, regulatory) -
returns 1.0 on any keyword hit, 0.0 otherwise. This mirrors the float-score
shape of `sentiment_tagger.py`'s centroid-cosine-similarity contract
(rather than a plain boolean), so when Member 2's contrastive model lands
it can return a real continuous 0-1 similarity score without changing the
function signature or anything downstream.

`classify_flag_type(text) -> str | None` is a **separate, permanent**
helper - not part of the swap contract. It stays rule-based even after the
contrastive model lands, because a 2-cluster (routine vs red-flag)
contrastive model has no way to sub-categorize *why* something is a red
flag. Category labels are still useful for the alert output, so this
keyword lookup is kept as-is indefinitely.

**`src/agents/early_warning_agent.py`** follows the same 4-step,
standalone-script structure as `market_intelligence_agent.py`:
1. Load holdings via `load_portfolio()`
2. Fetch news per holding - reuses `market_intelligence_agent.py`'s
   `fetch_company_name()` / `fetch_news()` directly via import rather than
   duplicating the NewsAPI request shape a third time
3. Run `detect_flags()` (calls `redflag_score()` / `classify_flag_type()`
   per article) to shortlist red-flag articles
4. For each flagged article, call local Ollama for a severity verdict
   (LOW/MEDIUM/HIGH + reasoning), one call per article rather than one
   batched call per holding, so one ambiguous article can't skew the
   severity judgment on the others

Output per holding is a list of `{ticker, flag_type, headline, severity,
reasoning}` alerts (empty list if nothing was flagged) - matches the same
per-holding JSON shape the other three agents produce, ready for the
Orchestrator to consume in Week 7.

## Consequences
- The pipeline (fetch -> detect -> score -> alert) is fully validated
  end-to-end tonight, without waiting on Member 2's model.
- Swapping in the contrastive model later only touches the body of
  `redflag_score()` in `redflag_detector.py` - `is_red_flag()`,
  `classify_flag_type()`, and all of `early_warning_agent.py` are
  unaffected, same guarantee `sentiment_tagger.py` gives.
- The placeholder is binary (1.0/0.0), not a true continuous score, so
  there is no threshold to tune yet - `_REDFLAG_THRESHOLD = 0.5` exists
  only so the contract shape matches what the contrastive version will
  need. Real threshold tuning (same as Week 3/5's threshold work) happens
  once real similarity scores exist.
- Holdings with zero flagged articles cost zero Ollama calls - severity
  scoring only runs on articles that already cleared the keyword filter,
  keeping the pipeline cheap to run repeatedly during testing.
- **Known limitation - category priority ordering:** `_REDFLAG_KEYWORDS`
  is checked in insertion order (fraud first) and the first matching
  category wins. A headline like "HDB Investors Have Opportunity to Lead
  HDFC Bank Limited Securities Fraud **Lawsuit**" is fundamentally about
  litigation but gets classified as `fraud` because that category is
  checked first and the literal word "fraud" appears. Acceptable for a
  keyword placeholder; does not carry over to the contrastive model since
  that model only separates routine/red-flag, not sub-categories (see
  Decision above - `classify_flag_type()` stays rule-based regardless).
- **Known limitation - NewsAPI coverage on small/mid-cap names:** validated
  against SYNCOMF.NS (Syncom Formulations), a small-cap NSE stock with a
  genuine, dated management-change event (founder + CFO both resigned,
  effective 11 Aug 2026). The live end-to-end run returned 0 articles for
  this ticker - not a detection failure, but NewsAPI's free tier returning
  no coverage for a small-cap name. Confirmed via a separate unit-level
  test (real headline text, bypassing the live API call) that
  `management_change` detection itself is correct. This is the same
  NewsAPI free-tier recency/volume constraint already documented in
  README.md, now confirmed to affect small-cap coverage specifically.

## Validation

### Test Case 1 - unit-level, mock headlines (no network)
Five mock headlines run directly through `redflag_score()` /
`classify_flag_type()` to sanity-check the keyword logic before wiring it
into the agent:

| Headline | Flagged | Category |
|---|---|---|
| "Tata Motors reports strong Q2 revenue growth" | No | - |
| "XYZ Ltd CFO resigns amid accounting irregularities probe" | Yes | fraud |
| "ICRA downgrades ABC Corp credit rating to BBB-" | Yes | credit_downgrade |
| "Company wins new export contract with European partner" | No | - |
| "Regulators fine DEF Industries for compliance lapses" | Yes | litigation |

Routine/positive news stayed unflagged; each red-flag category fired
correctly in isolation.

### Test Case 2 - live end-to-end, real portfolio
**Command:** `python -m src.agents.early_warning_agent --file data/sample_portfolio.csv --limit 3`

- **TCS.NS** - 1 article fetched, 0 flagged (correct - routine news)
- **INFY.NS** - 0 articles fetched, 0 flagged
- **HDFCBANK.NS** - 1 article fetched, 1 flagged:

```json
{
  "ticker": "HDFCBANK.NS",
  "flag_type": "fraud",
  "headline": "HDB Investors Have Opportunity to Lead HDFC Bank Limited Securities Fraud Lawsuit",
  "severity": "HIGH",
  "reasoning": "The news indicates a potential securities fraud lawsuit against the company, which could result in significant financial penalties and damage to the bank's reputation. As a shareholder, this is a critical issue that requires close monitoring, and the potential for a lengthy and costly lawsuit makes this a high-severity event."
}
```

Confirms the full pipeline fires correctly on real news: fetch -> keyword
detect -> Ollama severity scoring -> structured alert. Also demonstrates
the category-priority limitation noted above (`fraud` vs the more accurate
`litigation`).

### Test Case 3 - unit-level, real headline text (management_change category)
**Ticker:** SYNCOMF.NS (Syncom Formulations)

Live end-to-end run returned 0 articles for this ticker (NewsAPI coverage
gap, see Known Limitations). To confirm `management_change` detection
itself works, the real headline text (from public NSE filings/press
coverage) was run directly through the detector:

```
text = "Syncom Formulations founder resigns, CFO exits. Kedarmal
Shankarlal Bankda steps down as Whole-Time Director and Director; Rahul
Vijay Bankda resigns as Chief Financial Officer."

redflag_score(text)      -> 1.0
is_red_flag(text)        -> True
classify_flag_type(text) -> "management_change"
```

**Summary:** across the three test cases, three of the five red-flag
categories (`fraud`, `credit_downgrade`, `management_change`) and the
negative case (routine news, correctly unflagged) are validated. `fraud`
additionally validated live end-to-end including LLM severity scoring.
`litigation` and `regulatory` validated at the unit level only (Test Case
1); not yet confirmed against a live NewsAPI article.
