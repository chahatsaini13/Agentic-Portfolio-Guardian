# ADR 0012: Fundamentals-Only Backtest of Thesis-Flip Detection on Historical NSE Data

## Status
Accepted

## Context
ROADMAP.md's Week 9 plan calls for backtesting thesis-flip detection
against historical NSE data (1990–2020, `stoicstatic/india-stock-data-
nse-1990-2020` on Kaggle), owned by Member 1. The live Investment Thesis
Agent reasons over fundamentals (yfinance) **and** recent news (NewsAPI).
No historical NewsAPI archive exists, so a true replay of "what news
would the LLM have seen at time T" is not possible - the same constraint
that scoped the Part 2 extension to fundamentals-only. This backtest is
scoped the same way: **price/fundamentals features only, no historical
news**, and is explicitly a small pilot (5-10 stocks) to demonstrate the
idea works in principle, not a full 1,700-stock validation.

## Decision

### Approach
For each of several (ticker, cutoff date T) pairs:
1. Compute a 252-trading-day trailing window of price-derived fundamentals
   up to T, using the **same functions** `portfolio_health_agent.py` uses
   live: `compute_volatility()`, `compute_max_drawdown()`,
   `compute_var()`, `compute_cvar()`, `compute_downside_deviation()`.
2. Build a simplified version of `investment_thesis_agent.py`'s
   `build_prompt()` - same HOLDS/WEAKENING/BROKEN verdict shape, news
   block replaced with an explicit "not available" note, fed the computed
   fundamentals instead of live yfinance/NewsAPI data.
3. Ask Ollama (`llama3.2:latest` - see Known Limitations on model choice)
   for a verdict based on fundamentals trend alone.
4. Compare against what **actually happened** in the 126 trading days
   after T (cumulative return ≥ +15% = GROWTH, ≤ -15% = DECLINE, else
   FLAT), computed directly from the same historical price series.

### Thesis framing (a real limitation, not hidden)
No record of an actual historical investor's thesis exists for any of
these holdings. Each case uses a generic synthetic framing instead:
*"I expect [ticker]'s recent fundamental strength (price momentum,
volatility, and drawdown profile) to continue supporting the stock going
forward."* This tests whether the LLM can read fundamentals trend
correctly and reason about continuation/reversal - not whether it can
evaluate a specific, real investment thesis, which is what the live
agent actually does.

### Direction-scoring rule (chosen before seeing results, not after)
`HOLDS` is scored correct only if the actual outcome was GROWTH; `BROKEN`
only if DECLINE. `WEAKENING` is treated as genuinely ambiguous and is
**excluded** from the correct/incorrect ratio rather than force-mapped to
either side - an arbitrary WEAKENING→DECLINE mapping would inflate the
apparent accuracy without justification. Parse errors are excluded the
same way.

### Data-quality issue found and fixed mid-pilot: unadjusted stock splits
The Kaggle dataset is **not split/bonus-adjusted**. This was discovered
when an initial HDFCBANK candidate showed a "-83% decline" that turned
out to be a literal 5:1 split on 2011-07-14 (Close 2519.70 → next-day
Open 505.10, no corresponding `Prev Close` gap) - not a real price move.
Several other candidates initially suggested by an unguarded date-scan
(WIPRO 2000, INFY 2000/2004, HDFCBANK 2011) were later confirmed to be
split-contaminated the same way once checked.

**Fix:** added `contains_likely_split()` - flags any single-day return
whose magnitude exceeds 35% as a likely split/bonus artifact - applied to
both the lookback window (feature computation) and the forward window
(outcome classification), and built into the candidate-suggestion scan
itself so contaminated dates are excluded before being offered as
candidates. This is the same "catch it against real data, not just
synthetic" pattern ADR 0003/0004 used for their own bugs - the guard
wasn't anticipated going in, it was needed once real data exposed it.

## Validation

### Cases run
8 (ticker, cutoff) pairs were selected via the split-aware candidate
scanner, covering multiple market periods (dot-com era, 2008 GFC, 2009
recovery, 2019 idiosyncratic, 2020 COVID) and both directions. 6 of 8
completed; 2 were excluded for documented data-quality reasons (see
Known Limitations).

| Ticker | Cutoff | Trailing return | Volatility | Max DD | LLM Verdict | Actual outcome | Fwd return | Scored |
|---|---|---|---|---|---|---|---|---|
| TCS | 2020-04-08 | -14.6% | 32.5% | -28.6% | HOLDS | GROWTH | +65.0% | correct |
| TCS | 2008-04-15 | -23.1% | 37.2% | -40.7% | HOLDS | DECLINE | -54.7% | **incorrect** |
| INFY | 2001-03-28 | -53.8% | 65.3% | -60.4% | WEAKENING | DECLINE | -49.9% | not scored |
| RELIANCE | 2008-04-23 | +65.2% | 42.2% | -33.8% | HOLDS | DECLINE | -60.7% | **incorrect** |
| SUNPHARMA | 1999-08-03 | +60.0% | 54.8% | -32.3% | HOLDS | GROWTH | +341.7% | correct |
| WIPRO | 2009-03-03 | -51.5% | 62.2% | -62.2% | HOLDS | GROWTH | +177.1% | correct |

**Directionally scored: 3/5 correct** (INFY's WEAKENING excluded per the
scoring rule above; 6 completed cases, 5 scored).

### The headline number is misleading on its own - the real finding
5 of 6 completed cases returned **HOLDS**, regardless of how negative the
input fundamentals were - including TCS 2008 (-41% max drawdown) and
WIPRO 2009 (-62% max drawdown, -52% trailing return). **Not one of the 6
cases returned BROKEN**, even RELIANCE 2008 with a +65% trailing return
that preceded a real -61% decline. The only non-HOLDS verdict (INFY,
WEAKENING) came from the single most extreme negative case (-54% trailing
return, -60% drawdown).

This pattern means the "3/5 correct" number is not evidence of genuine
directional discrimination. The three "correct" cases (TCS 2020,
SUNPHARMA, WIPRO) were all cases where the actual outcome happened to be
GROWTH, and the model said HOLDS regardless of input - so those
predictions were right by a default answer landing on the right side, not
because the model weighed the fundamentals and concluded continuation was
likely. The two "incorrect" cases are the ones that actually stress-test
whether the model reads negative fundamentals as a reason to say
WEAKENING/BROKEN - and in both, it did not. **This fundamentals-only
setup shows a strong HOLDS-default bias, not a working discriminator.**

## Consequences

### Known limitations
- **Fundamentals-only is a real limitation, not the full system.** The
  live Investment Thesis Agent reasons over fundamentals AND recent news
  together; this pilot tests only the fundamentals-reasoning half. A weak
  result here does not necessarily mean the full (fundamentals+news)
  system performs this poorly - but it also means this pilot cannot
  claim to validate the full system, only this narrower slice of it.
- **No real historical thesis exists for any case** - each case uses a
  generic synthetic momentum-continuation framing (see Decision above),
  not an actual investor's stated reasoning. The live agent's real job -
  checking whether a *specific* thesis's *logic* still holds - is not
  what this pilot tests.
- **Small sample (6 completed cases across 5 tickers).** This is a
  pilot demonstrating the idea works in principle for a supervisor demo,
  explicitly not a statistically powered validation. 3/5 (scored) is not
  a number that should be quoted as an accuracy figure without this
  caveat attached every time.
- **HOLDS-default bias is the dominant finding**, and it means the
  headline "3/5 correct" ratio substantially overstates how much genuine
  fundamentals-based reasoning is happening - see Validation section
  above. This should be the lead finding in any summary of this pilot,
  not the accuracy ratio.
- **INFY 1998-07-23 excluded**: only 85 days of trailing history existed
  before this date (near the ticker's start in the dataset) versus the
  252 needed - a genuine data-availability gap, not a bug.
- **HDFCBANK 2019-03-13 excluded**: the 126-day forward window contained
  a likely split/bonus artifact per `contains_likely_split()` - correctly
  caught by the guard rather than silently producing a fabricated
  "decline."
- **Dataset is not split/bonus-adjusted** (see Decision). The 35%
  single-day-move threshold used to detect this is a first-pass value,
  not swept or tuned - a genuine (non-split) 35%+ single-day move, while
  rare, would also be incorrectly excluded by this guard. Not observed
  in this pilot's date range, but worth knowing before scaling up.
- **Model used was `llama3.2:latest`, not `llama3.1`** (the project's
  usual default) - `llama3.1` failed to load locally with a CPU memory
  allocation error (`ggml_backend_cpu_buffer_type_alloc_buffer: failed to
  allocate buffer`) on the machine this pilot ran on. Not compared
  against `llama3.1`'s verdicts - the HOLDS-default pattern above might
  behave differently on a larger model, untested here.
- **Candidate dates cluster around known market-wide cycles** (dot-com
  1999-2001, 2008 GFC, 2009 recovery, 2020 COVID) rather than
  idiosyncratic single-company events, because the candidate-selection
  scan simply looks for the strongest forward returns in each ticker's
  history, and those cluster around macro cycles. This pilot mostly
  tests whether the model can read "broad market regime," not
  company-specific thesis logic.

### Follow-up (not blocking, flagged honestly)
- Re-run the same 8 cases against `llama3.1` once machine memory allows,
  to check whether the HOLDS-default bias is model-specific.
- Try to find genuinely idiosyncratic (non-macro-cycle) cases for a more
  representative test of company-specific reasoning.
- If pursued further, a larger sample (20-30 cases) would be needed
  before any accuracy number could be treated as more than illustrative.