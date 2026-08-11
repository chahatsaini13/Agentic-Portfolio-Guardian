# ADR 0004: Extend Portfolio Health Agent with full analytics scope

## Status
Accepted

## Context
The original Week 4 scope for the Portfolio Health Agent (`portfolio_health_agent.py`,
v1) covered only structural analysis: loading holdings, sector weights, and
HHI-based concentration. A broader analytics scope was requested — asset-
class/market-cap/industry allocation, diversification metrics beyond HHI,
exposure/liquidity/position checks, historical risk metrics (volatility,
beta, drawdown, VaR, CVaR, downside deviation, risk contribution),
performance metrics (CAGR, Sharpe, Sortino, Treynor, Alpha, Information
Ratio), correlation/covariance analysis, MPT portfolio optimization
(min-variance / max-Sharpe), rebalancing recommendations with cost/tax
estimates, and an AI-generated health summary.

Cross-checked against `ROADMAP.md`: most of this had no existing home in
Weeks 4-9. Two subsections were added to the roadmap (Portfolio Analytics
under Week 4, Reporting under Week 8) before any code was written, so the
scope expansion is tracked, not silent.

## Decision
Extend `portfolio_health_agent.py` in place rather than starting a new
agent, since all of this operates on the same portfolio input and most of
it (allocation, diversification, risk, performance) is naturally scoped
to "how healthy is this portfolio," which is this agent's job. Two tiers
of functions:

- **Snapshot-only** (holdings + optional yfinance metadata, no historical
  prices needed): allocation breakdowns, diversification metrics beyond
  HHI, thematic overlap, exposure/liquidity/position-size checks,
  rebalancing recommendations, health scoring.
- **Price-history-dependent** (needs `yfinance.download()` over a date
  range plus a benchmark index, currently NIFTY 50 via `^NSEI`):
  volatility, beta, drawdown, VaR/CVaR, downside deviation, risk
  contribution, all performance ratios, correlation/covariance matrices,
  MPT optimization.

Every function is standalone-callable (same dependency-light philosophy
as ADR 0001), and `run()` / `load_portfolio()` / `compute_sector_weights()`
/ `compute_concentration_metrics()` keep their original v1 signatures and
output shape unchanged, so nothing built against v1's output breaks. A
new `run_full_analysis()` orchestrates the extended report; `--basic`
CLI flag still produces the exact v1 output for a quick backward-compat
check.

PDF export was deliberately NOT implemented with a real PDF library
(e.g. `reportlab`) to keep dependencies minimal - it falls back to a
`.txt` file with a warning printed, rather than silently producing
something that isn't actually a PDF.

## Consequences

### New dependencies
`numpy`, `pandas`, `scipy` added to `requirements.txt` (alongside the
existing `yfinance`, `requests`, `python-dotenv`) - required for VaR/CVaR,
Sharpe/Sortino/Treynor calculations, correlation/covariance matrices, and
`scipy.optimize.minimize` for MPT.

### Bugs found and fixed during real-data validation
The extended agent was first validated with synthetic price data (known
properties: e.g. min-variance volatility must be ≤ equal-weight volatility,
computed portfolio beta must match a known weighted-average beta) since
this environment couldn't reach Yahoo Finance. Two real bugs only surfaced
once run against live data on the developer's machine:

1. **`dropna()` cascade failure.** `TATAMOTORS.NS` returned a 404 from
   Yahoo Finance (the ticker was retired after Tata Motors' October 2025
   demerger into `TMPV.NS` / `TMCV.NS`). The original
   `compute_daily_returns()` called `.dropna()` with default row-wise
   behavior, which drops any row containing a NaN in *any* column - since
   the delisted ticker's entire column was NaN, every single row was
   dropped, silently wiping out risk/performance/correlation/MPT results
   for all 7 other (perfectly fine) tickers, not just the bad one.
   **Fix:** `fetch_price_history()` now detects fully-NaN columns and
   drops only those specific tickers (with a `[warn]` naming them),
   before any row-level `dropna()` runs. Verified with a synthetic test:
   one all-NaN column no longer empties the whole returns dataframe.
   Also switched to `ffill()` before computing returns, so small gaps
   (holidays, one exchange closed) don't cascade either.

2. **Downside deviation used the wrong denominator.** The standard
   Sortino-ratio downside deviation divides the sum of squared downside
   deviations by the *total* number of observations (upside days count
   as zero-contribution, not as excluded). The initial implementation
   divided by the count of downside days only, which overstates downside
   deviation whenever downside days are a minority of the period -
   caught because downside deviation (0.1701) came out *higher* than
   total annualized volatility (0.1559) on the real run, which shouldn't
   normally happen. Confirmed via hand-calculated synthetic case (8 up
   days, 2 down days) that the fix produces downside deviation below
   total volatility, as expected. This also corrected the Sortino ratio
   (-1.3766 -> -1.9667 on the real portfolio after the fix - the ratio's
   magnitude increased because the smaller, corrected denominator makes
   the same negative excess return look sharper).

Both bugs were caught by treating the developer's real yfinance run as
the actual validation step, not the synthetic-data pass - synthetic data
confirmed the formulas were internally consistent, but couldn't have
caught the ticker-delisting cascade, and only barely surfaced the
downside-deviation issue by coincidence of the numbers looking wrong.

### Known limitation, not a bug
`max_sharpe_portfolio` optimization returned a 100%-single-asset corner
solution (all weight in `MARUTI.NS`) on the real portfolio. This is
expected behavior for unconstrained mean-variance optimization over a
small number of assets and a short lookback window - it is not a defect,
but it also isn't something to present as investment advice as-is. A
future improvement would add a per-asset weight cap (e.g. max 25%) to
the optimizer's constraints if more realistic diversified suggestions
are wanted.

### Data caveats carried forward into the report
- `rebalancing_cost_estimate` assumes short-term capital gains tax on
  every sell, since no cost-basis or holding-period tracking exists yet -
  explicitly flagged as a simplified estimate, not real tax advice.
- `portfolio_health_score` is a starting heuristic (baseline 100, penalize/
  reward diversification, volatility, Sharpe, exposure alerts) - not a
  validated scoring model. Worth tuning once run against a few more real
  portfolios.
- Snapshot-only sections (allocation, diversification, exposure checks)
  were validated end-to-end without network access; price-history sections
  could only be formula-verified via synthetic data in this environment
  and were confirmed correct against the developer's real yfinance run.
