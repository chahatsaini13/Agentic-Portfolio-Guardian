# Week 4 Validation Note — Portfolio Health Agent (extended)

**Validated by:** [your name]
**Date:** 2026-08-11
**Script version:** `portfolio_health_agent.py` (extended, per ADR 0003)
**Run environment:** Local machine (Windows), real yfinance/internet access

## Scope
Independently validated the extended Portfolio Health Agent against a new
realistic 12-holding test portfolio (`data/sample_portfolio_test.csv`),
plus a deliberately-overweighted variant
(`data/sample_portfolio_overweight_test.csv`) to confirm alert logic, plus
synthetic-data formula spot-checks for the risk metrics.

## Test data
`data/sample_portfolio_test.csv` — 12 NSE holdings, 7 sectors, weights sum
to exactly 100.0%. Deliberately includes a cross-sector theme
(`usd_export`: TCS, INFY, SUNPHARMA, CIPLA = 37% combined) that exceeds the
default 35% theme limit while no individual stock/sector breaches its own
limit — chosen to test that theme-level exposure detection works
independently of sector/stock-level checks.

## Results

### 1. `--basic` mode (v1 output, no network)
| Check | Expected | Actual | Result |
|---|---|---|---|
| Sector weights sum | ~100% | 100% (7 sectors) | PASS |
| Largest sector | Banking, 25% | Banking, 0.25 | PASS |
| Largest holding | HDFCBANK.NS, 14% | HDFCBANK.NS, 0.14 | PASS |
| HHI score | 0.1786 (hand-calc: Σ sector_weight²) | 0.1786 | PASS |

### 2. Full analysis — realistic portfolio, real yfinance data
| Check | Expected | Actual | Result |
|---|---|---|---|
| `exposure_alerts` | exactly 1 alert (usd_export theme, 37% > 35%) | 1 alert, correct type/weight | PASS |
| `diversification_score` | 87.9 (hand-calc) | 87.9 | PASS |
| `volatility_annualized` | plausible range (10–25%) | 15.23% | PASS |
| `beta_vs_benchmark` | close to 1.0 for a diversified equity portfolio | 1.0259 | PASS |
| `max_drawdown` | negative, plausible range (-15% to -40%) | -23.04% | PASS |
| `value_at_risk_95` vs `conditional_var_95` | CVaR should be more negative than VaR | -1.56% vs -2.21% | PASS |
| `downside_deviation` vs `volatility_annualized` | downside dev should be < total vol (see Test 5 below re: ADR 0003 fix) | 0.1132 < 0.1523 | PASS |
| `sharpe_ratio` | -0.927 — negative, but explainable (see note) | -0.927 | PASS (verified not a bug) |
| `TCS.NS` / `INFY.NS` correlation | high (both IT/export names) | 0.799 | PASS — matches real-world intuition |
| `portfolio_health_score` | 90.4 (hand-calc: 100 − 0 − 0 + (-0.927×5) − 5×1) | 90.4 | PASS |

**Note on negative Sharpe ratio:** Initially looks concerning, but this is
correct behavior, not a bug. The formula was independently verified in
section 4 below. A negative Sharpe here reflects that this specific test
portfolio genuinely lost value over the trailing 1-year window (-7.8%
cumulative) while the NIFTY benchmark was roughly flat (+0.81%) — a real
market outcome, not a calculation error.

### 3. Full analysis — deliberately overweight portfolio
Overweighted HDFCBANK to 22%, ICICIBANK to 20%, Banking sector to 42%.

| Alert | Expected to fire | Fired | Result |
|---|---|---|---|
| single_stock (HDFCBANK, 22% > 15%) | yes | yes | PASS |
| single_stock (ICICIBANK, 20% > 15%) | yes | yes | PASS |
| sector (Banking, 42% > 30%) | yes | yes | PASS |
| theme (usd_export, 37% > 35%) | yes | yes | PASS |
| `position_size_flags` cross-check | HDFCBANK + ICICIBANK flagged `position_too_large` | both flagged | PASS |
| `portfolio_health_score` | 80.0 (hand-calc: 100 − 0 − 0 + 0 − 5×4) | 80.0 | PASS |

### 4. Formula spot-checks (synthetic data, `scripts/validate_formulas.py`)
| Formula | Method | Result |
|---|---|---|
| Volatility annualization | Normal(μ=0.001, σ=0.015) daily returns, 500 days | matches `daily_std × √252` within tolerance | PASS |
| Sharpe ratio | same series | matches `(annualized_return − rf) / annualized_vol` within tolerance | PASS |
| Max drawdown | hand-built path 100→110→90→95→120 | exact match: −0.1818 (trough 90 vs peak 110) | PASS |
| VaR (95%) | 1000 Normal(0, 0.02) returns | exact match to `np.percentile(returns, 5)` | PASS |
| Downside deviation denominator | 8 up days / 2 down days | exact match to "÷ total days" (correct); confirmed it does NOT match the "÷ downside days only" wrong version (0.65 vs correct 0.29) — **confirms the ADR 0003 bug fix is actually present in the code** | PASS |

## Bugs found
None. Every check above passed — no code changes made to
`portfolio_health_agent.py`.

## Files added
- `data/sample_portfolio_test.csv` — realistic 12-holding test portfolio
- `data/sample_portfolio_overweight_test.csv` — deliberately overweighted
  variant for exposure-alert testing
- `scripts/validate_formulas.py` — synthetic-data formula spot-check script
- This note