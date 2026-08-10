import sys
sys.path.insert(0, 'src/agents')
import numpy as np
import pandas as pd
from portfolio_health_agent import (
    compute_volatility, compute_max_drawdown, compute_var, compute_cvar,
    compute_sharpe_ratio, compute_downside_deviation, RISK_FREE_RATE, TRADING_DAYS
)

# TEST 1 & 2: Volatility & Sharpe on known synthetic series
np.random.seed(42)
known_daily_mean = 0.001
known_daily_std = 0.015
returns = pd.Series(np.random.normal(known_daily_mean, known_daily_std, 500))

expected_vol = known_daily_std * np.sqrt(TRADING_DAYS)
actual_vol = compute_volatility(returns)
print(f"TEST 1: Volatility annualization")
print(f"  expected ~ {expected_vol:.4f}")
print(f"  actual   = {actual_vol:.4f}")
print(f"  PASS: {abs(expected_vol - actual_vol) < 0.01}\n")

expected_annual_return = known_daily_mean * TRADING_DAYS
expected_sharpe = (expected_annual_return - RISK_FREE_RATE) / expected_vol
actual_sharpe = compute_sharpe_ratio(returns)
print(f"TEST 2: Sharpe ratio")
print(f"  expected ~ {expected_sharpe:.4f}")
print(f"  actual   = {actual_sharpe:.4f}")
print(f"  PASS: {abs(expected_sharpe - actual_sharpe) < 0.15}\n")

# TEST 3: Max drawdown on hand-built path
prices = pd.Series([100, 110, 90, 95, 120])
manual_returns = prices.pct_change().dropna()
expected_dd = (90 - 110) / 110
actual_dd = compute_max_drawdown(manual_returns)
print(f"TEST 3: Max drawdown")
print(f"  expected = {expected_dd:.4f}")
print(f"  actual   = {actual_dd:.4f}")
print(f"  PASS: {abs(expected_dd - actual_dd) < 0.0001}\n")

# TEST 4: VaR on known distribution
sorted_returns = pd.Series(sorted(np.random.normal(0, 0.02, 1000)))
expected_var = np.percentile(sorted_returns, 5)
actual_var = compute_var(sorted_returns, confidence=0.95)
print(f"TEST 4: VaR (95%)")
print(f"  expected = {expected_var:.4f}")
print(f"  actual   = {actual_var:.4f}")
print(f"  PASS: {abs(expected_var - actual_var) < 0.0001}\n")

# TEST 5: Downside deviation denominator
test_returns = pd.Series([0.02]*8 + [-0.03, -0.05])
downside_days = test_returns[test_returns < 0]
manual_dd = np.sqrt((downside_days**2).sum() / len(test_returns)) * np.sqrt(TRADING_DAYS)
actual_dd_calc = compute_downside_deviation(test_returns)
print(f"TEST 5: Downside deviation denominator")
print(f"  expected (div by total 10 days) = {manual_dd:.4f}")
print(f"  actual                          = {actual_dd_calc:.4f}")
print(f"  PASS: {abs(manual_dd - actual_dd_calc) < 0.0001}")
wrong_denominator = np.sqrt((downside_days**2).sum() / len(downside_days)) * np.sqrt(TRADING_DAYS)
print(f"  (wrong denominator would give: {wrong_denominator:.4f} - confirms fix is present)")