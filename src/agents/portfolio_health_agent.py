"""
Portfolio Health Agent - Portfolio Guardian (extended)

Second agent in the pipeline. load_portfolio / compute_sector_weights /
compute_concentration_metrics / run() are unchanged from v1 so nothing
built on top of the old output breaks. Everything else here is the
extended analytics scope from ROADMAP.md.

Snapshot stuff (allocation, diversification, exposure checks, rebalancing,
health score) doesn't need historical prices. Risk/performance/correlation/
MPT optimization all need yfinance price history + a benchmark - if that
fetch fails, those sections just come back None instead of crashing.

Usage:
    python portfolio_health_agent.py --file data/sample_portfolio.csv
    python portfolio_health_agent.py --file data/sample_portfolio.csv --summary --export json
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from scipy.optimize import minimize

RISK_FREE_RATE = 0.07          # rough proxy for Indian 10Y G-sec yield
TRADING_DAYS = 252
DEFAULT_BENCHMARK = "^NSEI"    # NIFTY 50
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1")

DEFAULT_RISK_CONFIG = {
    "max_single_stock_weight": 0.15,
    "max_single_sector_weight": 0.30,
    "max_theme_weight": 0.35,
    "min_position_weight": 0.01,
    "min_avg_volume": 100_000,
    "var_confidence": 0.95,
}


def _r(x, n=4):
    """round() that passes None through instead of blowing up - saves
    writing 'if x is not None else None' everywhere below."""
    return round(x, n) if x is not None else None


def load_portfolio(filepath: str) -> list:
    """CSV -> list of dicts. Needs ticker/sector/weight columns.
    asset_class and theme are optional, default to Equity / None."""
    holdings = []
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            holdings.append({
                "ticker": row["ticker"].strip(),
                "sector": row["sector"].strip(),
                "weight": float(row["weight"]) / 100,
                "asset_class": (row.get("asset_class") or "Equity").strip(),
                "theme": (row.get("theme") or "").strip() or None,
            })

    total = sum(h["weight"] for h in holdings)
    if not (0.99 <= total <= 1.01):
        print(f"[warn] portfolio weights sum to {total*100:.1f}%, expected ~100%. "
              f"Check {filepath} for typos.")

    return holdings


def compute_sector_weights(holdings: list) -> dict:
    sector_weights = {}
    for h in holdings:
        sector_weights[h["sector"]] = sector_weights.get(h["sector"], 0) + h["weight"]
    return sector_weights


def compute_concentration_metrics(holdings: list, sector_weights: dict) -> dict:
    hhi_score = sum(w ** 2 for w in sector_weights.values())
    largest_holding = max(holdings, key=lambda h: h["weight"])
    largest_sector_name = max(sector_weights, key=sector_weights.get)

    return {
        "hhi_score": _r(hhi_score),
        "largest_holding": {
            "ticker": largest_holding["ticker"],
            "weight": _r(largest_holding["weight"]),
        },
        "largest_sector": {
            "sector": largest_sector_name,
            "weight": _r(sector_weights[largest_sector_name]),
        },
    }


# ---- allocation breakdowns ----

def compute_asset_class_allocation(holdings: list) -> dict:
    alloc = {}
    for h in holdings:
        ac = h["asset_class"]
        alloc[ac] = alloc.get(ac, 0) + h["weight"]
    return {k: _r(v) for k, v in alloc.items()}


def bucket_market_cap(market_cap) -> str:
    """Rough INR-crore buckets - not SEBI's official ranked classification,
    just a stand-in that doesn't need the full market universe."""
    if market_cap is None:
        return "Unknown"
    cr = market_cap / 1e7
    if cr >= 20_000:
        return "Large Cap"
    elif cr >= 5_000:
        return "Mid Cap"
    return "Small Cap"


def fetch_market_data(holdings: list) -> dict:
    """Industry/market-cap/volume per ticker via yfinance. One ticker
    failing doesn't kill the rest - just falls back to Unknown/None."""
    data = {}
    for h in holdings:
        ticker = h["ticker"]
        try:
            info = yf.Ticker(ticker).info
            data[ticker] = {
                "industry": info.get("industry") or "Unknown",
                "market_cap": info.get("marketCap"),
                "avg_volume": info.get("averageVolume"),
            }
        except Exception as e:
            print(f"[warn] could not fetch market data for {ticker}: {e}")
            data[ticker] = {"industry": "Unknown", "market_cap": None, "avg_volume": None}
    return data


def compute_industry_allocation(holdings: list, market_data: dict) -> dict:
    alloc = {}
    for h in holdings:
        industry = market_data.get(h["ticker"], {}).get("industry", "Unknown")
        alloc[industry] = alloc.get(industry, 0) + h["weight"]
    return {k: _r(v) for k, v in alloc.items()}


def compute_market_cap_allocation(holdings: list, market_data: dict) -> dict:
    alloc = {}
    for h in holdings:
        bucket = bucket_market_cap(market_data.get(h["ticker"], {}).get("market_cap"))
        alloc[bucket] = alloc.get(bucket, 0) + h["weight"]
    return {k: _r(v) for k, v in alloc.items()}


# ---- diversification beyond HHI, thematic overlap, exposure checks ----

def compute_diversification_metrics(holdings: list, sector_weights: dict) -> dict:
    """effective_number_of_holdings = 1/HHI - 10 equal-weighted stocks
    gives you 10, one dominated by a 50% position gives you way less than
    its literal count. diversification_score is that as a % of actual
    holding count (100 = perfectly equal-weighted)."""
    holding_weights = [h["weight"] for h in holdings]
    holding_hhi = sum(w ** 2 for w in holding_weights)
    effective_n_holdings = (1 / holding_hhi) if holding_hhi else 0

    sector_hhi = sum(w ** 2 for w in sector_weights.values())
    effective_n_sectors = (1 / sector_hhi) if sector_hhi else 0

    diversification_score = round((effective_n_holdings / len(holdings)) * 100, 2) if holdings else 0

    return {
        "effective_number_of_holdings": round(effective_n_holdings, 2),
        "effective_number_of_sectors": round(effective_n_sectors, 2),
        "holding_concentration_score": _r(holding_hhi),
        "diversification_score": diversification_score,
    }


def compute_thematic_overlap(holdings: list) -> dict:
    """Weight by theme tag - catches stuff that spans sectors but shares
    one driver (e.g. everything exposed to USD/INR). Untagged holdings
    just don't show up here."""
    theme_weights = {}
    for h in holdings:
        theme = h["theme"]
        if not theme:
            continue
        theme_weights[theme] = theme_weights.get(theme, 0) + h["weight"]
    return {k: _r(v) for k, v in theme_weights.items()}


def check_exposure_limits(holdings: list, sector_weights: dict, theme_weights: dict,
                           config: dict = None) -> list:
    config = config or DEFAULT_RISK_CONFIG
    alerts = []
    for h in holdings:
        if h["weight"] > config["max_single_stock_weight"]:
            alerts.append({"type": "single_stock", "ticker": h["ticker"],
                            "weight": _r(h["weight"]), "limit": config["max_single_stock_weight"]})
    for sector, w in sector_weights.items():
        if w > config["max_single_sector_weight"]:
            alerts.append({"type": "sector", "sector": sector,
                            "weight": _r(w), "limit": config["max_single_sector_weight"]})
    for theme, w in theme_weights.items():
        if w > config["max_theme_weight"]:
            alerts.append({"type": "theme", "theme": theme,
                            "weight": _r(w), "limit": config["max_theme_weight"]})
    return alerts


def check_liquidity(holdings: list, market_data: dict, config: dict = None) -> list:
    config = config or DEFAULT_RISK_CONFIG
    flags = []
    for h in holdings:
        vol = market_data.get(h["ticker"], {}).get("avg_volume")
        if vol is not None and vol < config["min_avg_volume"]:
            flags.append({"ticker": h["ticker"], "avg_volume": vol, "issue": "low_liquidity"})
    return flags


def check_position_sizes(holdings: list, config: dict = None) -> list:
    config = config or DEFAULT_RISK_CONFIG
    flags = []
    for h in holdings:
        if h["weight"] < config["min_position_weight"]:
            flags.append({"ticker": h["ticker"], "weight": _r(h["weight"]), "issue": "position_too_small"})
        elif h["weight"] > config["max_single_stock_weight"]:
            flags.append({"ticker": h["ticker"], "weight": _r(h["weight"]), "issue": "position_too_large"})
    return flags


# ---- price history (everything below needs this) ----

def fetch_price_history(tickers: list, benchmark: str = DEFAULT_BENCHMARK, period: str = "1y"):
    """Daily close prices for holdings + benchmark. If a ticker comes back
    totally empty (delisted, renamed after a demerger, typo) we drop just
    that column instead of letting it wipe out every row via dropna() -
    used to be a bug here, see ADR 0003."""
    all_tickers = list(dict.fromkeys(tickers + [benchmark]))
    try:
        raw = yf.download(all_tickers, period=period, progress=False, auto_adjust=True)["Close"]
    except Exception as e:
        print(f"[warn] price history fetch failed entirely: {e}")
        return None

    if isinstance(raw, pd.Series):  # only happens if just one ticker came back
        raw = raw.to_frame(name=all_tickers[0])

    fully_missing = [c for c in raw.columns if raw[c].isna().all()]
    if fully_missing:
        print(f"[warn] no price data for: {fully_missing} - dropping. "
              f"If a ticker got renamed, update the CSV with the current symbol.")
        raw = raw.drop(columns=fully_missing)

    if raw.empty or benchmark not in raw.columns:
        print("[warn] no usable price data left after dropping missing tickers.")
        return None if raw.empty else raw

    return raw.dropna(how="all")


def compute_daily_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill small gaps (holidays etc.) before pct_change so one
    missing day in one column doesn't drop that row for every ticker."""
    filled = price_df.ffill()
    return filled.pct_change(fill_method=None).dropna(how="any")


def compute_portfolio_returns(returns_df: pd.DataFrame, weights_dict: dict) -> pd.Series:
    """Weighted sum of daily returns, renormalized over whatever tickers
    actually have price data."""
    tickers = [t for t in returns_df.columns if t in weights_dict]
    if not tickers:
        return pd.Series(dtype=float)
    w = np.array([weights_dict[t] for t in tickers])
    w = w / w.sum()
    return returns_df[tickers].dot(w)


# ---- risk metrics ----

def compute_volatility(returns: pd.Series, annualize: bool = True) -> float:
    vol = returns.std()
    return vol * np.sqrt(TRADING_DAYS) if annualize else vol


def compute_beta(portfolio_returns: pd.Series, benchmark_returns: pd.Series):
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 2:
        return None
    cov = aligned.cov().iloc[0, 1]
    var = aligned.iloc[:, 1].var()
    return (cov / var) if var else None


def compute_max_drawdown(returns: pd.Series) -> float:
    """Worst peak-to-trough decline, as a negative fraction."""
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()


def compute_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical VaR - the daily loss exceeded (1-confidence)% of the time."""
    return float(np.percentile(returns, (1 - confidence) * 100))


def compute_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """Average loss on the days that breached VaR - how bad the bad days are."""
    var = compute_var(returns, confidence)
    tail = returns[returns <= var]
    return float(tail.mean()) if len(tail) else var


def compute_downside_deviation(returns: pd.Series, target: float = 0.0) -> float:
    """Sortino's downside deviation - only counts returns below target,
    but divides by the TOTAL day count (not just downside days), same way
    regular volatility counts every day. Dividing by downside-count-only
    overstates this - caught that bug once, see ADR 0003."""
    if not len(returns):
        return 0.0
    downside = returns[returns < target] - target
    return float(np.sqrt((downside ** 2).sum() / len(returns)) * np.sqrt(TRADING_DAYS))


def compute_risk_contribution(weights_dict: dict, returns_df: pd.DataFrame) -> dict:
    """Each holding's share of total portfolio variance - not just its
    weight. A small volatile uncorrelated position can punch above its
    weight here."""
    tickers = [t for t in returns_df.columns if t in weights_dict]
    if not tickers:
        return {}
    w = np.array([weights_dict[t] for t in tickers])
    w = w / w.sum()
    cov = returns_df[tickers].cov().values * TRADING_DAYS
    port_var = w @ cov @ w
    if port_var == 0:
        return {t: 0.0 for t in tickers}
    marginal_contrib = cov @ w
    risk_contrib = w * marginal_contrib / port_var
    return {tickers[i]: round(float(risk_contrib[i]), 4) for i in range(len(tickers))}


def compute_risk_metrics(holdings: list, returns_df: pd.DataFrame, benchmark_returns: pd.Series,
                          confidence: float = 0.95) -> dict:
    weights_dict = {h["ticker"]: h["weight"] for h in holdings}
    port_returns = compute_portfolio_returns(returns_df, weights_dict)
    if port_returns.empty:
        return None
    beta = compute_beta(port_returns, benchmark_returns) if benchmark_returns is not None else None

    return {
        "volatility_annualized": _r(float(compute_volatility(port_returns))),
        "beta_vs_benchmark": _r(float(beta)) if beta is not None else None,
        "standard_deviation_daily": _r(float(port_returns.std())),
        "max_drawdown": _r(float(compute_max_drawdown(port_returns))),
        "value_at_risk_95": _r(compute_var(port_returns, confidence)),
        "conditional_var_95": _r(compute_cvar(port_returns, confidence)),
        "downside_deviation": _r(compute_downside_deviation(port_returns)),
        "risk_contribution_by_holding": compute_risk_contribution(weights_dict, returns_df),
    }


# ---- performance metrics ----

def compute_cumulative_return(returns: pd.Series) -> float:
    return float((1 + returns).prod() - 1)


def compute_cagr(returns: pd.Series, periods_per_year: int = TRADING_DAYS):
    n_years = len(returns) / periods_per_year
    if n_years <= 0:
        return None
    cumulative = (1 + returns).prod()
    return float(cumulative ** (1 / n_years) - 1)


def compute_annualized_return(returns: pd.Series) -> float:
    return float(returns.mean() * TRADING_DAYS)


def compute_sharpe_ratio(returns: pd.Series, risk_free_rate: float = RISK_FREE_RATE):
    vol = compute_volatility(returns)
    if not vol:
        return None
    return float((compute_annualized_return(returns) - risk_free_rate) / vol)


def compute_sortino_ratio(returns: pd.Series, risk_free_rate: float = RISK_FREE_RATE):
    dd = compute_downside_deviation(returns)
    if not dd:
        return None
    return float((compute_annualized_return(returns) - risk_free_rate) / dd)


def compute_treynor_ratio(returns: pd.Series, beta: float, risk_free_rate: float = RISK_FREE_RATE):
    if not beta:
        return None
    return float((compute_annualized_return(returns) - risk_free_rate) / beta)


def compute_alpha(portfolio_return: float, beta: float, benchmark_return: float,
                   risk_free_rate: float = RISK_FREE_RATE):
    """CAPM alpha - actual return minus what CAPM predicts given the beta."""
    if beta is None:
        return None
    expected_return = risk_free_rate + beta * (benchmark_return - risk_free_rate)
    return float(portfolio_return - expected_return)


def compute_information_ratio(portfolio_returns: pd.Series, benchmark_returns: pd.Series):
    """Active return over benchmark, scaled by tracking error - not just
    whether it beat the benchmark, but how consistently."""
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 2:
        return None
    active_return = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    tracking_error = active_return.std() * np.sqrt(TRADING_DAYS)
    if not tracking_error:
        return None
    return float((active_return.mean() * TRADING_DAYS) / tracking_error)


def compute_performance_metrics(holdings: list, returns_df: pd.DataFrame,
                                 benchmark_returns: pd.Series,
                                 risk_free_rate: float = RISK_FREE_RATE) -> dict:
    weights_dict = {h["ticker"]: h["weight"] for h in holdings}
    port_returns = compute_portfolio_returns(returns_df, weights_dict)
    if port_returns.empty:
        return None

    beta = compute_beta(port_returns, benchmark_returns) if benchmark_returns is not None else None
    port_annual_return = compute_annualized_return(port_returns)
    bench_annual_return = compute_annualized_return(benchmark_returns) if benchmark_returns is not None else None

    sharpe = compute_sharpe_ratio(port_returns, risk_free_rate)
    sortino = compute_sortino_ratio(port_returns, risk_free_rate)
    treynor = compute_treynor_ratio(port_returns, beta, risk_free_rate) if beta else None
    alpha = compute_alpha(port_annual_return, beta, bench_annual_return, risk_free_rate) if (
        beta is not None and bench_annual_return is not None) else None
    info_ratio = compute_information_ratio(port_returns, benchmark_returns) if benchmark_returns is not None else None
    excess_return = (port_annual_return - bench_annual_return) if bench_annual_return is not None else None

    return {
        "cumulative_return": _r(compute_cumulative_return(port_returns)),
        "cagr": _r(compute_cagr(port_returns)),
        "annualized_return": _r(port_annual_return),
        "sharpe_ratio": _r(sharpe),
        "sortino_ratio": _r(sortino),
        "treynor_ratio": _r(treynor),
        "alpha_vs_benchmark": _r(alpha),
        "information_ratio": _r(info_ratio),
        "benchmark_annualized_return": _r(bench_annual_return),
        "excess_return_vs_benchmark": _r(excess_return),
    }


# ---- correlation / covariance ----

def compute_correlation_matrix(returns_df: pd.DataFrame) -> dict:
    """Catches hidden concentration - two stocks in different sectors can
    still move together (e.g. both exposed to USD/INR), which sector
    weight alone won't show."""
    return returns_df.corr().round(4).to_dict()


def compute_covariance_matrix(returns_df: pd.DataFrame) -> dict:
    return (returns_df.cov() * TRADING_DAYS).round(6).to_dict()


# ---- MPT optimization ----

def _portfolio_perf(weights, mean_returns, cov_matrix):
    ret = float(np.dot(weights, mean_returns) * TRADING_DAYS)
    vol = float(np.sqrt(weights.T @ cov_matrix @ weights * TRADING_DAYS))
    return ret, vol


def _optimize(returns_df: pd.DataFrame, objective_fn) -> dict:
    tickers = list(returns_df.columns)
    n = len(tickers)
    mean_returns = returns_df.mean().values
    cov_matrix = returns_df.cov().values

    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1},)
    bounds = tuple((0, 1) for _ in range(n))  # long-only, no leverage
    init = np.array([1 / n] * n)

    result = minimize(objective_fn, init, args=(mean_returns, cov_matrix),
                       method="SLSQP", bounds=bounds, constraints=constraints)
    weights = result.x
    ret, vol = _portfolio_perf(weights, mean_returns, cov_matrix)
    return {
        "weights": {tickers[i]: round(float(weights[i]), 4) for i in range(n)},
        "expected_annual_return": round(ret, 4),
        "expected_annual_volatility": round(vol, 4),
        "converged": bool(result.success),
    }


def optimize_min_variance(returns_df: pd.DataFrame) -> dict:
    def objective(w, mean_returns, cov_matrix):
        return _portfolio_perf(w, mean_returns, cov_matrix)[1]
    return _optimize(returns_df, objective)


def optimize_max_sharpe(returns_df: pd.DataFrame, risk_free_rate: float = RISK_FREE_RATE) -> dict:
    def objective(w, mean_returns, cov_matrix):
        ret, vol = _portfolio_perf(w, mean_returns, cov_matrix)
        return -(ret - risk_free_rate) / vol if vol else 0
    result = _optimize(returns_df, objective)
    vol = result["expected_annual_volatility"]
    ret = result["expected_annual_return"]
    result["expected_sharpe_ratio"] = round((ret - risk_free_rate) / vol, 4) if vol else None
    return result


# ---- rebalancing + cost estimate ----

def compute_rebalancing_recommendations(holdings: list, target_allocations: dict) -> list:
    """target_allocations: {ticker: target_weight_fraction}."""
    current = {h["ticker"]: h["weight"] for h in holdings}
    all_tickers = set(current) | set(target_allocations)
    recs = []
    for ticker in all_tickers:
        cur_w = current.get(ticker, 0)
        tgt_w = target_allocations.get(ticker, 0)
        diff = tgt_w - cur_w
        if abs(diff) < 1e-6:
            continue
        recs.append({
            "ticker": ticker,
            "current_weight": _r(cur_w),
            "target_weight": _r(tgt_w),
            "action": "BUY" if diff > 0 else "SELL",
            "weight_change": _r(diff),
        })
    return sorted(recs, key=lambda r: abs(r["weight_change"]), reverse=True)


def estimate_rebalancing_costs(recommendations: list, portfolio_value: float,
                                transaction_cost_pct: float = 0.001,
                                stcg_tax_rate: float = 0.15) -> dict:
    """Rough estimate, not tax advice - assumes STCG on every sell since
    there's no cost-basis/holding-period tracking yet."""
    estimates = []
    total_cost = 0.0
    for rec in recommendations:
        trade_value = abs(rec["weight_change"]) * portfolio_value
        txn_cost = trade_value * transaction_cost_pct
        tax_cost = trade_value * stcg_tax_rate if rec["action"] == "SELL" else 0
        total = txn_cost + tax_cost
        total_cost += total
        estimates.append({**rec, "trade_value": round(trade_value, 2),
                           "transaction_cost": round(txn_cost, 2),
                           "estimated_tax": round(tax_cost, 2),
                           "total_cost": round(total, 2)})
    return {"per_trade": estimates, "total_estimated_cost": round(total_cost, 2),
            "note": "Simplified estimate - assumes STCG on all sells, no cost-basis tracking."}


# ---- health score, AI summary, export ----

def compute_health_score(diversification: dict, risk_metrics: dict,
                          performance_metrics: dict, exposure_alerts: list) -> float:
    """Rough heuristic, not a validated model - tune once you've run this
    against a few real portfolios. Starts at 100, penalize/reward from there."""
    score = 100.0
    score -= max(0, (70 - diversification.get("diversification_score", 70)) * 0.3)

    vol = (risk_metrics or {}).get("volatility_annualized")
    if vol:
        score -= max(0, (vol - 0.25) * 100)

    sharpe = (performance_metrics or {}).get("sharpe_ratio")
    if sharpe is not None:
        score += min(10, max(-10, sharpe * 5))

    score -= len(exposure_alerts) * 5
    return round(max(0, min(100, score)), 1)


def generate_health_summary(report: dict) -> str:
    """Same local Ollama pattern as the Thesis Agent. Falls back to a
    plain message instead of crashing the whole report if Ollama's down."""
    prompt = f"""You are a portfolio analyst. Summarize this portfolio health
report in plain language for a retail investor, in 4-6 sentences, then give
2-3 actionable recommendations. Do not give buy/sell advice on individual
stocks - focus on structural portfolio health (diversification, risk,
concentration). Respond in plain text only, no markdown, no JSON.

PORTFOLIO REPORT:
{json.dumps(report, indent=2, default=str)}
"""
    try:
        resp = requests.post(OLLAMA_URL, json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
                              timeout=120)
        resp.raise_for_status()
        return resp.json()["response"].strip()
    except requests.exceptions.ConnectionError:
        return "[Ollama not running - AI summary unavailable. Run `ollama serve` to enable it.]"
    except Exception as e:
        return f"[AI summary failed: {e}]"


def export_report(report: dict, filepath: str, fmt: str = "json") -> str:
    fmt = fmt.lower()
    if fmt == "json":
        with open(filepath, "w") as f:
            json.dump(report, f, indent=2, default=str)
        return filepath

    if fmt == "csv":
        # nested dicts/lists get JSON-stringified into one cell rather
        # than lost - CSV isn't great for nested data but stays complete
        flat = {k: (json.dumps(v, default=str) if isinstance(v, (dict, list)) else v)
                for k, v in report.items()}
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(flat.keys())
            writer.writerow(flat.values())
        return filepath

    if fmt == "pdf":
        # not pulling in reportlab just for this - .txt fallback instead
        # of silently faking a PDF
        print("[warn] PDF export needs reportlab (not installed). Writing a .txt fallback.")
        txt_path = filepath.rsplit(".", 1)[0] + ".txt"
        with open(txt_path, "w") as f:
            f.write(json.dumps(report, indent=2, default=str))
        return txt_path

    raise ValueError(f"Unsupported export format: {fmt}")


# ---- orchestration ----

def run(filepath: str) -> dict:
    """Original v1 output, unchanged."""
    holdings = load_portfolio(filepath)
    sector_weights = compute_sector_weights(holdings)
    concentration = compute_concentration_metrics(holdings, sector_weights)

    return {
        "sector_weights": {k: _r(v) for k, v in sector_weights.items()},
        "hhi_score": concentration["hhi_score"],
        "largest_holding": concentration["largest_holding"],
        "largest_sector": concentration["largest_sector"],
    }


def run_full_analysis(filepath: str, target_allocations: dict = None,
                       benchmark: str = DEFAULT_BENCHMARK, period: str = "1y",
                       risk_config: dict = None, fetch_market: bool = True,
                       fetch_prices: bool = True, generate_summary: bool = False,
                       portfolio_value: float = 1_000_000) -> dict:
    holdings = load_portfolio(filepath)
    sector_weights = compute_sector_weights(holdings)
    concentration = compute_concentration_metrics(holdings, sector_weights)
    diversification = compute_diversification_metrics(holdings, sector_weights)
    theme_weights = compute_thematic_overlap(holdings)
    exposure_alerts = check_exposure_limits(holdings, sector_weights, theme_weights, risk_config)
    position_flags = check_position_sizes(holdings, risk_config)
    asset_class_allocation = compute_asset_class_allocation(holdings)

    market_data, industry_allocation, market_cap_allocation, liquidity_flags = {}, {}, {}, []
    if fetch_market:
        market_data = fetch_market_data(holdings)
        industry_allocation = compute_industry_allocation(holdings, market_data)
        market_cap_allocation = compute_market_cap_allocation(holdings, market_data)
        liquidity_flags = check_liquidity(holdings, market_data, risk_config)

    risk_metrics = performance_metrics = correlation_matrix = covariance_matrix = None
    min_var_portfolio = max_sharpe_portfolio = None

    if fetch_prices:
        tickers = [h["ticker"] for h in holdings]
        price_df = fetch_price_history(tickers, benchmark=benchmark, period=period)
        if price_df is not None and not price_df.empty:
            returns_df = compute_daily_returns(price_df)
            if benchmark in returns_df.columns:
                benchmark_returns = returns_df[benchmark]
                holding_returns_df = returns_df.drop(columns=[benchmark])
            else:
                benchmark_returns = None
                holding_returns_df = returns_df

            risk_metrics = compute_risk_metrics(holdings, holding_returns_df, benchmark_returns)
            performance_metrics = compute_performance_metrics(holdings, holding_returns_df, benchmark_returns)
            correlation_matrix = compute_correlation_matrix(holding_returns_df)
            covariance_matrix = compute_covariance_matrix(holding_returns_df)
            min_var_portfolio = optimize_min_variance(holding_returns_df)
            max_sharpe_portfolio = optimize_max_sharpe(holding_returns_df)
        else:
            print("[warn] no price history available - skipping risk/performance/"
                  "correlation/optimization sections.")

    rebalancing = compute_rebalancing_recommendations(holdings, target_allocations) if target_allocations else []
    cost_estimate = estimate_rebalancing_costs(rebalancing, portfolio_value) if rebalancing else None

    health_score = compute_health_score(diversification, risk_metrics or {},
                                         performance_metrics or {}, exposure_alerts)

    report = {
        "sector_weights": {k: _r(v) for k, v in sector_weights.items()},
        "asset_class_allocation": asset_class_allocation,
        "industry_allocation": industry_allocation,
        "market_cap_allocation": market_cap_allocation,
        "thematic_overlap": theme_weights,
        "hhi_score": concentration["hhi_score"],
        "largest_holding": concentration["largest_holding"],
        "largest_sector": concentration["largest_sector"],
        "diversification_metrics": diversification,
        "exposure_alerts": exposure_alerts,
        "liquidity_flags": liquidity_flags,
        "position_size_flags": position_flags,
        "risk_metrics": risk_metrics,
        "performance_metrics": performance_metrics,
        "correlation_matrix": correlation_matrix,
        "covariance_matrix": covariance_matrix,
        "min_variance_portfolio": min_var_portfolio,
        "max_sharpe_portfolio": max_sharpe_portfolio,
        "rebalancing_recommendations": rebalancing,
        "rebalancing_cost_estimate": cost_estimate,
        "portfolio_health_score": health_score,
    }

    if generate_summary:
        report["ai_summary"] = generate_health_summary(report)

    return report


def main():
    parser = argparse.ArgumentParser(description="Portfolio Health Agent (extended)")
    parser.add_argument("--file", default="data/sample_portfolio.csv")
    parser.add_argument("--basic", action="store_true",
                         help="Run only the original v1 metrics (fast, no network needed)")
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    parser.add_argument("--period", default="1y", help="yfinance period, e.g. 1y, 6mo, 5y")
    parser.add_argument("--targets", help="Path to a JSON file of {ticker: target_weight_pct}")
    parser.add_argument("--no-prices", action="store_true", help="Skip price-history-dependent sections")
    parser.add_argument("--no-market", action="store_true", help="Skip yfinance metadata (industry/cap/volume)")
    parser.add_argument("--summary", action="store_true", help="Generate an AI summary via local Ollama")
    parser.add_argument("--export", choices=["json", "csv", "pdf"], help="Export the report to this format")
    parser.add_argument("--export-path", default="portfolio_health_report")
    args = parser.parse_args()

    if args.basic:
        result = run(args.file)
    else:
        target_allocations = None
        if args.targets:
            with open(args.targets) as f:
                raw_targets = json.load(f)
            target_allocations = {k: v / 100 for k, v in raw_targets.items()}

        result = run_full_analysis(
            args.file,
            target_allocations=target_allocations,
            benchmark=args.benchmark,
            period=args.period,
            fetch_market=not args.no_market,
            fetch_prices=not args.no_prices,
            generate_summary=args.summary,
        )

    print("\n" + "=" * 60)
    print("PORTFOLIO HEALTH RESULT")
    print("=" * 60)
    print(json.dumps(result, indent=2, default=str))

    if args.export:
        path = f"{args.export_path}.{args.export}"
        out_path = export_report(result, path, args.export)
        print(f"\n[export] wrote {out_path}")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"\n[error] Could not find file: {e}")
        sys.exit(1)
