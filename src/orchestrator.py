"""
Orchestrator Agent - Portfolio Guardian
"""

import argparse
import json
from typing import TypedDict, Optional

import requests
from langgraph.graph import StateGraph, END

from src.agents.portfolio_health_agent import load_portfolio, run_full_analysis
from src.agents.investment_thesis_agent import run as run_thesis_agent

THESES_PATH = "data/theses.json"


class PortfolioGuardianState(TypedDict):
    holdings: list
    theses: dict
    portfolio_health_result: Optional[dict]
    per_holding_results: dict
    errors: list


def load_theses(path: str = THESES_PATH) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"[warn] {path} not found - thesis_node will skip every holding.")
        return {}


def make_portfolio_health_node(filepath: str, fetch_prices: bool = True, fetch_market: bool = True):
    def node(state: PortfolioGuardianState) -> PortfolioGuardianState:
        print("[orchestrator] portfolio_health_node: running full analysis ...")
        try:
            result = run_full_analysis(filepath, fetch_prices=fetch_prices, fetch_market=fetch_market)
        except Exception as e:
            print(f"[orchestrator] portfolio_health_node failed: {e}")
            state["errors"].append({"node": "portfolio_health_node", "ticker": None, "error": str(e)})
            state["portfolio_health_result"] = None
            return state

        state["portfolio_health_result"] = result
        print("[orchestrator] portfolio_health_node: done.")
        return state
    return node


def thesis_node(state: PortfolioGuardianState) -> PortfolioGuardianState:
    print("[orchestrator] thesis_node: running investment thesis agent per holding ...")
    for h in state["holdings"]:
        ticker = h["ticker"]
        thesis = state["theses"].get(ticker)

        if not thesis:
            print(f"  [skip] {ticker}: no thesis found in theses.json")
            state["errors"].append({"node": "thesis_node", "ticker": ticker, "error": "no thesis on file"})
            continue

        print(f"  [running] {ticker} ...")
        try:
            result = run_thesis_agent(ticker, thesis)
        except requests.exceptions.ConnectionError as e:
            print(f"  [error] {ticker}: could not reach Ollama - {e}")
            state["errors"].append({"node": "thesis_node", "ticker": ticker, "error": f"ConnectionError: {e}"})
            continue
        except ValueError as e:
            print(f"  [error] {ticker}: {e}")
            state["errors"].append({"node": "thesis_node", "ticker": ticker, "error": f"ValueError: {e}"})
            continue

        state["per_holding_results"].setdefault(ticker, {})["thesis_agent"] = result

    print("[orchestrator] thesis_node: done.")
    return state


def build_graph(filepath: str, fetch_prices: bool = True, fetch_market: bool = True):
    graph = StateGraph(PortfolioGuardianState)
    graph.add_node("portfolio_health_node", make_portfolio_health_node(filepath, fetch_prices, fetch_market))
    graph.add_node("thesis_node", thesis_node)

    graph.set_entry_point("portfolio_health_node")
    graph.add_edge("portfolio_health_node", "thesis_node")
    graph.add_edge("thesis_node", END)

    return graph.compile()


def run(filepath: str, limit: int = None, fetch_prices: bool = True, fetch_market: bool = True) -> PortfolioGuardianState:
    holdings = load_portfolio(filepath)
    if limit:
        holdings = holdings[:limit]
    theses = load_theses()

    initial_state: PortfolioGuardianState = {
        "holdings": holdings,
        "theses": theses,
        "portfolio_health_result": None,
        "per_holding_results": {},
        "errors": [],
    }

    app = build_graph(filepath, fetch_prices=fetch_prices, fetch_market=fetch_market)
    return app.invoke(initial_state)


def main():
    parser = argparse.ArgumentParser(description="Portfolio Guardian Orchestrator (Week 7)")
    parser.add_argument("--file", default="data/sample_portfolio.csv")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only run thesis_node on the first N holdings. Omit for all holdings.")
    parser.add_argument("--no-prices", action="store_true")
    parser.add_argument("--no-market", action="store_true")
    args = parser.parse_args()

    final_state = run(
        args.file, limit=args.limit,
        fetch_prices=not args.no_prices, fetch_market=not args.no_market,
    )

    print("\n" + "=" * 60)
    print("ORCHESTRATOR RESULT")
    print("=" * 60)
    print(f"portfolio_health_result: {'present' if final_state['portfolio_health_result'] else 'MISSING'}")
    print(f"per_holding_results tickers: {list(final_state['per_holding_results'].keys())}")
    print(f"errors: {len(final_state['errors'])}")
    for e in final_state["errors"]:
        print(f"  - [{e['node']}] {e.get('ticker')}: {e['error']}")

    print("\nFull per_holding_results:")
    print(json.dumps(final_state["per_holding_results"], indent=2, default=str))


if __name__ == "__main__":
    main()