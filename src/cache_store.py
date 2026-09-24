"""
Cache Store - Portfolio Guardian

Persists the Orchestrator's latest full-portfolio result to disk, and
appends a per-ticker verdict-history log every time a run completes.

Two files, same "log everything as plain JSON, fail loudly not silently"
philosophy as investment_thesis_agent.py's log_result():

  results/latest_state.json
      The single most recent orchestrator run - overwritten every run.
      {
        "final_output": <final_state["final_output"]>,
        "errors": <final_state["errors"]>,
        "last_updated": "<ISO timestamp>"
      }
      This is what the dashboard reads on page load instead of triggering
      a fresh run - see views/common.py's load_cached_state().

  results/verdict_history/<ticker>.json
      One file per ticker, each a JSON list that grows by one entry per
      scheduled run (never overwritten) - same append-not-replace pattern
      as investment_thesis_agent.py's log_result(), just split per ticker
      instead of one shared file, since this is what feeds the
      verdict-stability/drift metric for the BTP report.
      [
        {"timestamp": "...", "verdict": "HOLDS"|"WEAKENING"|"BROKEN"|null,
         "key_changes": [...]},
        ...
      ]

Both are read/written directly from the Orchestrator's PortfolioGuardianState
dict (the exact thing orchestrator.run() returns) - nothing here touches
orchestrator.py or any agent file.
"""

import json
import os
import datetime

LATEST_STATE_PATH = "results/latest_state.json"
VERDICT_HISTORY_DIR = "results/verdict_history"


def _now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def save_latest_state(final_state: dict, path: str = LATEST_STATE_PATH) -> None:
    """Overwrite the single 'current state' snapshot. Called after every
    scheduled run (and after a manual 'Run now') so the dashboard always
    has something to read without re-running the pipeline."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "final_output": final_state.get("final_output"),
        "errors": final_state.get("errors", []),
        "last_updated": _now_iso(),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def load_latest_state(path: str = LATEST_STATE_PATH):
    """Returns the cached payload, or None if nothing has run yet - the
    dashboard should show an empty/"no data yet" state in that case, not
    crash."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _ticker_history_path(ticker: str, directory: str = VERDICT_HISTORY_DIR) -> str:
    # tickers are already filesystem-safe (e.g. "TCS.NS") but guard
    # against anything unexpected sneaking in via a stray "/"
    safe = ticker.replace("/", "_")
    return os.path.join(directory, f"{safe}.json")


def append_verdict_history(final_state: dict, directory: str = VERDICT_HISTORY_DIR) -> None:
    """One entry per ticker per run, appended (never overwritten) - this
    is the raw log the verdict-stability/drift metric gets computed from
    later. Pulls key_changes from the raw per-holding thesis_agent output
    (final_state['per_holding_results']), since merge_node's final_output
    doesn't carry key_changes through - see ADR 0013."""
    os.makedirs(directory, exist_ok=True)
    timestamp = _now_iso()

    final_output = final_state.get("final_output") or {}
    per_holding = {h["ticker"]: h for h in final_output.get("per_holding", [])}
    per_holding_raw = final_state.get("per_holding_results", {})

    for h in final_state.get("holdings", []):
        ticker = h["ticker"]
        merged = per_holding.get(ticker, {})
        raw_thesis = per_holding_raw.get(ticker, {}).get("thesis_agent") or {}

        entry = {
            "timestamp": timestamp,
            "verdict": merged.get("thesis_status"),
            "key_changes": raw_thesis.get("key_changes", []),
        }

        path = _ticker_history_path(ticker, directory)
        if os.path.exists(path):
            with open(path) as f:
                history = json.load(f)
        else:
            history = []
        history.append(entry)
        with open(path, "w") as f:
            json.dump(history, f, indent=2, default=str)


def load_verdict_history(ticker: str, directory: str = VERDICT_HISTORY_DIR) -> list:
    """Full history for one ticker, oldest first. Empty list if the
    ticker has never been run yet."""
    path = _ticker_history_path(ticker, directory)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)