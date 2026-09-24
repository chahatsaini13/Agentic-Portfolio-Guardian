"""
Scheduler - Portfolio Guardian

Runs the Orchestrator's full pipeline (orchestrator.run()) on a fixed
interval in the background, independent of any Streamlit session, and
writes the result to disk via cache_store.py. This is what makes the
dashboard's "see current state instantly on page load" behavior possible
- see ADR 0013 for why APScheduler + flat-file caching, not a task queue
or a database, was chosen for this.

Deliberately NOT session-aware: unlike views/common.py's run_analysis()
(which reads portfolio_path/fetch_prices/fetch_market from
st.session_state, since it's triggered by a logged-in user's button
click), this module has no Streamlit session to read from - it runs on
its own clock, possibly with no dashboard open at all. Its config comes
from .env instead (see the SCHEDULE_* vars below), same load_dotenv()
pattern every agent file already uses.

Usage (started once, from views/common.py at app import time - see
start_scheduler_once() below for the guard against Streamlit's re-run
behavior spawning duplicate schedulers):

    from src.scheduler import start_scheduler_once
    start_scheduler_once()
"""

import os
import traceback
import datetime

from dotenv import load_dotenv
from apscheduler.schedulers.background import BackgroundScheduler

from src import orchestrator
from src.cache_store import save_latest_state, append_verdict_history

load_dotenv()

SCHEDULE_INTERVAL_HOURS = float(os.environ.get("SCHEDULE_INTERVAL_HOURS", 4))
SCHEDULE_PORTFOLIO_PATH = os.environ.get("SCHEDULE_PORTFOLIO_PATH", "data/sample_portfolio.csv")
SCHEDULE_FETCH_PRICES = os.environ.get("SCHEDULE_FETCH_PRICES", "true").lower() == "true"
SCHEDULE_FETCH_MARKET = os.environ.get("SCHEDULE_FETCH_MARKET", "true").lower() == "true"
SCHEDULE_RUN_ON_STARTUP = os.environ.get("SCHEDULE_RUN_ON_STARTUP", "true").lower() == "true"

_scheduler = None  # module-level singleton, same lazy-singleton shape as
                    # relevance_scorer.py's _model - guards against
                    # creating more than one BackgroundScheduler per process


def run_scheduled_analysis():
    """One full pipeline run + cache write. This is the job body -
    APScheduler calls this on its own thread, so it must not touch
    st.session_state or any other Streamlit-request-scoped object."""
    print(f"[scheduler] {datetime.datetime.now().isoformat(timespec='seconds')} "
          f"- starting scheduled run on {SCHEDULE_PORTFOLIO_PATH} ...")
    try:
        final_state = orchestrator.run(
            SCHEDULE_PORTFOLIO_PATH,
            limit=None,
            fetch_prices=SCHEDULE_FETCH_PRICES,
            fetch_market=SCHEDULE_FETCH_MARKET,
        )
    except Exception as e:
        # A bad run should never kill the scheduler thread - log it and
        # wait for the next interval, same "one bad ticker/one bad node
        # doesn't take down the rest" philosophy the agents already use.
        print(f"[scheduler] run failed: {e}")
        traceback.print_exc()
        return

    save_latest_state(final_state)
    append_verdict_history(final_state)
    print(f"[scheduler] run complete - "
          f"{len(final_state.get('errors', []))} node-level error(s), "
          f"cache written.")


def start_scheduler_once():
    """Starts the background job exactly once per process. Streamlit
    re-runs the whole script on every interaction, so this is called from
    views/common.py behind a `st.session_state` guard at the call site -
    but the module-level `_scheduler` singleton here is a second line of
    defense in case start_scheduler_once() ever gets called from more
    than one place."""
    global _scheduler
    if _scheduler is not None:
        return _scheduler

    _scheduler = BackgroundScheduler()
    _scheduler.add_job(
        run_scheduled_analysis,
        "interval",
        hours=SCHEDULE_INTERVAL_HOURS,
        id="portfolio_guardian_run",
        next_run_time=datetime.datetime.now() if SCHEDULE_RUN_ON_STARTUP else None,
        coalesce=True,        # if a run overruns the interval, don't queue
                               # up backlog runs - just run once when free
        max_instances=1,      # never run two scheduled analyses at once
    )
    _scheduler.start()
    print(f"[scheduler] started - interval={SCHEDULE_INTERVAL_HOURS}h, "
          f"portfolio={SCHEDULE_PORTFOLIO_PATH}, run_on_startup={SCHEDULE_RUN_ON_STARTUP}")
    return _scheduler