# ADR 0013: Background scheduling + disk cache, replacing on-demand-only runs

## Status
Accepted

## Context
Every prior week (ADR 0001–0012) built the pipeline to run strictly
on-demand: a user clicks "Run Full Analysis" in the dashboard, and
`views/common.py`'s `run_analysis()` blocks synchronously while
`orchestrator.run()` walks all 4 agents - 3 of which make Ollama calls,
one per holding each. For an 8-12 holding portfolio this is a multi-minute
wait, and the result lives only in `st.session_state`, so it's gone the
moment the session ends or the app restarts.

This directly contradicts the product's own stated premise (README.md:
"continuously re-checks the original reasoning... as news, fundamentals,
and market conditions change") - nothing in the system actually runs on
its own; every "continuous" claim was aspirational until this ADR.

Separately, Week 9's backtest work (ADR 0012) surfaced a related gap: no
mechanism exists to observe how often or how fast a thesis verdict
actually changes over time - the backtest was a one-shot historical
replay, not a record of live verdict drift. A verdict-history log,
building naturally out of periodic runs, is needed for that metric.

## Decision

### Scheduling: APScheduler `BackgroundScheduler`, not a task queue
Chosen over Celery/RQ/cron because the whole system already runs as one
Python process (Streamlit) with no existing message broker or worker
infrastructure - adding one would be disproportionate to a BTP timeline.
`BackgroundScheduler` runs the job on a thread inside the same process
Streamlit is already running, with a fixed interval read from
`.env` (`SCHEDULE_INTERVAL_HOURS`, default 4h) rather than hardcoded,
matching every other agent's `.env`-driven config pattern
(`OLLAMA_MODEL`, `NEWSAPI_KEY`, etc.).

`coalesce=True` + `max_instances=1` on the job: if a run overruns the
interval (plausible, given multi-minute runs), the next trigger doesn't
queue a backlog of runs - it skips ahead and runs once when the previous
run finishes. Avoids pipeline runs pilin up if Ollama or NewsAPI is slow
on a given day.

### Cache: flat JSON files, not SQLite
`src/cache_store.py` writes two things:
- `results/latest_state.json` - the most recent run, overwritten every
  time. This is what the dashboard reads on page load now
  (`views/common.py`'s new `init_state_from_cache()`), instead of the
  page starting blank until a manual run completes.
- `results/verdict_history/<ticker>.json` - one file per ticker, appended
  (never overwritten) every scheduled run, `{timestamp, verdict,
  key_changes}` per entry.

Chose flat JSON over SQLite for consistency with the project's existing
logging pattern (`results/week3_log.json`, `results/week9_backtest_log.json`
both already do exactly this - append/overwrite plain JSON, no DB
dependency). A real database is flagged as a future improvement once
verdict-history volume or dashboard query needs outgrow "load the whole
file into memory," which isn't the case yet at BTP scale (single
portfolio, handful of tickers).

### Session-vs-background split
`run_analysis()` (`views/common.py`) is untouched - it's still what the
manual "Run Full Analysis" button calls, still reads
`st.session_state`-scoped config (`portfolio_path`, `fetch_prices`,
`fetch_market`), still blocks with the existing spinner. The scheduler
does NOT call `run_analysis()` - it calls `orchestrator.run()` directly
from `src/scheduler.py`'s own job function, with its own `.env`-sourced
config (`SCHEDULE_PORTFOLIO_PATH`, `SCHEDULE_FETCH_PRICES`,
`SCHEDULE_FETCH_MARKET`), because APScheduler's background thread has no
Streamlit request context - touching `st.session_state` from it would be
undefined behavior at best.

This means manual runs and scheduled runs can currently point at
different portfolio files/settings if `.env` and the dashboard's Settings
page disagree - flagged as a known limitation below, not silently
assumed away.

### Duplicate-scheduler guard
Streamlit reruns the entire script on every interaction, so
`start_scheduler_once()` is called from `app.py` on every rerun. Two
guards prevent duplicate `BackgroundScheduler` instances:
1. `st.session_state["scheduler_started"]` in `views/common.py` - cheap
   per-session check, avoids re-entering the function on every rerun.
2. A module-level singleton (`_scheduler`, same lazy-singleton shape as
   `relevance_scorer.py`'s `_model`) inside `src/scheduler.py` itself -
   the real guard, since Streamlit can also spawn genuinely new sessions
   (new browser tabs) that would each pass guard #1 independently.

### `key_changes` gap in `merge_node`
While wiring `append_verdict_history()`, found that `orchestrator.py`'s
`merge_node` drops `key_changes` when building `final_output["per_holding"]`
- only `thesis_status` and `thesis_reasoning` are carried through, even
though `investment_thesis_agent.py`'s raw output includes `key_changes`.
Not fixed here (out of scope - touching `orchestrator.py` was explicitly
excluded from this task) - worked around by having
`append_verdict_history()` read `key_changes` directly from
`final_state["per_holding_results"][ticker]["thesis_agent"]` (the raw,
pre-merge data, still present on the state dict) instead of from
`final_output`. Flagged here so it isn't rediscovered as a fresh bug
later; a real fix belongs in `merge_node` whenever orchestrator.py is
next touched.

## Consequences

### What's faster
- Dashboard page load is now instant on a warm cache (reads one JSON
  file) instead of blocking on a live run.
- Independent-node parallelism (Portfolio Health / Market Intelligence /
  Early Warning don't depend on each other's output, per `orchestrator.py`'s
  own node wiring) and fetch-level TTL caching were both scoped as
  candidates for this task but **not implemented in this pass** - this
  ADR covers scheduling + caching only. Flagged as follow-up work, not
  silently dropped.

### Known limitations
- **Single-process, non-persistent schedule.** The `BackgroundScheduler`
  lives in the Streamlit process's memory - a server restart resets the
  schedule (next run fires per `SCHEDULE_RUN_ON_STARTUP`, not from where
  it left off). Acceptable for a BTP demo environment where the process
  isn't expected to run unattended for long stretches; would need
  `APScheduler`'s persistent job store (e.g. SQLAlchemy) for a real
  deployment.
- **Scheduled vs. manual run config can diverge**, as noted above -
  `.env`'s `SCHEDULE_PORTFOLIO_PATH` and the dashboard's Settings-page
  portfolio path are two independent values with no validation that they
  match.
- **No lock against a scheduled run and a manual run overlapping.**
  `max_instances=1` only guards against two *scheduled* runs overlapping
  each other - if a user clicks "Run Full Analysis" while a scheduled run
  is mid-flight, both run concurrently, each writing to
  `st.session_state`/cache independently. Not observed as a problem in
  testing, but not explicitly guarded against either.
- **`merge_node`'s `key_changes` gap** (above) is a pre-existing bug this
  ADR worked around, not fixed.