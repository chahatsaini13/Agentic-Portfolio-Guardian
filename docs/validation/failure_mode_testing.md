# Day 4 Failure-Mode Testing Note — Orchestrator Resilience

**Date:** 2026-09-09
**Validated by:** Chahat
**Environment:** Local Windows machine, Streamlit `app.py` → `orchestrator.run()`,
full 4-agent + merge run via "Run Full Analysis", `data/sample_portfolio.csv` (8 holdings)

## Scope
Three deliberate failure-injection tests run ahead of Thursday's demo checkpoint,
to confirm the orchestrator degrades gracefully rather than crashing when a
dependency is unavailable. All three were exercised through Streamlit's "Run
Full Analysis" button (the full graph — all 4 agents + `merge_node`), not via
individual agent CLI runs.

## Methodology caveat — read before trusting exact strings below
For all three tests, only the Streamlit UI was observed directly — no
terminal/console output was captured or reviewed. Wherever this note states
an exact error string or log line, that string was reconstructed by reading
the relevant agent/orchestrator code path, **not** confirmed by watching it
actually print. This is flagged explicitly rather than presented as verified,
consistent with this repo's validate-before-documenting practice (see ADR
philosophy across 0001–0008). What **is** confirmed directly is the
UI-visible behavior — banners, counts, populated vs. empty sections — in each
test below.

## Test 1 — Ollama stopped
**What was tested:** Ollama process killed, then "Run Full Analysis" clicked
(all 8 holdings, all 4 agents).

**Confirmed (UI):**
- "8 node error(s) — showing partial results" banner
- Thesis Monitor: 0 Holds / 0 Weakening / 0 Broken (all N/A)
- Portfolio Health: fully populated (score 75.0, diversification, allocation
  donut, CAGR/Sharpe, etc.) — unaffected
- Market Pulse: still worked (2 Positive, 6 Neutral)
- Early Warning: alerts still surfaced on INFY.NS and HDFCBANK.NS

**Not directly observed, inferred from code:** the 8 errors are expected to
come entirely from `thesis_node` — one `ConnectionError`-shaped entry per
holding (8 holdings = 8 errors), matching the banner count. Market
Intelligence's and Early Warning's Ollama calls are each wrapped in their own
try/except (`"Summary unavailable - could not reach local Ollama server."`,
severity `"MEDIUM"`) and caught **inside** the agent itself, so those
failures never reach the orchestrator's `errors` list. This is why the
banner shows 8, not 24 (8 holdings × 3 Ollama-calling agents) — only
`thesis_node` has no internal try/except around its Ollama call.

**Verdict:** No code change needed. Working as designed. Portfolio Health is
unaffected because it doesn't depend on Ollama for its numeric analysis
(only for the optional AI summary paragraph, not exercised in this run).

**Open note:** none of this was confirmed against real console output — only
inferred from code plus the UI counts. Worth keeping the terminal open during
tomorrow's cold-start rehearsal to confirm directly before the actual demo.

## Test 2 — Missing thesis for a valid ticker
**What was tested:** Not a delisted/invalid-ticker case — MARUTI.NS is a
fully valid, active ticker. Its thesis entry was deliberately deleted from
`data/theses.json`, forcing `thesis_node`'s `if not thesis: skip` path for a
holding that otherwise exists correctly in the portfolio.

**Confirmed (UI):** No crash. MARUTI.NS's thesis status showed as "N/A" in
Thesis Monitor; the other 7 holdings ran completely normally.

**Not directly observed, inferred from code:** an error-list entry shaped
like `{"node": "thesis_node", "ticker": "MARUTI.NS", "error": "no thesis on file"}`.

**Verdict:** No code change needed. Working as designed — a missing thesis
for one holding degrades to a clean per-ticker skip, not a pipeline-wide
failure.

## Test 3 — NewsAPI key unset
**What was tested:** Key unset, not invalid — `NEWSAPI_KEY` in `.env`
renamed to `NEWSAPI_KEY_DISABLED`, so `os.environ.get("NEWSAPI_KEY")` returns
`None`. Full orchestrator run, covering all three news-dependent agents
(Investment Thesis, Market Intelligence, Early Warning) in one pass.

**Confirmed (UI):**
- 0 node errors — no partial-results banner at all
- Market Pulse: Neutral 8 / Positive 0 / Negative 0 across all holdings
- AI Synthesis: "No red-flag alerts on any holding"
- Thesis Monitor: still produced verdicts (2 Broken, 6 Weakening) — reasoning
  fell back to fundamentals only, with no news context

**Not directly observed, inferred from code:** each agent's `fetch_news()`
prints `"[warn] NEWSAPI_KEY not set - skipping news..."` and returns an empty
list rather than raising, so no exception ever reaches the orchestrator —
consistent with 0 recorded errors.

**Verdict:** No code change needed. Working as designed — every
news-dependent agent has an explicit early-return when the key is missing,
so the whole pipeline completes with fundamentals-only reasoning instead of
failing.

## Summary
All three failure scenarios degrade gracefully — no code changes required
for any of them. Two different failure *shapes* were observed: Test 1
surfaces as a visible partial-results banner because `thesis_node` has no
internal try/except around its Ollama call, while Tests 2 and 3 are fully
silent (0 errors) because those failure modes are caught inside the relevant
function before they'd ever raise. This inconsistency isn't a bug — the
Market Intelligence/Early Warning Ollama fallback and the NewsAPI
early-return are both intentional, already-documented design choices — but
it's worth knowing going into the demo that a partial-results banner will
only ever appear for `thesis_node` failures, not for the other two agents'
internal fallbacks.

## Known gap
None of the three tests' underlying error/log text was verified against
real console output — only against the Streamlit UI and a code
read-through. Recommend keeping the terminal open during tomorrow's
rehearsal run so this can be confirmed directly, rather than relying on
inferred behavior for the final demo.