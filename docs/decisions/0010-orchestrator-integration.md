# ADR 0010: Orchestrator - LangGraph Skeleton and Full 4-Agent Wiring

## Status
Accepted

## Context
Week 7 (per ROADMAP.md) calls for a LangGraph Orchestrator connecting
all 4 standalone agents (Investment Thesis, Portfolio Health, Market
Intelligence, Early Warning) into one shared state graph, merging their
outputs into a single per-portfolio + per-holding insight, with graceful
per-ticker failure handling so one bad ticker or one down service
doesn't crash the whole run.

Following this project's established pattern (ADR 0001: build the
standalone piece first, validate it, then integrate) this was built
incrementally across two sessions: Day 1 built the graph skeleton with
2 of 4 nodes and validated the state-passing/error-handling shape before
adding complexity; Day 2 added the remaining 2 nodes plus a merge step,
and used the first full end-to-end run to surface real integration bugs
before calling Week 7 done.

## Decision

### Shared state and node contract (Day 1)
`src/orchestrator.py` defines `PortfolioGuardianState` (a `TypedDict`):
`holdings`, `theses`, `portfolio_health_result`, `per_holding_results`,
`errors` (Day 2 adds `final_output`, see below). Every node function is
`(state) -> state`, matching LangGraph's expected node signature, and
every per-ticker failure is appended to `state["errors"]` as
`{"node": ..., "ticker": ..., "error": ...}` rather than raised - so
`state["errors"]` doubles as both a debug log and a way to confirm at
the end that a "clean" run genuinely had zero problems, not just no
crash.

### Thesis data: separate file, not a CSV column (Day 1)
The portfolio CSV (`ticker`/`sector`/`weight`/`asset_class`/`theme`) has
no field for thesis text, but `investment_thesis_agent.run(ticker, thesis)`
needs one. Rather than restructure the CSV (scope creep, and every other
agent already depends on its current shape), a separate
`data/theses.json` (`{ticker: thesis_text}`) was created. Theses were
backfilled from `results/week3_log.json`'s logged runs where a ticker
matched; where a ticker had multiple logged entries with different
thesis wording (e.g. `HDFCBANK.NS`, which shows both "will drive
earnings" and "will drive margins" across different Week 3 test runs),
the most recent timestamped entry was used. Three tickers in
`sample_portfolio.csv` (`ICICIBANK.NS`, `MARUTI.NS`, `HINDUNILVR.NS`)
were never logged in Week 3 and were deliberately left without a
thesis rather than fabricating one - `thesis_node`'s existing
no-thesis-on-file handling (see below) was trusted to skip these
cleanly rather than inventing plausible-sounding text.

### Day 1 scope: 2 nodes only, sequential
Day 1 shipped only `portfolio_health_node` and `thesis_node`, wired
`portfolio_health_node -> thesis_node -> END`. `market_intelligence_node`
and `early_warning_node` were deliberately deferred to a second session -
same incremental-validation reasoning as ADR 0001 (isolate which layer a
bug belongs to before adding the next one).

`portfolio_health_node` always analyzes the full portfolio regardless of
the CLI `--limit` flag; `--limit` only truncates the holdings list that
`thesis_node` (and, from Day 2, `market_intelligence_node` /
`early_warning_node`) loops over. This is intentional, not an oversight -
health/diversification/risk metrics computed on an arbitrarily truncated
subset of holdings would be meaningless (e.g. HHI/sector-weight
percentages would no longer sum to anything real), whereas per-ticker
agent calls are naturally fine to sample for a quick test run.

`thesis_node`'s per-ticker loop catches `requests.exceptions.ConnectionError`
(Ollama down) and `ValueError` (invalid/delisted yfinance ticker,
matching `fetch_fundamentals()`'s existing raise) separately, logging
each to `state["errors"]` and `continue`-ing rather than raising - one
bad ticker doesn't stop the loop. A missing thesis (ticker not in
`theses.json`) is treated the same way: logged as an error
(`"no thesis on file"`) and skipped, not a fabricated thesis and not a
crash.

### Day 2 scope: remaining 2 nodes + merge, same pattern
`market_intelligence_node` and `early_warning_node` were added following
`thesis_node`'s exact shape: loop over `state["holdings"]`, wrap the
corresponding agent's `run_for_holding()` call in try/except
(`ConnectionError` plus a broad `Exception` fallback - broader than
`thesis_node`'s, because `fetch_news()` in both
`market_intelligence_agent.py` and its reuse in `early_warning_agent.py`
has no internal try/except around the NewsAPI request, so *any* request
failure needed a catch-all to avoid taking down the whole node), and
append into `state["per_holding_results"][ticker]` under a per-agent key
(`"market_intelligence"`, `"early_warning"`).

`merge_node` was added as a final step, combining
`state["portfolio_health_result"]` with the three per-ticker keys
(`thesis_agent`, `market_intelligence`, `early_warning`) into one
`final_output` dict:
```
{
  "portfolio_health": {...},
  "per_holding": [
    {"ticker", "thesis_status", "thesis_reasoning",
     "market_sentiment", "market_summary", "redflag_alerts"}
  ]
}
```
The graph is wired fully sequentially -
`portfolio_health_node -> thesis_node -> market_intelligence_node ->
early_warning_node -> merge_node -> END` - deliberately not parallelized
this week, to keep the first full 4-agent run debuggable (a failure's
node is unambiguous from log order alone).

## Issues found and fixed (Day 2's first full end-to-end run)

Running all 4 nodes together for the first time on real holdings
(TCS.NS, INFY.NS, HDFCBANK.NS, ICICIBANK.NS from
`data/sample_portfolio.csv`) surfaced three real integration issues -
one genuinely new failure category, and two silent defects from
integration work merged into `main` between the two sessions. Documented
here per this project's practice of recording bugs found during
real-data validation (see ADR 0003, ADR 0004), not just the final
passing state.

### 1. New failure mode: private Hugging Face repo, no local auth (blocking)
`sentiment_tagger.py`'s `_MODEL_NAME` default was pointed at
`navneet11/contrastive-sentiment-v1` (a private HF Hub repo) by an
earlier commit ("Point sentiment model default at hosted HF repo").
Running the orchestrator on a machine without HF credentials for that
repo caused every `tag_sentiment()` call to raise:

```
RepositoryNotFoundError: 401 Client Error ... Invalid username or password.
```

Because `market_intelligence_agent.run_for_holding()` calls
`tag_sentiment()` on every kept article, this didn't just fail sentiment
tagging - it took down the entire `market_intelligence_node` result for
any holding with >=1 relevant article (TCS.NS, HDFCBANK.NS, ICICIBANK.NS
all lost their whole market-intelligence output in the first run; only
INFY.NS survived, coincidentally, because it had 0 kept articles that
run).

This is a **new failure category** not covered by any prior ADR
(previously documented: Ollama down, empty news, delisted ticker,
malformed JSON) - it's a **credentials/access** failure specific to
private model hosting, and it fails at the node level rather than the
per-article level, because the orchestrator's per-ticker try/except
catches it as one `Exception` per ticker rather than a benign "no
sentiment available" case.

**Fix:** Authenticated the local machine against the private HF repo via
`huggingface-cli login` with a personal read-scope access token, after
confirming HF-account-level collaborator access on the repo. Re-ran the
full graph - all 4 holdings' market intelligence output populated
correctly.

**Not fixed, flagged as follow-up:** the orchestrator has no graceful
degradation for this case - a future teammate without HF access (or a
CI environment) will hit the identical failure. Options: (a) document
the `huggingface-cli login` requirement in README's setup section,
(b) add a try/except around `tag_sentiment()` inside
`market_intelligence_agent.py` itself so an auth failure degrades to
"sentiment unavailable" per-article rather than failing the whole node,
mirroring how Ollama connection failures are already handled elsewhere.
Option (b) is preferred for consistency with the rest of the pipeline's
error-handling philosophy but was not implemented this session, since
the immediate priority was finishing the graph wiring.

### 2. Dead `signal_agreement` field - incomplete ADR 0009 integration
ADR 0009 ("Restore keyword cross-check alongside embedding model") added
`check_signal_agreement()` to `redflag_detector.py`, returning a
`signal_agreement` breakdown (`"both"` / `"embedding_only"` /
`"keyword_only"` / `"neither"`). That commit also added a line in
`early_warning_agent.py`'s `run_for_holding()`:

```python
"signal_agreement": article.get("signal_agreement"),
```

But `detect_flags()` - the function that actually builds each `article`
dict - was never updated to call `check_signal_agreement()` or set this
key. It still called the older `is_red_flag(text)` / `redflag_score(text)`
pair directly. Result: every alert's `signal_agreement` silently
evaluated to `None` via `.get()`'s default, with no error and no
warning - a dead field shipping in every alert. This was only caught
because Day 2's task explicitly asked to flag any new failure mode
encountered, and an unexplained `null` field in `merge_node`'s output
prompted tracing it back through `git log` to the ADR 0009 commit.

**Fix:** `detect_flags()` now calls `check_signal_agreement(text)`
directly and populates `redflag_score`, `flag_type`, and
`signal_agreement` from its return value, instead of calling
`is_red_flag()` and `redflag_score()` separately. Re-ran the graph - all
three HDFCBANK.NS alerts now correctly show
`"signal_agreement": "both"` (both the embedding model and the keyword
match fired on the same fraud/litigation headlines).

**Lesson for the team:** when a swappable-contract module (per the
placeholder-swap pattern used throughout this project - `relevance_scorer.py`,
`sentiment_tagger.py`, `redflag_detector.py`) grows a new *additional*
field or function, the ADR introducing it should explicitly confirm the
calling file was updated to actually invoke it, not just that a field
name was threaded through the output dict shape. A field can be wired
into the output shape without ever being computed, and nothing crashes
to reveal it.

### 3. Ollama truncated-JSON severity output (recovered, not a new category)
`early_warning_agent.py`'s severity-scoring step occasionally received a
truncated response from Ollama (missing closing brace/quote on the
`reasoning` field), which failed `json.loads()` and fell through to the
existing MEDIUM-fallback path, dumping the raw text - including the
truncated JSON fragment itself - into the `reasoning` field. This is the
same underlying "Ollama generation length / prompt-length" issue already
noted in ADR 0002 (fixed there for `investment_thesis_agent.py` via an
explicit `num_predict: 500`), not a new failure category - it simply
hadn't been observed in `early_warning_agent.py`'s severity-scoring path
before this integration run.

**Fix:** Added a regex-based recovery step to `parse_severity_output()`:
if `json.loads()` fails, attempt to extract `"severity"` and
`"reasoning"` values directly via regex before falling back to the raw
MEDIUM dump. Both truncated cases observed during this run were
recovered cleanly (`severity: "HIGH"` and `severity: "LOW"`, with clean,
un-truncated reasoning text) instead of shipping a `reasoning` field
containing embedded raw JSON.

**Not fixed, flagged as a cleaner follow-up:** the root cause (Ollama's
default generation length) was addressed for `investment_thesis_agent.py`
in ADR 0002 but was never applied to `early_warning_agent.py`'s
`call_ollama()`. Adding `"options": {"num_predict": 300}` there would
likely prevent truncation at the source rather than recovering from it
after the fact - left as a follow-up, since the regex recovery is a
working stopgap and the priority was finishing the graph wiring.

## Data gap fixed alongside (not a code bug)
`data/theses.json` was missing an entry for `ICICIBANK.NS`, which
`sample_portfolio.csv` includes as a holding. `thesis_node` correctly
skipped it via its existing no-thesis-on-file handling rather than
crashing, but this meant `ICICIBANK.NS` had `thesis_status: null` in
every full-graph run. Added a placeholder thesis ("retail loan growth
and digital banking adoption will drive earnings") so all 4 test
holdings produce complete output. This is a reasonable placeholder for
pipeline-testing purposes, not a personally-vetted investment thesis -
worth revisiting before using this portfolio for anything beyond
integration testing.

## Validation

### Day 1
- **Run 1** (`--limit 3`: TCS.NS, INFY.NS, HDFCBANK.NS - all had a
  thesis on file): `errors: 0`, all three `thesis_agent` results
  populated correctly in `state["per_holding_results"]`.
- **Run 2** (full portfolio, no limit, 8 holdings): 5 had a thesis
  (TCS, INFY, HDFCBANK, TMPV, ITC), 3 did not (ICICIBANK, MARUTI,
  HINDUNILVR) -> `errors: 3`, all three logged as `"no thesis on file"`
  for the expected tickers, no crash.
- HDFCBANK.NS fetched 3 articles, all scoring below the 0.5 relevance
  threshold (0.40 / 0.38 / 0.21) -> 0 kept, thesis verdict based on
  fundamentals alone. Not a bug - consistent with the threshold
  tightness already documented in ADR 0002/0005.

### Day 2
Full 4-node graph (`portfolio_health_node -> thesis_node ->
market_intelligence_node -> early_warning_node -> merge_node`) run
end-to-end via:
```
python -m src.orchestrator --file data/sample_portfolio.csv --limit 4 --no-prices --no-market
```
on 4 real holdings (TCS.NS, INFY.NS, HDFCBANK.NS, ICICIBANK.NS), after
all three fixes above:
- `errors: 0`
- All 4 holdings produced complete `thesis_status`, `market_sentiment`,
  `market_summary` in the merged output.
- HDFCBANK.NS's 3 red-flag alerts all correctly show a populated
  `signal_agreement` value (`"both"` in all 3 cases for this run).
- No malformed/truncated JSON leaked into any `reasoning` field.

## Consequences
- `langgraph` added to `requirements.txt` (Day 1).
- The orchestrator file lives at `src/orchestrator.py`, not
  `src/agents/` - it was initially placed in `src/agents/` and then
  moved, since it's an orchestration layer sitting above the agents, not
  a peer agent module itself.
- `--limit`'s CLI default was initially `3` (silently capping "no limit"
  runs to 3 holdings) - fixed to `default=None` on Day 1.
- The orchestrator is now validated end-to-end across all 4 agents on
  real portfolio data, not just Day 1's 2-node subset.
- Two follow-ups are explicitly left open (not silently dropped):
  graceful HF-auth degradation in `market_intelligence_agent.py`, and
  `num_predict` truncation-prevention in `early_warning_agent.py`'s
  `call_ollama()`. Both are candidates for `OPEN_ITEMS.md`.
- `--no-prices --no-market` flags were used for Day 2's validation run
  to keep it fast; `risk_metrics`, `performance_metrics`, and the
  correlation/optimization sections of `portfolio_health` were not
  exercised in that specific run (they were separately validated in
  `docs/validation/week4_portfolio_health_validation.md`).
- The graph remains fully sequential (not parallelized) by design for
  this week - a candidate for revisiting once the 4-agent pipeline has
  had more real runs and failure patterns are better understood.

## Known limitations
- Day 1's 2-node subset is superseded by Day 2's full 4-node graph, but
  is left in the Decision section above for the record of how the graph
  was incrementally built and validated.
- Fabricated theses were deliberately avoided (`ICICIBANK.NS`'s Day 2
  backfill is explicitly flagged as a placeholder, not a real thesis;
  `MARUTI.NS` and `HINDUNILVR.NS` remain without theses entirely as of
  this ADR).
- Sequential-only graph structure (no parallel node execution) trades
  some runtime for debuggability - acceptable for now, worth revisiting
  once the pipeline is stable.
- No graceful degradation yet for the Hugging Face private-repo auth
  failure mode (Issue 1 above) - a fresh clone of this repo on a machine
  without HF credentials will reproduce the same failure until this is
  addressed.