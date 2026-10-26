# ADR 0014: Scaled Fundamentals-Only Backtest (27 Cases) - Testing the HOLDS-Default Bias

## Status
Accepted

## Context
ADR 0012 ran a 6-case fundamentals-only pilot of thesis-flip detection on
historical NSE data and found that 5 of 6 verdicts were HOLDS regardless of
how negative the fundamentals were, so its "3/5 correct" figure was not
evidence of real discrimination. The ROADMAP Week 9 backtest task asked for
a larger run. This ADR scales the same experiment to 27 cases to test
whether the HOLDS-default bias is real. It is still fundamentals-only (no
historical news archive exists) and uses the same synthetic
momentum-continuation thesis framing as ADR 0012.

## Decision

### What was kept identical to the pilot
Same script (`scripts/backtest_fundamentals_pilot.py`, not modified), same
model (`llama3.2:latest`), same prompt, same 252-day lookback and 126-day
forward window, same GROWTH/DECLINE definition, same scoring rule (HOLDS
correct if GROWTH, BROKEN correct if DECLINE, WEAKENING and parse errors
not scored), same split guard.

### What differed from the pilot (stated, not hidden)
- **Case selection.** 27 cases, 3 per ticker over 9 tickers (TCS, INFY,
  RELIANCE, HDFCBANK, ITC, MARUTI, SUNPHARMA, ADANIENT, YESBANK), chosen
  from `scripts/suggest_noncrash.py`. That helper excludes 2007-07 to
  2009-12 and 2019-08 to 2021-12 (the broad crashes), skips windows with a
  daily close ratio below 0.6 or above 1.6 (a pre-filter, separate from the
  script's own 35% guard, which stayed the final check), keeps candidate
  dates at least 126 days apart, and takes each ticker's largest declines
  and growths. The case list was therefore built from extreme forward
  returns, not sampled at random. 14 DECLINE-type and 13 GROWTH-type
  candidates were run on purpose, so that the bias test had enough decline
  cases.
- **Machine.** Run on a different laptop (macOS, Python 3.9.6, Ollama
  installed via Homebrew) than the pilot (Windows). Ollama and model builds
  may differ; not checked.
- **Parse failures** (see Validation) did not occur in the pilot.

### Two reported views
Because 23 of 26 cases failed the script's JSON parse, two views are
reported, produced by `scripts/summarize_backtest.py` from the same log:
- **STRICT**: verdicts exactly as the unmodified script parsed them. This is
  the primary, as-run result.
- **RECOVERED**: for PARSE_ERROR rows, the verdict is read from the
  `thesis_status` field inside the model's own `llm_reasoning` text. The
  model produced the verdict; only the closing brace was missing. This is a
  post-hoc reading, labelled as such, and was not used to change the script.

## Validation

Log: `results/week9_backtest_scaled_log.json` (27 rows). Run output:
`results/week9_backtest_scaled_run.txt`. Summary:
`results/week9_backtest_scaled_summary.txt`. The pilot log
`results/week9_backtest_log.json` was not touched.

### Run outcome
- 27 cases attempted, 26 with an actual outcome. 1 failed: ITC 1997-11-19
  (no trailing history before that date, 0 days).
- Outcome distribution (26): 14 DECLINE, 12 GROWTH. No FLAT cases, because
  cases were picked from extreme forward returns.
- No case was skipped by the split guard.
- 2 cases ran with a shorter lookback than 252 days: HDFCBANK 1999-02-25
  (234 days) and SUNPHARMA 1999-01-21 (209 days).

### Parse failures
23 of 26 cases returned `PARSE_ERROR` under the unmodified script: the
model's JSON was missing its closing brace. The cause was not investigated.
Under STRICT, only 3 verdicts were usable (1 HOLDS, 2 WEAKENING), and only 1
case was scored (TCS 2015-08-20, HOLDS vs DECLINE, incorrect). STRICT
accuracy is therefore 0/1 and means nothing.

### Verdict distribution next to accuracy (RECOVERED view)

| Verdict | Count (of 26) |
|---|---|
| HOLDS | 18 |
| WEAKENING | 6 |
| BROKEN | 2 |

| Metric | Value |
|---|---|
| Scored cases (HOLDS or BROKEN) | 20 |
| Correct | 10 |
| Accuracy on scored | 0.500 |
| Majority baseline on the same 20 scored cases | 0.500 |
| Always-DECLINE baseline on all 26 | 0.538 |
| Recall GROWTH (verdict HOLDS) | 9/12 |
| Recall DECLINE (verdict BROKEN) | 1/14 |
| DECLINE cases called HOLDS | 9/14 |
| DECLINE cases flagged WEAKENING or BROKEN | 5/14 |

HOLDS was correct 9 times out of 18 (50%). Both BROKEN verdicts: YESBANK
2019-03-26 (DECLINE, correct) and SUNPHARMA 1999-01-21 (GROWTH, incorrect;
its reasoning said the data window was too short to judge, not that the
thesis had broken).

### Per-case table (RECOVERED verdict; values from the log)

| Ticker | Cutoff | Strict | Recovered | Outcome | Fwd return |
|---|---|---|---|---|---|
| TCS | 2011-02-22 | PARSE_ERROR | WEAKENING | DECLINE | -15.1% |
| TCS | 2015-08-20 | HOLDS | HOLDS | DECLINE | -17.5% |
| TCS | 2010-07-01 | PARSE_ERROR | HOLDS | GROWTH | +57.3% |
| INFY | 2011-02-22 | PARSE_ERROR | HOLDS | DECLINE | -28.7% |
| INFY | 2003-07-07 | PARSE_ERROR | HOLDS | GROWTH | +83.3% |
| INFY | 2001-10-03 | PARSE_ERROR | HOLDS | GROWTH | +67.2% |
| RELIANCE | 2011-04-06 | PARSE_ERROR | HOLDS | DECLINE | -20.8% |
| RELIANCE | 2003-05-08 | PARSE_ERROR | WEAKENING | GROWTH | +92.3% |
| RELIANCE | 2007-04-12 | PARSE_ERROR | HOLDS | GROWTH | +90.1% |
| HDFCBANK | 2000-03-23 | PARSE_ERROR | HOLDS | DECLINE | -22.0% |
| HDFCBANK | 2007-05-08 | PARSE_ERROR | HOLDS | GROWTH | +76.3% |
| HDFCBANK | 1999-02-25 | PARSE_ERROR | HOLDS | GROWTH | +80.6% |
| ITC | 2016-06-21 | PARSE_ERROR | HOLDS | DECLINE | -37.1% |
| ITC | 2014-12-08 | PARSE_ERROR | HOLDS | DECLINE | -25.6% |
| ITC | 1997-11-19 | (failed) | (failed) | - | - |
| MARUTI | 2018-07-24 | PARSE_ERROR | WEAKENING | DECLINE | -33.8% |
| MARUTI | 2015-08-19 | PARSE_ERROR | HOLDS | DECLINE | -25.3% |
| MARUTI | 2014-03-04 | PARSE_ERROR | HOLDS | GROWTH | +82.7% |
| SUNPHARMA | 2018-09-06 | WEAKENING | WEAKENING | DECLINE | -32.7% |
| SUNPHARMA | 2005-10-27 | PARSE_ERROR | HOLDS | GROWTH | +64.5% |
| SUNPHARMA | 1999-01-21 | PARSE_ERROR | BROKEN | GROWTH | +88.7% |
| ADANIENT | 2011-07-07 | PARSE_ERROR | HOLDS | DECLINE | -59.9% |
| ADANIENT | 2013-03-11 | WEAKENING | WEAKENING | DECLINE | -43.4% |
| ADANIENT | 2005-11-14 | PARSE_ERROR | HOLDS | GROWTH | +222.0% |
| YESBANK | 2019-03-26 | PARSE_ERROR | BROKEN | DECLINE | -87.4% |
| YESBANK | 2013-02-25 | PARSE_ERROR | HOLDS | DECLINE | -52.9% |
| YESBANK | 2014-02-26 | PARSE_ERROR | WEAKENING | GROWTH | +107.2% |

Worst single example: INFY 2001-10-03 had a trailing return of -71.5% and
a max drawdown of -72.9%, and the model still said HOLDS (the case then
went to GROWTH, so it scored "correct" for the wrong reason).

### Non-determinism observed
A one-case test run of TCS 2011-02-22 returned HOLDS; the same case in the
full run returned WEAKENING. Both outputs were parse errors. The test log
was written outside the repo and is not committed. So a single verdict can
differ between runs of the same input.

### Comparison with the pilot
| | Pilot (ADR 0012) | Scaled (this ADR, recovered) |
|---|---|---|
| Cases with outcome | 6 | 26 |
| HOLDS share | 5/6 (83%) | 18/26 (69%) |
| BROKEN returned | 0 | 2 |
| Recall DECLINE (BROKEN) | 0/3 | 1/14 |
| Accuracy on scored | 0.600 (3/5) | 0.500 (10/20) |
| Majority baseline on scored | 0.600 | 0.500 |
| Parse errors | 0 | 23 |

### Was the HOLDS bias confirmed?
**Largely yes.** On 14 decline cases the model said HOLDS 9 times and BROKEN
once. Its accuracy on scored cases equals the majority baseline, so the
model shows no demonstrated ability to separate declines from growth in this
setup. It is not a pure always-HOLDS default: 8 of 26 verdicts were
WEAKENING or BROKEN, and 5 of 14 declines were flagged. But WEAKENING is not
scored, and those 8 verdicts include cases that went the other way (for
example RELIANCE 2003 and YESBANK 2014, both followed by large growth). The
HOLDS share fell from 83% to 69%, but with 6 pilot cases that difference is
not meaningful.

## Consequences

### Known limitations
- **Fundamentals-only, no news.** The live agent uses news as well. This
  tests only the fundamentals-reasoning half.
- **Only momentum-style features** (trailing return, volatility, drawdown,
  VaR, CVaR, downside deviation), all computed from price. No earnings,
  valuation, or balance-sheet data.
- **Lookahead contamination not tested.** The ticker and as-of date appear
  in the prompt, so a local model could in principle recall what happened
  next to well-known companies (for example YESBANK 2019). No test for this
  was run, so the verdicts, including the correct ones, cannot be assumed to
  come from the features alone.
- **Small sample.** 20 scored cases (RECOVERED) is too few for a precise
  accuracy figure; 0.500 has a wide interval. The case list was built from
  extreme forward returns, which is not representative of ordinary
  periods.
- **Selection by extremes and era clustering.** Many cases fall in 1997-2005
  (dot-com era) or 2010-2019. Broad-market crashes of 2008-09 and 2020 were
  excluded, but market-wide effects (for example the 2011 and 2013 market
  moves) are not removed.
- **One model only** (`llama3.2:latest`). The `llama3.1` re-run flagged in
  ADR 0012 was not done.
- **Parse failures.** 23/26 as-run outputs were unparseable; the headline
  result depends on the RECOVERED view, which is a post-hoc reading. No fix
  was made to the script or prompt in order to keep comparability with the
  pilot.
- **Non-determinism.** Same input gave different verdicts across runs; only
  one run per case was done, so run-to-run variance is not quantified.
- **Short lookbacks** for HDFCBANK 1999 (234 days) and SUNPHARMA 1999 (209
  days); the model's own reasoning for SUNPHARMA 1999 cited the short window.
- **Helper vs script forward returns differ** slightly (for example TCS
  2011-02-22: helper -19.4%, script -15.1%). Not investigated; all figures
  in this ADR are from the log.
- **Different machine and OS** from the pilot (see Decision).
- **Synthetic thesis framing**, same as ADR 0012: no real investor thesis
  exists for any case.

### Follow-up (not blocking)
- Fix or harden JSON parsing (separate from this experiment) and re-run, to
  remove the dependence on the RECOVERED view.
- Repeat the run several times per case to measure variance.
- Run the same cases on `llama3.1` or a larger model.
- Test lookahead contamination, for example with anonymised tickers and
  dates.
- Add features beyond price momentum, and news once an archive is available.
