# ADR 0009: Restore keyword cross-check alongside the contrastive red-flag model

## Status
Accepted

## Context
`ROADMAP.md`'s Week 6 pipeline design explicitly called for two
independent signals: "Embed news using contrastively trained encoder →
Flag news that falls in the 'red-flag' cluster (anomaly signal) →
**Cross-check against rule-based keyword detection** → Score
severity/urgency using LLM." A repo audit found that ADR 0008's swap did
not implement this - `redflag_score()` was fully replaced by the
nearest-centroid embedding model, and the original keyword logic only
survived inside `classify_flag_type()`, which assigns a category label to
an already-flagged article rather than independently deciding whether it
*is* a red flag. So "combine two signals for a more robust decision"
(the roadmap's design) had quietly become "replace one signal with
another" (what was actually built) - a real mismatch between design and
implementation, not a documentation typo.

## Decision
Restored the keyword signal as a genuine second opinion in
`src/agents/redflag_detector.py`, without touching the embedding model
(which stays the validated, permanent piece per ADR 0008):

- **`keyword_redflag_score(text) -> float`** - a 0.0/1.0 rule-based score,
  reusing the exact same `_REDFLAG_KEYWORDS` lookup `classify_flag_type()`
  already relies on (one keyword list in the file, not a duplicated
  second copy sourced from `redflag_detector_keyword_backup.py`).
- **`check_signal_agreement(text) -> dict`** - runs both `redflag_score()`
  (embedding) and `keyword_redflag_score()` (keyword) and returns
  `{is_red_flag, embedding_score, embedding_flagged, keyword_flagged,
  signal_agreement}`, where `signal_agreement` is one of `"both"`,
  `"embedding_only"`, `"keyword_only"`, `"neither"`. An article is
  flagged (`is_red_flag=True`) if **either** signal independently clears
  its own threshold - this is a genuine OR, not a replacement of one
  signal by the other.
- **`is_red_flag(text, threshold)`** - unchanged call signature, now a
  thin wrapper around `check_signal_agreement()`.
- **`early_warning_agent.py`'s `detect_flags()`** now calls
  `check_signal_agreement()` and carries `signal_agreement` through to
  each flagged article, and the final per-holding alert output
  (`run_for_holding()`) includes `signal_agreement` alongside
  `flag_type`, `severity`, and `reasoning` - so a future severity-scoring
  step (or a human reviewing alerts) can treat `"both"` alerts with more
  confidence than single-signal ones.

## Validation
Ran `scripts/evaluate_redflag_signal_agreement.py` against the same
17-sentence held-out `data/redflag/val_sentences.jsonl` ADR 0008 used,
so the embedding-only and combined numbers are directly comparable -
same data, same threshold (0.5), only the detection logic changed.

| Metric | Result |
|---|---|
| Accuracy, embedding-only (same logic as ADR 0008) | 17/17 (100%) |
| Accuracy, combined (embedding OR keyword) | 17/17 (100%) |
| Signal agreement: `both` | 8 |
| Signal agreement: `embedding_only` | 0 |
| Signal agreement: `keyword_only` | 0 |
| Signal agreement: `neither` | 9 |
| Predictions changed by adding the keyword cross-check | **0** |

### Honest read of this result
On this particular 17-sentence val set, the keyword cross-check made
**zero difference** to the outcome - every genuinely red-flag sentence
was caught by both signals (`"both"`, 8/8), and every routine sentence
was correctly left unflagged by both (`"neither"`, 9/9). No sentence in
this val set produced an `embedding_only` or `keyword_only` case, so the
disagreement scenario the cross-check exists to catch was never
exercised here.

This is **not evidence that the cross-check is redundant in general** -
it's evidence that this specific 17-sentence sample happens to be
"easy" for both signals in the same direction. The hand-labeled dataset
(86 sentences total, per ADR 0008) skews toward clear-cut examples by
construction; real NewsAPI articles are noisier and more likely to
produce genuine `embedding_only`/`keyword_only` disagreement (e.g. an
article using red-flag vocabulary the keyword list doesn't cover, or
one using a listed keyword in a routine context). The cross-check is
kept because it matches the roadmap's original design intent and because
it's essentially free (no retraining, no extra model calls), not because
this evaluation proves it improves accuracy - it doesn't, on this data.
This is flagged as a follow-up in `OPEN_ITEMS.md`: spot-check the
combined logic against live NewsAPI output (same open item ADR 0008
already flagged for the embedding model alone) to see whether real news
produces `embedding_only`/`keyword_only` cases this val set didn't.

## Consequences
- `is_red_flag()`'s and `redflag_score()`'s existing callers are
  unaffected - both keep their original signatures and behavior when
  called directly. Only `is_red_flag()`'s internal logic changed (OR
  instead of embedding-only), which is a strict widening: nothing that
  was previously flagged stops being flagged, but articles that only
  match a keyword and score below 0.5 on the embedding are now also
  caught, whereas before they would have been silently missed.
- No known regression risk from the OR-widening on this val set (0/17
  flips), but on unseen data at higher volume, expect the keyword signal
  to occasionally catch cases the embedding model misses at the margin
  (score just under 0.5) - this is the intended benefit and matches
  ADR 0007's keyword categories.
- `redflag_detector_keyword_backup.py` is unchanged and still exists as
  a standalone reference/rollback copy, but is no longer the *only*
  place the keyword logic lives - `keyword_redflag_score()` in
  `redflag_detector.py` is now the live copy used by the cross-check.
- **Known limitation, same as noted above**: the 17-sentence val set
  cannot demonstrate the cross-check's value because it never exercises
  disagreement between the two signals. A live-data spot check (open
  item, see `OPEN_ITEMS.md`) is the natural next validation step.

## Cross-reference
Restores the Week 6 design intent from `ROADMAP.md` that ADR 0008's swap
had unintentionally dropped. Builds on ADR 0007 (original keyword
placeholder + categories) and ADR 0008 (contrastive embedding model,
which remains unchanged here).