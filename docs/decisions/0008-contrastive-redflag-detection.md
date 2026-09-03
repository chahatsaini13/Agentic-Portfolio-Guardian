# ADR 0008: Nearest-Centroid Embedding Model for Red-Flag Detection

## Status
Accepted

## Context
ADR 0007 shipped the Early Warning Agent with a rule-based keyword
placeholder behind `redflag_score(text) -> float`, so the pipeline
(fetch → detect → score severity → alert) wasn't blocked waiting on a
trained model - same reasoning as ADR 0001/0003/0005 (build the
standalone piece first, swap the real model in later without touching
callers). Per ROADMAP.md's Week 6 plan, the goal was to treat red-flag
detection as anomaly detection: contrastively trained embeddings
separating "routine business news" from "red-flag news" into
distinguishable clusters, same underlying technique as ADR 0003
(thesis-relevance) and ADR 0006 (sentiment), applied here to a
routine/red-flag distinction instead.

Unlike Financial PhraseBank (ADR 0006's ~4,846 pre-labeled sentences),
there is no ready-made labeled dataset for "routine vs red-flag
financial news." This ADR covers building that dataset by hand and
deciding how to use it responsibly given its size.

## Decision

### Why contrastive/embedding framing over a standard classifier
Same reasoning as ADR 0006: framing this as "find the nearest centroid
in an embedding space" rather than training a dedicated binary
classifier head keeps `redflag_score()` on the same `SentenceTransformer`
load/encode code path as `relevance_scorer.py` and `sentiment_tagger.py`.
Adding a third category later (e.g. distinguishing "high-severity" vs
"low-severity" red flags) means adding one more centroid, not retraining
a classifier with a different output shape. This also naturally gives a
continuous 0.0-1.0 score instead of a binary keyword hit, which is what
`redflag_score()`'s contract (inherited from ADR 0007) expects.

### Data: hand-labeled, not fine-tuned
- **Source:** `data/redflag/raw_redflag_news.json`, hand-labeled by
  Member 2 - 86 sentences total (44 routine / 42 red_flag), a mix of
  real financial-news style headlines and constructed examples, covering
  the five red-flag categories `classify_flag_type()` already
  distinguishes (fraud, litigation, credit_downgrade,
  management_change, regulatory), plus assorted routine business news
  (earnings, product launches, partnerships, expansions).
- **Split:** `scripts/build_redflag_pairs.py`, sentence-level,
  stratified by label, 80/20 → 69 train / 17 val. Same
  leakage-prevention logic as ADR 0003/0006: no sentence appears in both
  splits, and centroids are built only from the 69 train sentences.
- **Triplets:** 138 (anchor, positive, negative) triplets built from the
  69 train sentences, 2 positives sampled per anchor - same shape as
  `build_sentiment_pairs.py`, just binary labels instead of 3.

### Decision: nearest-centroid on a pretrained model, NOT a fine-tune
ADR 0006's contrastive fine-tune used 7,752 triplets from 3,876 unique
sentences. This dataset has 138 triplets from 69 unique sentences -
roughly **56x fewer unique training sentences**. Attempting
`MultipleNegativesRankingLoss` fine-tuning (as `train_sentiment_contrastive.py`
does) on a set this small was judged likely to overfit badly and not
achievable safely within the 3-hour session budget once training,
evaluation, and documentation are accounted for.

**Instead:** `redflag_detector.py` uses `sentence-transformers/all-MiniLM-L6-v2`
**as-is, with no fine-tuning**, and builds two centroids (routine,
red_flag) by averaging embeddings of the 69 train sentences per label -
same nearest-centroid mechanism as `sentiment_tagger.py`, just without
the fine-tuning step underneath it. This is a genuine upgrade over the
ADR 0007 keyword placeholder (continuous similarity score vs a binary
keyword hit, and generalizes to phrasing that doesn't contain an exact
keyword), but is explicitly **not** claimed to be as robust as ADR 0003's
or ADR 0006's fine-tuned models, which had 40-100x more unique training
sentences to work with.

Full contrastive fine-tuning (mirroring `train_sentiment_contrastive.py`)
is flagged as a **documented follow-up**, once the hand-labeled dataset
is grown well beyond ~86 examples (ideally several hundred per label,
closer to what ADR 0006 had).

### Score contract
`redflag_score(text)` returns a 0.0-1.0 value: cosine similarity to the
red_flag centroid, rescaled against the routine centroid's similarity so
the output stays in the [0,1] range the ADR 0007 keyword version already
used (`redflag_sim / (redflag_sim + routine_sim)`), rather than returning
a raw cosine value in a narrow band. `is_red_flag()` (threshold=0.5) and
`classify_flag_type()` (unchanged, rule-based) keep their exact
signatures - `early_warning_agent.py` needed no changes.

## Validation

Evaluated via `scripts/evaluate_redflag_model.py` against the 17
genuinely held-out val sentences (never used to build centroids),
threshold=0.5:

| Metric | Result |
|---|---|
| Accuracy | **100% (17/17)** |
| Lowest red_flag score | 0.6450 |
| Highest routine score | 0.4413 |
| Margin (gap) | **0.2037**, zero overlap |

Every routine sentence scored under 0.45; every red_flag sentence scored
above 0.64 - a clean separation with no borderline misses, similar in
spirit to the margin verification ADR 0003 did for its own model.

### Honest caveat on the 100% number
17 held-out sentences is a **very small evaluation set**. A perfect
score here is a genuinely positive signal (same margin-based sanity
check ADR 0003 applied to its own 20-pair 100% result), but it is
**not** a statistically robust guarantee the way a few-hundred-sentence
eval set would be. This is flagged explicitly rather than presented as
a validated production-grade accuracy number.

### Integration check
Post-swap, `redflag_detector.py`'s `redflag_score()`, `is_red_flag()`,
and `classify_flag_type()` were called directly (bypassing
`early_warning_agent.py`'s NewsAPI fetch, for a quick manual check):


Confirms the swap-in worked with no changes needed in
`early_warning_agent.py`, same guarantee ADR 0007 promised for this
handoff.

## Consequences

### Swap-in confirmation
`redflag_score()`'s body was replaced directly with the nearest-centroid
implementation, following the same `_get_model()` / lazy-centroid-
singleton pattern as `sentiment_tagger.py`. The original keyword-based
implementation is preserved at
`src/agents/redflag_detector_keyword_backup.py` for reference/rollback.
`classify_flag_type()` is unchanged (still rule-based, by design - see
Decision above).

### Known limitations
- **Small hand-labeled dataset (86 sentences).** No PhraseBank-equivalent
  ground truth exists for this task; all labels are self-assigned by
  Member 2, not independently verified by a second annotator or expert
  agreement process (unlike PhraseBank's `50Agree` annotation scheme).
  Some labels may reflect the labeler's own judgment calls on ambiguous
  cases.
- **No fine-tuning performed** - this is a pretrained-embedding
  nearest-centroid placeholder, not a contrastively fine-tuned model
  like ADR 0003/0006 produced. Flagged as a follow-up once the dataset
  grows.
- **17-sentence val set is small.** 100% accuracy is a positive signal,
  not a statistically robust guarantee - see Validation section above.
- **Category imbalance within red_flag not analyzed.** Unlike ADR 0006's
  confusion-matrix breakdown across 3 sentiment classes, this ADR does
  not break down accuracy by the five `classify_flag_type()` categories
  (fraud/litigation/credit_downgrade/management_change/regulatory) -
  the binary routine/red_flag distinction was the only thing evaluated.
  Flagged as follow-up work, not blocking.
- **Real-world spot check not yet run.** Unlike ADR 0003, which
  confirmed its model against live NewsAPI data post-integration, this
  model has only been validated against the hand-labeled val set - a
  live end-to-end run through `early_warning_agent.py` on real news is
  a natural next validation step.
- **Threshold (0.5) not tuned** - inherited from ADR 0007's placeholder
  value, not re-derived from this model's own score distribution the
  way ADR 0002's threshold was tuned from real score bands. The clean
  0.2037 margin observed suggests 0.5 is reasonable, but this wasn't a
  deliberate sweep.

## Cross-reference
Same technique lineage as ADR 0003 (contrastive `SentenceTransformer`
fine-tuning) and ADR 0006 (nearest-centroid classification pattern),
applied here to a hand-labeled routine/red-flag distinction instead of
thesis-relevance or PhraseBank sentiment - but deliberately stops short
of fine-tuning given dataset size, unlike those two ADRs.
