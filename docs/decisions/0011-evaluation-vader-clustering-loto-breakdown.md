# ADR 0011: Evaluation - VADER Baseline, Clustering Quality, and LOTO Negative-Type Breakdown

## Status
Accepted

## Context
Three follow-ups from `OPEN_ITEMS.md` targeted for Week 9 are closed by
this ADR:
1. True VADER-vs-contrastive sentiment accuracy comparison (ADR 0006 had
   only compared generic-embedding vs contrastive-embedding, and
   explicitly flagged the VADER-specific number as missing).
2. LOTO per-fold easy/hard-negative accuracy breakdown for the relevance
   model (ADR 0003 had only computed this for the fixed-split model, not
   per LOTO fold).
3. Contrastive embedding clustering quality (silhouette/purity) vs the
   generic-embedding baseline, on the same sentiment val set.

## Decision / Method
- **VADER baseline:** `scripts/evaluate_vader_sentiment.py`, run against
  the same 970-sentence `data/sentiment/val_sentences.jsonl` ADR 0006
  used, same +/-0.05 compound-score cutoff ADR 0005's original VADER
  placeholder used.
- **Clustering quality:** `scripts/evaluate_sentiment_clustering.py`, run
  once for the generic baseline model and once for the fine-tuned
  contrastive model, on the same val set. Reports silhouette score
  against true labels, silhouette against unsupervised KMeans clusters,
  and purity from KMeans (k=3).
- **LOTO negative-type breakdown:** new companion script
  `scripts/evaluate_loto_negative_breakdown.py`, which reuses
  `train_contrastive_model.py`'s `load_triplets()` / `to_examples()` /
  `make_evaluator()` unchanged. Trains each of the 8 LOTO folds the same
  way `run_loto()` already does (4 epochs, batch size 8, lr 2e-5,
  `MultipleNegativesRankingLoss`), and additionally evaluates each fold's
  trained model separately on its held-out easy-negative triplets and
  hard-negative triplets (12 of each per fold), not just the blended
  24-triplet val set `run_loto()` already reports.

## Validation

### 1. Sentiment accuracy - all three methods, same held-out 970 sentences
| Method | Accuracy |
|---|---|
| VADER (lexicon, +/-0.05 cutoff) | **52.99% (514/970)** |
| Generic embedding (nearest-centroid, untrained) | 65.36% (634/970) |
| Contrastive embedding (fine-tuned, nearest-centroid) | 83.71% (812/970) |

This is the genuinely apples-to-apples VADER comparison ADR 0005 asked
for and ADR 0006 flagged as still missing.

**Honest, surprising result:** VADER scores *lower* than the untrained
generic embedding baseline (52.99% vs 65.36%), not higher as might be
assumed for a purpose-built sentiment tool. This is plausible, not a
bug: VADER is a general-purpose lexicon scorer tuned on social-media-style
text (it responds strongly to emphasis, punctuation, emoji, and common
sentiment words), whereas Financial PhraseBank sentences use flat,
domain-specific financial language ("profit fell", "shares gained") that
doesn't trip VADER's lexicon the way informal text does - and
PhraseBank's `neutral` class is the majority class (59%) but genuinely
neutral-toned financial statements are exactly the kind of text VADER's
+/-0.05 cutoff tends to misclassify as positive/negative on subtle wording.
Not independently root-caused further in this session (e.g. no
per-class VADER confusion matrix was built), but the result is reported
as-is rather than only presenting the two more favorable numbers.

### 2. Embedding clustering quality
| Metric | Baseline (generic) | Contrastive (fine-tuned) |
|---|---|---|
| Silhouette (vs true labels) | -0.0058 | **0.3133** |
| Silhouette (vs KMeans clusters) | 0.0681 | **0.4525** |
| Purity (KMeans k=3 vs true labels) | 0.5938 | **0.8258** |

Clustering quality improves in the same direction as accuracy, and by a
similarly large margin. The baseline's near-zero silhouette against true
labels (-0.0058) indicates the three sentiment classes are essentially
**not separated at all** in the untrained embedding space - consistent
with its weaker 65.36% classification accuracy. The contrastive model's
silhouette (0.3133) is a positive, meaningful improvement, though still
well below 1.0 - the three classes are separated but not sharply, which
matches the classification result of 83.71% (strong, not perfect).
Purity (0.8258 vs 0.5938) tells the same story from the unsupervised-
clustering angle: KMeans on the contrastive embeddings recovers something
much closer to the true 3-way sentiment split than KMeans on the
baseline embeddings does.

### 3. LOTO per-fold easy/hard-negative breakdown (relevance model)
| Held-out thesis | n_easy | n_hard | Combined acc | Easy acc | Hard acc |
|---|---|---|---|---|---|
| t1_tatamotors_ev | 12 | 12 | 1.0000 | 1.0000 | 1.0000 |
| t2_reliance_jio | 12 | 12 | 1.0000 | 1.0000 | 1.0000 |
| t3_infosys_ai | 12 | 12 | 1.0000 | 1.0000 | 1.0000 |
| t4_hdfcbank_retail | 12 | 12 | 1.0000 | 1.0000 | 1.0000 |
| t5_sunpharma_generics | 12 | 12 | 1.0000 | 1.0000 | 1.0000 |
| t6_adanigreen_renewables | 12 | 12 | 0.9583 | 1.0000 | **0.9167** |
| t7_itc_fmcg | 12 | 12 | 0.8750 | 1.0000 | **0.7500** |
| t8_dlf_realestate | 12 | 12 | 0.9167 | 1.0000 | **0.8333** |

**Average across 8 folds:**
- Combined: **0.96875** (matches ADR 0003's previously-reported 96.88%
  blended LOTO average almost exactly - good cross-check consistency
  between this new breakdown and the original number).
- Easy negatives: **1.0000**
- Hard negatives: **0.9375**

**Honest read - this is the most important finding in this ADR.** The
blended 96.88% figure ADR 0003 originally reported obscures a real
pattern: **easy negatives are perfect (100%) in every single fold**,
while **hard negatives are perfect in only 5 of 8 folds** and
noticeably weaker in the other 3 (t6: 91.7%, t7: 75.0%, t8: 83.3%).
`t7_itc_fmcg` is the clear outlier at 75% hard-negative accuracy - a 1-in-
4 miss rate on exactly the case (same company, thesis-irrelevant news)
that motivated contrastive training over keyword matching in the first
place (ADR 0003's original stated goal).

This does not overturn ADR 0003's overall conclusion - the average hard-
negative accuracy (93.75%) is still strong, and the blended number was
not wrong, just incomplete. But presenting only the blended 96.88% average
(as ADR 0003 did, for lack of this breakdown at the time) would have hidden
that roughly 3 of 8 theses have a meaningfully higher hard-negative miss
rate than the headline number suggests. With only 12 hard-negative
triplets per fold, a single miss moves that fold's hard accuracy by ~8.3
percentage points, so these per-fold numbers should be read as indicative
of a real pattern (weakness concentrated in specific folds, not spread
evenly) rather than statistically precise estimates on their own.

## Consequences
- `OPEN_ITEMS.md`'s three Week-9-targeted items above are now closed -
  updated to check them off and link here.
- **New, more specific follow-up surfaced by this ADR** (not previously
  known): hard-negative accuracy is not uniform across theses - three
  folds (`t6_adanigreen_renewables`, `t7_itc_fmcg`, `t8_dlf_realestate`)
  show meaningfully weaker hard-negative separation than the other five.
  Worth investigating whether this is a property of those specific
  theses' hard-negative examples (e.g. genuinely harder/more ambiguous
  same-company news for these sectors), a training-data quality issue
  specific to those theses, or an artifact of small-sample noise (12
  triplets per fold). Added to `OPEN_ITEMS.md` as a new item.
- The VADER-lower-than-baseline result is unexpected but plausible and
  reported honestly rather than omitted; a natural follow-up (not done
  in this session) would be a per-class VADER confusion matrix to
  confirm whether the `neutral` class specifically drives the gap.
- No code in `redflag_detector.py`, `sentiment_tagger.py`, or
  `relevance_scorer.py` was changed by this ADR - this is a pure
  evaluation/documentation pass, closing measurement gaps flagged in
  prior ADRs, not changing any deployed model or threshold.

## Cross-reference
Closes open items from `docs/decisions/0003-contrastive-relevance-model-training-and-evaluation.MD`
and `docs/decisions/0006-contrastive-sentiment-model.md`. Companion
script: `scripts/evaluate_loto_negative_breakdown.py`. Raw per-fold data:
`data/contrastive/loto_negative_breakdown.json`.
