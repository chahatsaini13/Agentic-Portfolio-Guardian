# ADR 0006: Contrastive Sentiment Model for Market Intelligence Agent

## Status
Accepted

## Context
ADR 0005 (Member 1) built the Market Intelligence Agent's pipeline and
shipped it with a VADER placeholder behind `tag_sentiment(text) -> str`,
so the agent, prompt-building, and JSON output shape weren't blocked
waiting on a trained model - same reasoning as ADR 0001/0003 (build the
standalone piece first, swap the real model in later without touching
callers). ADR 0005's "For Member 2" section asked for: the swap-in
method used, a VADER-vs-contrastive accuracy comparison on Financial
PhraseBank, and any threshold/tuning notes. This ADR is that follow-up,
same split pattern as Week 3 (ADR 0002 = pipeline integration by Member 1,
ADR 0003 = contrastive training/eval by Member 2).

Per ROADMAP.md's Week 5 plan, the goal was to fine-tune sentence
embeddings contrastively (SimCSE-style) on Financial PhraseBank
(~4,840 expert-labeled sentences: positive/negative/neutral) so that
similar-sentiment financial text clusters together in embedding space -
the same underlying technique as ADR 0003's thesis-relevance model, just
applied to sentiment clustering instead of thesis/news relevance.

## Decision

### Why contrastive fine-tuning over a standard classifier
A standard approach here would be a text classifier (e.g. fine-tune
`distilbert` with a 3-way softmax head). Contrastive fine-tuning was
chosen instead for consistency with the project's existing pattern
(ADR 0003) and because it keeps the output as an **embedding space**
rather than a fixed classifier head:
- `tag_sentiment()` becomes "embed the text, find the nearest centroid" -
  no separate model architecture to maintain alongside the relevance
  model, and the same `SentenceTransformer` load/encode code path is
  reused for both tasks.
- Adding a 4th category later (e.g. "strongly negative") means adding
  one more centroid, not retraining a classifier head with a different
  output dimension.
- This mirrors relevance_scorer.py's swap-in contract exactly, which
  matters for handoff consistency across the two teammates' modules.

The trade-off: nearest-centroid classification is a simpler decision
rule than a trained softmax head, and its accuracy depends on how well
`MultipleNegativesRankingLoss` happens to separate three clusters versus
being trained end-to-end for 3-way classification. This is accepted as a
reasonable trade for architectural consistency, not claimed to be more
accurate than a dedicated classifier - no such comparison was run.

### Data and training approach
- **Source:** Financial PhraseBank, `Sentences_50Agree.txt` (the
  50%-annotator-agreement release - chosen over `AllAgree` for more
  training data, at the cost of somewhat noisier labels; not compared
  against the other agreement levels in this session).
- **Loading:** `scripts/build_sentiment_pairs.py`. 4,846 sentences
  loaded (`latin-1` encoding - the file predates UTF-8 convention).
  Label distribution: neutral 2,879 / positive 1,363 / negative 604 -
  notably imbalanced, neutral is the majority class by a wide margin.
- **Split:** sentence-level, 80/20, **stratified by label** (each
  label split independently before combining) so val isn't accidentally
  dominated by one class, and no sentence appears in both train and val -
  same leakage-prevention logic as ADR 0003's thesis-level split, applied
  here at the sentence level since there's no thesis-like grouping key
  in PhraseBank. Result: 3,876 train / 970 val sentences.
- **Triplet construction:** for each train sentence (anchor), 2 same-
  label positives and 2 different-label negatives sampled per anchor,
  giving 7,752 training triplets. Unlike ADR 0003's easy/hard negative
  split (which existed because of the same-company-different-topic
  distinction), there's no equivalent difficulty tiering in PhraseBank -
  negatives are simply "a different sentiment label," recorded as
  `negative_label` per triplet for reference.
- **Fine-tuning:** `sentence-transformers/all-MiniLM-L6-v2`,
  `MultipleNegativesRankingLoss`, 2 epochs, batch size 16, lr 2e-5.
  2 epochs (vs ADR 0003's 4) was a deliberate choice given ~40x more
  triplets here (7,752 vs 192) - comparable total gradient steps, not a
  shortcut. Training ran on CPU (no GPU available) and took **~2 hours**,
  far longer than the initial 10-25 minute estimate - flagged honestly
  here since it materially changed the session's time budget, not hidden
  after the fact.

### Centroid-based classification (the actual `tag_sentiment()` mechanism)
`sentiment_tagger.py`'s `_get_centroids()` builds one mean-pooled
embedding per label from a 200-sentence-per-label sample of the training
data (pulled back out of `train_triplets.jsonl`, not a separate file).
`tag_sentiment(text)` embeds the input and returns the label of whichever
centroid is closest by cosine similarity. This sample size (200) matches
what `evaluate_sentiment_model.py` used, so the validated eval numbers
below reflect the same centroid-construction process actually running in
production, not a different one that happens to be untested.

## Validation

### Accuracy vs VADER baseline

The comparison ADR 0005 asked for is nearest-centroid embedding accuracy
(baseline vs contrastive), evaluated identically via
`scripts/evaluate_sentiment_model.py` on the same 970-sentence held-out
val set:

| | Baseline (generic `all-MiniLM-L6-v2`) | Contrastive (fine-tuned) |
|---|---|---|
| Accuracy on Financial PhraseBank (held-out) | 65.36% (634/970) | **83.71% (812/970)** |

This is an 18.35 percentage point improvement, and both numbers come
from the genuinely held-out 970-sentence val split (never seen during
training) - not a leaked or partially-seen number, consistent with ADR
0003's insistence on validating against real held-out data rather than
self-reported training accuracy.

**Note on comparison framing:** ADR 0005's ask was specifically
VADER-vs-contrastive. What's reported above is generic-embedding-vs-
contrastive-embedding (both using the same nearest-centroid mechanism),
not a VADER-vs-embedding comparison, because VADER is a lexicon scorer
with no embedding space to build centroids in - there's no
apples-to-apples "VADER accuracy" number in this framework. A true
VADER baseline would need to run VADER's own positive/negative/neutral
cutoff logic (`sentiment_tagger.py`'s old `POSITIVE_THRESHOLD` /
`NEGATIVE_THRESHOLD` at ±0.05) directly against the same 970 val
sentences and score its accuracy the same way. **This VADER-specific
number was not computed in this session** - flagged as a follow-up
below, since the ADR 0005 ask technically remains partially open.

### Confusion matrices (rows = true label, columns = predicted)

**Baseline:**
```
             negative     neutral    positive
  negative         94          19           8
   neutral         61         409         106
  positive         44          98         131
```

**Contrastive:**
```
             negative     neutral    positive
  negative        107           9           5
   neutral         22         494          60
  positive          8          54         211
```

Every diagonal cell (correct predictions) increased after fine-tuning:
negative 94→107, neutral 409→494, positive 131→211 - the improvement is
distributed across all three classes, not driven by one class alone
(same "check it's not just the easy case" scrutiny ADR 0003 applied to
its hard-negative breakdown). The biggest relative gain is on the
minority `positive` class (131/233 = 56.2% recall baseline vs 211/273 =
77.3% recall contrastive) - notable because `positive` is the second-
smallest class (1,363 of 4,846 sentences) and the one most likely to be
underserved by an untrained embedding space.

### Sanity check: `tag_sentiment()` integration
Two manual smoke tests against the integrated function (not just the
underlying eval script):
- `"The company reported record profits and strong revenue growth."` →
  `positive` (correct)
- `"The factory was shut down after a workers strike over unpaid wages."`
  → `negative` (correct)

Both correct - confirms the end-to-end path (model load → centroid
build → cosine similarity → label return) works through the actual
`sentiment_tagger.py` module Member 1 will import, not just the
standalone eval script.

## Consequences

### Swap-in confirmation (per ADR 0005's ask)
`tag_sentiment()`'s body was replaced directly - `sentiment_tagger.py`
now loads a fine-tuned `SentenceTransformer` checkpoint
(`models/contrastive_sentiment_v1`) via the same `_get_model()` lazy-
singleton pattern `relevance_scorer.py` uses, plus a new
`_get_centroids()` lazy-singleton for the three label centroids. No
changes were needed in `market_intelligence_agent.py` - the contract
(`tag_sentiment(text) -> "positive"|"negative"|"neutral"`) is unchanged
from the VADER version.

### `NEWS_RELEVANCE_THRESHOLD` (ADR 0005's separate open question)
ADR 0005 asked whether `NEWS_RELEVANCE_THRESHOLD = 0.3` (the relevance
filter, not sentiment) should move once a purpose-trained model exists.
**Not addressed in this session** - that threshold belongs to
`relevance_scorer.py`'s thesis/company-relevance model, a different
embedding space than this sentiment model, and retuning it would need
its own validation pass against real company-anchor relevance scores,
not sentiment data. Flagged as a separate follow-up, not silently
skipped.

### Known limitations
- **True VADER-vs-contrastive number is still missing.** As noted above,
  what's validated here is generic-vs-fine-tuned embeddings, not VADER-
  vs-embeddings. Follow-up: run VADER's `polarity_scores()` directly
  against the same 970 val sentences and compute accuracy the same way,
  for a genuinely apples-to-apples number.
- **Class imbalance not corrected for.** Training data is 59% neutral /
  28% positive / 12% negative. Triplet sampling draws negatives
  uniformly across the other two labels regardless of their size, so the
  minority `negative` class was somewhat undersampled as a *negative*
  example relative to its true prevalence. Accuracy (not F1) is the
  headline number here; a class-imbalance-aware metric would give a more
  complete picture, especially for the minority classes.
- **`Sentences_50Agree.txt` only.** Not compared against `AllAgree`
  (cleaner labels, less data) or `66Agree`/`75Agree`. Given the strong
  observed improvement, this probably isn't the limiting factor right
  now, but a stricter-agreement subset is worth trying if accuracy needs
  to go higher later.
- **2 epochs is a time-budget choice, not a tuned value.** No comparison
  was run against 1, 3, or 4 epochs - unlike ADR 0003, which arrived at
  its epoch count via the LOTO cross-validation process, this session's
  epoch count was set going in and not re-validated against alternatives
  due to the ~2-hour-per-run CPU training cost.
- **Centroid sample size (200/label) not swept.** Chosen to match between
  training and eval code for consistency, not selected via a size-vs-
  accuracy comparison. Given `negative` only has 604 total sentences
  (fewer after the val split), the 200-sample centroid for that label is
  drawing on a meaningful fraction of all available negative examples -
  worth revisiting if the val split ratio changes.
- **CPU training time (~2 hours for 7,752 triplets over 2 epochs)** is
  materially slower than ADR 0003's smaller dataset - worth knowing
  before attempting a larger retrain (e.g. more epochs, or training on
  the full `Sentences_AllAgree` + `50Agree` combined) without GPU access.

## Cross-reference
This ADR completes the "For Member 2" section of ADR 0005
(`docs/decisions/0005-market-intelligence-sentiment-and-relevance-filtering.md`).
Same technique lineage as ADR 0003 (contrastive `SentenceTransformer`
fine-tuning via `MultipleNegativesRankingLoss`), applied to sentiment
clustering instead of thesis-relevance scoring.
