# ADR 0003: Contrastive Relevance Model — Training, Hosting, and Evaluation Methodology

## Status
Accepted

## Context
ADR 0002 established embedding-similarity filtering as a replacement for
keyword-based news filtering, using a generic off-the-shelf sentence
embedding model (`all-MiniLM-L6-v2`) as a placeholder. That model was
never trained on our specific task — it has no notion of "thesis
relevance," only general semantic similarity.

The placeholder's known weakness was **hard negatives**: news about the
same company but unrelated to the specific investment thesis (e.g. "Tata
Motors appoints new regional sales head" for an EV-adoption thesis).
Keyword matching fails on these because the company name matches; a
generic embedding model does somewhat better but has no reason to have
specifically learned this distinction.

Member 2 undertook fine-tuning a contrastive embedding model on our own
thesis/news data to specifically learn this distinction, and hosting the
result so it could be swapped into the existing `relevance_scorer.py`
interface with no changes to the rest of the pipeline.

## Decision

### Data and training approach
- Built `data/contrastive/raw_theses_news.json`: 8 investment theses, each
  with relevant news, easy negatives (unrelated company/topic), and hard
  negatives (same company, thesis-irrelevant).
- `scripts/build_triplets.py` converts this into `(anchor, positive,
  negative)` triplets for contrastive training, with a `negative_type`
  field (`easy`/`hard`) carried through so evaluation can later separate
  the two.
- Split strategy: thesis-level splitting (never row-level) so no thesis
  appears in both train and val — prevents the model from "recognizing" a
  thesis it saw in training rather than genuinely generalizing.
  - `fixed` mode: one 6-theses-train / 2-theses-val split.
  - `loto` mode: leave-one-thesis-out, 8 folds, each holding out exactly
    one thesis — added specifically because with only 8 theses total, a
    single fixed split's val set (2 theses) is too small to trust on its
    own.
- Fine-tuned `sentence-transformers/all-MiniLM-L6-v2` using
  `MultipleNegativesRankingLoss`, 4 epochs.

### Hosting: Hugging Face Hub (private repo)
Chose Hugging Face Hub over committing the checkpoint to GitHub or sharing
via cloud storage:
- The checkpoint (~90MB+) is too large to commit cleanly to a Git repo
  without LFS overhead.
- HF Hub gives `SentenceTransformer("org/model-name")` loading with no
  code change from loading a local path — same interface either way.
- Given a stated interest in potentially commercializing this component,
  the repo (`navneet11/contrastive-relevance-v1`) should be kept **private**,
  not public — a public repo would let anyone download and reuse the
  trained weights freely, which conflicts with commercialization if that
  path is pursued later.

## Consequences (including two evaluation bugs found and fixed)

Consistent with this project's ADR philosophy (see ADR 0001, 0002), this
section documents the real iteration — including two evaluation mistakes
that were caught before being trusted, not just the final numbers.

### Bug 1 — misleading val_cosine_accuracy on the deployable checkpoint
The training script trains 8 LOTO folds for a generalization estimate,
then separately retrains one final "deployable" model on **all** 192
triplets (144 train + 48 val combined) — a reasonable choice, since you
want the best possible model for actual deployment, not one that
deliberately withholds 25% of available data. However, the first version
of this script also computed a "validation accuracy" for that final model
using a small slice of the *same* 192 triplets it had just trained on —
i.e. it evaluated the model on data it had already seen. The model card
correctly showed `training samples: 192` (matching 144+48, not a
leakage bug in the literal sense) but reported a `val_cosine_accuracy:
1.0`, which was uninterpretable, since there was no genuinely held-out
data left for that specific model. Fixed by removing the fake evaluation
step from the final-model training path entirely — the final deployable
model's quality is understood via the LOTO average, not a self-reported
number.

### Bug 2 — eval_pairs.jsonl evaluated on non-held-out theses
`build_eval_pairs()` builds evaluation pairs from all 8 theses by design
(it's meant as a general-purpose eval set, reused across different
models/splits). When the *fixed-split* model (trained on 6 theses, with
`t4_hdfcbank_retail` and `t5_sunpharma_generics` held out) was evaluated
against the full `eval_pairs.jsonl`, ~75% of that evaluation set (6 of 8
theses) was actually data the model had already seen in training. The
resulting 96.25% accuracy was inflated and not a fair generalization
estimate.

Fixed by adding `--val-thesis-ids` to `evaluate_relevance_model.py`,
restricting evaluation to only the thesis_ids genuinely held out from a
given model's training. Re-evaluating on just the correct 2 held-out
theses (20 pairs) gave 100% accuracy — the same headline number as
before, but this time trustworthy, because the sample is confirmed
unseen.

### Verification: is the 100% figure believable?
A perfect score on a small sample (20 pairs) is a legitimate reason for
suspicion on its own, independent of any leakage. To check this wasn't
luck, raw cosine similarity scores were inspected directly rather than
just the pass/fail count:
- Lowest score among genuinely relevant pairs: **0.6097**
- Highest score among genuinely irrelevant pairs: **0.3755**
- Clean gap of **0.2342** with zero overlap between the two groups.
- Critically, hard-negative pairs (same company, irrelevant news) scored
  consistently in the 0.24–0.38 range — clearly separated from relevant
  pairs (0.61–0.72) — confirming the model learned to distinguish
  same-company-irrelevant from genuinely-relevant, not just
  different-company-irrelevant (which even the untrained placeholder
  could mostly do).

This is the evidence that matters most for this project: the whole
motivation for contrastive training over keyword matching was the hard
negative case, and the held-out data shows the model handling it
correctly, not just producing a suspiciously clean number.

### Final numbers (for reference in reports)

| Number | Status | Use it? |
|---|---|---|
| Fixed-split accuracy computed on full eval_pairs.jsonl (all 8 theses) — 96.25% | Leaked, ~75% seen data | Do not quote |
| Final deployable model's self-reported val_cosine_accuracy — 1.0 | Meaningless (no genuine held-out data for this model) | Do not quote |
| LOTO average across 8 folds, combined easy+hard negatives — 96.88% | Real, but blends easy and hard cases together | Usable as a broad estimate, caveat the blending |
| Fixed-split model, filtered to its true 2 held-out theses, margin-verified — 100% (20 pairs) | Real and verified | Primary number to report |

### Real-world spot check (post-integration)
After swapping the model into `relevance_scorer.py`, live runs against
NewsAPI data reinforced the same pattern seen in evaluation. For an
Infosys AI-services thesis, a news item about the company's employee
time-tracking system — same company, but irrelevant to the thesis — was
correctly filtered out with a relevance score in the 0.27 range, while a
genuinely on-thesis article about AI revenue contribution was kept with a
score around 0.55–0.59. This is the same hard-negative separation
observed in the held-out evaluation set, now confirmed on real,
previously-unseen news data rather than only the curated dataset.

### Known limitations (documented honestly, not hidden)
- **Small dataset.** Only 8 theses total; the fixed-split training set
  draws from just 6 unique anchor sentences. 100% accuracy on 20 held-out
  pairs is a strong positive signal, not a statistically robust guarantee
  — a larger evaluation set would be needed for that.
- **LOTO's easy/hard breakdown is not yet computed per-fold** — only the
  fixed-split model's held-out evaluation has been broken out by negative
  type so far. This is flagged as follow-up work, not blocking.
- **Downstream hallucination risk remains** (documented in ADR 0002):
  when both fundamentals and news fetches return little data, the
  downstream LLM (in `investment_thesis_agent.py`) produces confident,
  specific-sounding claims not grounded in any retrieved data (observed
  directly during Week 3 integration testing on a delisted ticker). This
  is a downstream agent-design issue, not a relevance-model issue, but is
  noted here since it was surfaced during the same testing pass.

## Integration
`relevance_scorer.py`'s `_MODEL_NAME` was updated from `"all-MiniLM-L6-v2"`
to `"navneet11/contrastive-relevance-v1"` — no other file needed to
change, confirming the swap-point design from ADR 0002 worked as intended.
`DEFAULT_RELEVANCE_THRESHOLD` in `news_filter.py` was retuned from `0.28`
(tuned for the untrained placeholder's narrower score distribution) to
`0.5`, based on the verified 0.24–0.72 score range above.
