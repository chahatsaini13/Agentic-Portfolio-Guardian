# Open Items

Running checklist of known follow-ups already documented inside existing
ADRs, consolidated here so they aren't forgotten before Week 9
(Evaluation), where most of them are meant to be closed. This is a
tracking list, not a full ADR - each item links back to where it was
originally raised.

- [ ] True VADER-vs-contrastive sentiment accuracy comparison (currently only generic-embedding-vs-contrastive-embedding has been measured) — raised in `docs/decisions/0006-contrastive-sentiment-model.md`, targeted: Week 9
- [ ] Leave-one-thesis-out (LOTO) per-fold easy/hard-negative accuracy breakdown (currently only computed for the fixed-split model) — raised in `docs/decisions/0003-contrastive-relevance-model-training-and-evaluation.MD`, targeted: Week 9
- [ ] Downstream LLM hallucination risk on sparse/missing data (fundamentals and news both thin or absent) — no validation/self-check step exists yet — raised in `docs/decisions/0002-embedding-based-news-relevance-filtering.MD` and `docs/decisions/0003-contrastive-relevance-model-training-and-evaluation.MD`, targeted: Week 7 (candidate for a self-reflection pass once the Orchestrator exists)
- [ ] Cross-listing news bleed-through (NSE-listed holding picking up news about a different, ADR/NYSE-listed entity of the same underlying company) — raised in `docs/decisions/0005-market-intelligence-sentiment-and-relevance-filtering.md`, targeted: unscheduled — ask Chahat
- [ ] Red-flag embedding model has never been spot-checked against live, real NewsAPI data end-to-end — only the 17-sentence hand-labeled validation set has been tested (this now also applies to the combined signal_agreement cross-check added in ADR 0009, which the same 17-sentence set could not exercise meaningfully - see ADR 0009's Validation section) — raised in `docs/decisions/0008-contrastive-redflag-detection.md`, targeted: unscheduled — worth doing before the Week 9 full-system evaluation, since it's cheap to run
- [ ] Red-flag detection threshold (currently 0.5) was inherited from the original keyword-placeholder's contract and never re-tuned against the embedding model's own real score distribution the way the Week 3 relevance threshold was (0.35 → 0.28 → 0.5, each backed by observed score bands) — raised in `docs/decisions/0008-contrastive-redflag-detection.md`, targeted: unscheduled