"""
Red-flag detection for the Early Warning Agent -
contrastive-embedding nearest-centroid version, replacing the
keyword-matching placeholder documented in ADR 0007.

Same PLACEHOLDER-SWAP pattern as sentiment_tagger.py: this module's core
swappable contract is still

    redflag_score(text: str) -> float   # 0.0 (routine) .. 1.0 (red-flag)

UPDATE (ADR 0009): ROADMAP.md's Week 6 design always called for two
signals - the contrastive embedding AND a rule-based keyword cross-check
- but ADR 0008's swap fully replaced the keyword signal instead of
keeping it as a second opinion. The keyword logic (same categories/terms
as the original ADR 0007 placeholder, now living in
redflag_detector_keyword_backup.py) is restored here as an independent
signal via check_signal_agreement(). is_red_flag() now flags an article
if EITHER signal clears its own threshold - it no longer relies solely
on the embedding score. See ADR 0009 for the real evaluation numbers on
data/redflag/val_sentences.jsonl.

Member 1's early_warning_agent.py calls check_signal_agreement() (via
is_red_flag() for a plain bool, or directly for the full signal
breakdown) and classify_flag_type() for category labels.

HOW IT WORKS (nearest-centroid, NOT a fine-tuned model - see ADR 0008):
Unlike sentiment_tagger.py, the embedding model here is used AS-IS
(pretrained sentence-transformers/all-MiniLM-L6-v2, no contrastive
fine-tuning). With only ~86 hand-labeled examples, fine-tuning the way
ADR 0006 did (7,752 triplets from Financial PhraseBank) was judged too
small a dataset to fine-tune safely in the time available - see ADR 0008
for the full reasoning. Instead: embed a sample of hand-labeled routine/
red-flag sentences, average each label's embeddings into one centroid,
and classify new text by cosine similarity to the nearest centroid. This
still gives a real, continuous 0-1 similarity score (not a binary
keyword hit), and is a strict upgrade over the keyword placeholder.

Centroid source data: data/redflag/train_triplets.jsonl (built by
scripts/build_redflag_pairs.py from data/redflag/raw_redflag_news.json).
Must exist and be reachable from wherever this module is imported - if
missing, _get_centroids() raises loudly rather than silently falling
back to something else.

Two integration paths if this needs to change again later (same two
paths sentiment_tagger.py documents for its own handoff):
  (a) If given a function `redflag_score(text) -> float` directly ->
      replace the body of this function with a call to theirs.
  (b) If given a fine-tuned checkpoint -> change `_MODEL_NAME` below.
      `SentenceTransformer(path)` loads local checkpoints the same way
      it loads a HF model name, so nothing else in this file changes.

classify_flag_type() is NOT part of this swap contract - it stays
rule-based (imported from the original keyword-based redflag_detector.py
logic), since a 2-cluster (routine vs red-flag) model has no way to
sub-categorize *why* something is a red flag.
"""

import json
import os
from collections import defaultdict

from sentence_transformers import SentenceTransformer, util

_MODEL_NAME = os.environ.get("REDFLAG_MODEL_PATH", "sentence-transformers/all-MiniLM-L6-v2")

_TRIPLETS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..",
    "data", "redflag", "train_triplets.jsonl"
)
_CENTROID_SAMPLE_SIZE = 200  # sentences per label used to build each centroid -
                              # in practice this just takes ALL of them, since
                              # the hand-labeled set is much smaller than 200/label

_REDFLAG_THRESHOLD = 0.5

# category -> keywords, unchanged from the original rule-based module -
# classify_flag_type() stays rule-based regardless of the swap (see docstring)
_REDFLAG_KEYWORDS = {
    "fraud": ["fraud", "scam", "embezzlement", "accounting irregularit"],
    "litigation": ["lawsuit", "litigation", "sued", "sec probe", "investigation", "penalty", "fine imposed"],
    "credit_downgrade": ["downgrade", "downgraded", "rating cut", "credit rating"],
    "management_change": ["resignation", "resigned", "steps down", "stepped down", "ceo exit", "cfo exit", "quits as"],
    "regulatory": ["regulatory action", "banned", "suspended", "non-compliance"],
}

_model = None
_centroids = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def _load_labeled_sentences() -> dict:
    """Pulls labeled sentences back out of train_triplets.jsonl (each row
    has an anchor+label and a negative+label - both are real labeled
    sentences, just packaged as triplets for the training path we chose
    NOT to take yet - see module docstring / ADR 0008)."""
    if not os.path.exists(_TRIPLETS_PATH):
        raise FileNotFoundError(
            f"Could not find {_TRIPLETS_PATH}. This file is required to build "
            f"redflag centroids - run scripts/build_redflag_pairs.py first "
            f"(see ADR 0008 for the full setup)."
        )

    by_label = defaultdict(list)
    with open(_TRIPLETS_PATH) as f:
        for line in f:
            row = json.loads(line)
            by_label[row["anchor_label"]].append(row["anchor"])
            by_label[row["negative_label"]].append(row["negative"])
    return by_label


def _get_centroids() -> dict:
    """Lazy-built once per process, same singleton pattern as _get_model()."""
    global _centroids
    if _centroids is None:
        model = _get_model()
        by_label = _load_labeled_sentences()
        centroids = {}
        for label, texts in by_label.items():
            sample = texts[:_CENTROID_SAMPLE_SIZE]
            embeddings = model.encode(sample, convert_to_tensor=True, show_progress_bar=False)
            centroids[label] = embeddings.mean(dim=0)
        _centroids = centroids
    return _centroids


def redflag_score(text: str) -> float:
    """
    Return a 0.0-1.0 anomaly score for a piece of text (pass in
    "title. description" - see early_warning_agent.py).

    Nearest-centroid cosine similarity to the red_flag centroid, in a
    pretrained (not fine-tuned) embedding space - see ADR 0008 for why
    fine-tuning wasn't attempted with this dataset size, and for
    accuracy numbers on the held-out val set.
    """
    if not text or not text.strip():
        return 0.0

    model = _get_model()
    centroids = _get_centroids()

    emb = model.encode(text, convert_to_tensor=True, show_progress_bar=False)
    redflag_sim = util.cos_sim(emb, centroids["red_flag"]).item()
    routine_sim = util.cos_sim(emb, centroids["routine"]).item()

    # Rescale so the score is a 0-1 "how much closer to red_flag than
    # routine" signal, not a raw cosine value (which sits in a narrow
    # band and isn't directly comparable to the old keyword 0.0/1.0
    # contract). Softmax-style: higher redflag_sim relative to
    # routine_sim -> score closer to 1.0.
    if redflag_sim + routine_sim == 0:
        return 0.5
    score = redflag_sim / (redflag_sim + routine_sim) if (redflag_sim > 0 and routine_sim > 0) else (
        1.0 if redflag_sim > routine_sim else 0.0
    )
    return float(max(0.0, min(1.0, score)))


def keyword_redflag_score(text: str) -> float:
    """
    Rule-based 0.0/1.0 score - same _REDFLAG_KEYWORDS lookup
    classify_flag_type() already uses, so there's exactly one keyword
    list in this file, not a second copy. Restored as an independent
    second opinion alongside redflag_score() per ADR 0009 - ROADMAP.md's
    Week 6 design always called for both signals, not a full replacement.
    """
    if not text or not text.strip():
        return 0.0

    lowered = text.lower()
    for keywords in _REDFLAG_KEYWORDS.values():
        if any(kw in lowered for kw in keywords):
            return 1.0
    return 0.0


def check_signal_agreement(text: str, embedding_threshold: float = _REDFLAG_THRESHOLD,
                            keyword_threshold: float = 1.0) -> dict:
    """
    Runs both signals - contrastive embedding (redflag_score) and
    rule-based keyword (keyword_redflag_score) - and reports whether they
    agree. An article is flagged if EITHER signal independently clears
    its own threshold (see ADR 0009); signal_agreement tells the caller
    how much to trust that call:

        "both"           - embedding AND keyword both fired
        "embedding_only" - only the embedding score cleared threshold
        "keyword_only"   - only a keyword matched
        "neither"        - not flagged by either signal

    early_warning_agent.py uses this (via is_red_flag() for a plain bool,
    or directly for the full breakdown) so downstream severity scoring
    can treat "both" alerts with more confidence than single-signal ones.
    """
    embedding_score = redflag_score(text)
    keyword_score = keyword_redflag_score(text)
    embedding_flagged = embedding_score >= embedding_threshold
    keyword_flagged = keyword_score >= keyword_threshold

    if embedding_flagged and keyword_flagged:
        agreement = "both"
    elif embedding_flagged:
        agreement = "embedding_only"
    elif keyword_flagged:
        agreement = "keyword_only"
    else:
        agreement = "neither"

    return {
        "is_red_flag": embedding_flagged or keyword_flagged,
        "embedding_score": embedding_score,
        "embedding_flagged": embedding_flagged,
        "keyword_flagged": keyword_flagged,
        "signal_agreement": agreement,
    }


def is_red_flag(text: str, threshold: float = _REDFLAG_THRESHOLD) -> bool:
    """Thin wrapper around check_signal_agreement() - flags if EITHER the
    embedding score clears `threshold` OR the keyword rule fires (see
    ADR 0009). Contract (text, threshold) -> bool is unchanged from the
    embedding-only version; callers who only need the boolean don't need
    to change anything."""
    return check_signal_agreement(text, embedding_threshold=threshold)["is_red_flag"]


def classify_flag_type(text: str) -> str | None:
    """
    Which red-flag category a piece of text matches, or None if none
    match. Stays rule-based (see module docstring) - unchanged from the
    original keyword-based redflag_detector.py.
    """
    if not text or not text.strip():
        return None

    lowered = text.lower()
    for category, keywords in _REDFLAG_KEYWORDS.items():
        if any(kw in lowered for kw in keywords):
            return category
    return None