"""
Sentiment tagging for the Market Intelligence Agent (Week 5) -
contrastive embedding version, replacing the VADER placeholder.

Same PLACEHOLDER-SWAP pattern as relevance_scorer.py: this module's only
public contract is

    tag_sentiment(text: str) -> str   # "positive" | "negative" | "neutral"

Member 1's market_intelligence_agent.py calls exactly this function and
does a majority vote keyed on these three exact lowercase strings - the
contract has not changed from the VADER version, only the implementation
underneath it.

HOW IT WORKS (contrastive nearest-centroid, not a classifier head):
A sentence-transformer model (fine-tuned contrastively on Financial
PhraseBank via scripts/train_sentiment_contrastive.py - see ADR 0006)
embeds financial text so that same-sentiment sentences cluster together.
At import time this module embeds a sample of labeled PhraseBank
sentences per label and averages them into three "centroid" vectors
(one per sentiment). To tag new text, embed it and return whichever
centroid it's closest to by cosine similarity. This needs no separate
classifier head or training beyond the embedding fine-tune itself - the
same "swap the model, nothing downstream changes" property relevance_scorer.py
relies on.

Two integration paths if this needs to change again later (same two
paths relevance_scorer.py documents for its own handoff):
  (a) If given a function `tag_sentiment(text) -> str` directly -> just
      replace the body of this function with a call to theirs.
  (b) If given a different saved checkpoint -> change `_MODEL_NAME` below.
      `SentenceTransformer(path)` loads local checkpoints the same way it
      loads a HF model name, so nothing else in this file changes.

Centroid source data: data/sentiment/train_triplets.jsonl (built by
scripts/build_sentiment_pairs.py from Financial PhraseBank). This file
must exist and be reachable from wherever this module is imported - if
it's missing, _get_centroids() raises loudly rather than silently
falling back to something else, since a wrong/empty centroid would
mis-tag every article without any visible error.
"""

import json
import os
from collections import defaultdict

from sentence_transformers import SentenceTransformer, util

_MODEL_NAME = os.environ.get("SENTIMENT_MODEL_PATH", "models/contrastive_sentiment_v1")

_TRIPLETS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..",
    "data", "sentiment", "train_triplets.jsonl"
)
_CENTROID_SAMPLE_SIZE = 200  # sentences per label used to build each centroid -
                              # matches evaluate_sentiment_model.py's build_centroids()
                              # so centroid quality here matches what was validated in ADR 0006

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
    PhraseBank sentences, just packaged as triplets for training)."""
    if not os.path.exists(_TRIPLETS_PATH):
        raise FileNotFoundError(
            f"Could not find {_TRIPLETS_PATH}. This file is required to build "
            f"sentiment centroids - run scripts/build_sentiment_pairs.py first "
            f"(see ADR 0006 for the full setup)."
        )

    by_label = defaultdict(list)
    with open(_TRIPLETS_PATH) as f:
        for line in f:
            row = json.loads(line)
            by_label[row["anchor_label"]].append(row["anchor"])
            by_label[row["negative_label"]].append(row["negative"])
    return by_label


def _get_centroids() -> dict:
    """Lazy-built once per process, same singleton pattern as _get_model().
    One mean-pooled embedding vector per sentiment label."""
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


def tag_sentiment(text: str) -> str:
    """
    Return "positive", "negative", or "neutral" for a piece of text
    (pass in "title. description" - see market_intelligence_agent.py).

    Nearest-centroid classification in a contrastively fine-tuned
    embedding space (see ADR 0006 for training/eval details and accuracy
    vs the VADER baseline). Contract is identical to the VADER version
    this replaces - callers don't need to change anything.
    """
    if not text or not text.strip():
        return "neutral"

    model = _get_model()
    centroids = _get_centroids()

    emb = model.encode(text, convert_to_tensor=True, show_progress_bar=False)
    scores = {label: util.cos_sim(emb, centroid).item() for label, centroid in centroids.items()}
    return max(scores, key=scores.get)
