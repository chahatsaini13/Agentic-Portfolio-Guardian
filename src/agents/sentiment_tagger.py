"""
Sentiment tagging for the Market Intelligence Agent (Week 5).

This is a PLACEHOLDER implementation using VADER - a rule-based, lexicon
driven sentiment scorer. No model download, no GPU, scores instantly.
Member 2 is separately fine-tuning sentence embeddings contrastively
(SimCSE-style, on Financial PhraseBank) to tag sentiment via nearest-
cluster match in embedding space - when that's ready, swap it in here.
Nothing else in the pipeline should need to change, because the function
signature is the contract:

    tag_sentiment(text: str) -> str   # "positive" | "negative" | "neutral"

Two integration paths for Member 2's real model, once handed over (same
shape as relevance_scorer.py's handoff):
  (a) If they give you a function `tag_sentiment(text) -> str` directly ->
      just replace the body of this function with a call to theirs.
  (b) If they give you a saved model/cluster-centroid file -> change
      `_get_model()` below the same way relevance_scorer.py's
      `_get_model()` swaps in a checkpoint path. Nothing downstream
      changes.
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = None


def _get_analyzer() -> SentimentIntensityAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentIntensityAnalyzer()
    return _analyzer


# VADER's compound score sits in [-1, 1]. These cutoffs are VADER's own
# documented defaults - not tuned for financial text. Expect this to need
# retuning (or full replacement by Member 2's model) once real news is
# checked against Financial PhraseBank labels in Week 9.
POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05


def tag_sentiment(text: str) -> str:
    """
    Return "positive", "negative", or "neutral" for a piece of text
    (pass in "title. description" - see market_intelligence_agent.py).

    Generic lexicon-based sentiment for now, not a financial-domain
    judgment - treat tags as a rough signal, not ground truth.
    """
    if not text or not text.strip():
        return "neutral"

    scores = _get_analyzer().polarity_scores(text)
    compound = scores["compound"]

    if compound >= POSITIVE_THRESHOLD:
        return "positive"
    if compound <= NEGATIVE_THRESHOLD:
        return "negative"
    return "neutral"