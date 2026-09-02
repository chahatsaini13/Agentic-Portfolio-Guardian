"""
Relevance scoring for news filtering

This is a PLACEHOLDER implementation using an off-the-shelf pretrained
sentence-transformer + cosine similarity. Member 2 is separately training
a contrastive embedding model on thesis-relevant vs irrelevant news pairs -
when that's ready, swap it in here (or wherever `score_relevance` is
imported from). Nothing else in the pipeline should need to change,
because the function signature is the contract:

    score_relevance(thesis: str, news_text: str) -> float

Two integration paths for Member 2's real model, once handed over:
  (a) If they give you a function `score_relevance(thesis, news_text) -> float`
      directly -> just replace the body of this function with a call to
      theirs (or re-export theirs under this name).
  (b) If they give you a saved model file (e.g. a fine-tuned
      SentenceTransformer checkpoint) -> change `_MODEL_NAME` below to the
      path of that checkpoint. `SentenceTransformer(path)` loads local
      checkpoints the same way it loads a HF model name, so nothing else
      changes.
"""

from sentence_transformers import SentenceTransformer, util

_MODEL_NAME = "navneet11/contrastive-relevance-v1" 

_model = None  


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def score_relevance(thesis: str, news_text: str) -> float:
    """
    Return a similarity score in roughly [-1, 1] between a thesis and a
    piece of news text (pass in "title. description" - see filter_news.py).

    Higher = more relevant to the thesis. This is a generic embedding
    similarity for now, not a trained relevance judgment - treat scores as
    a rough filter, not ground truth.
    """
    if not news_text or not news_text.strip():
        return 0.0

    model = _get_model()
    embeddings = model.encode([thesis, news_text], convert_to_tensor=True)
    return util.cos_sim(embeddings[0], embeddings[1]).item()