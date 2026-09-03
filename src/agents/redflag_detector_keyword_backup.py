"""
Red-flag detection for the Early Warning Agent -
rule-based placeholder, ahead of Member 2's contrastive version.

Same PLACEHOLDER-SWAP pattern as sentiment_tagger.py: this module's only
public contract is

    redflag_score(text: str) -> float   # 0.0 (routine) .. 1.0 (red-flag)

Member 1's early_warning_agent.py calls exactly this function and
thresholds it via is_red_flag() below - the contract will not change when
Member 2's model lands, only the implementation underneath it.

HOW IT WORKS (placeholder, tonight's version):
Rule-based keyword matching against a small list of red-flag terms
(resignation, litigation, downgrade, fraud, etc., see _REDFLAG_KEYWORDS).
Score is 1.0 if any keyword matches, 0.0 otherwise - a blunt instrument on
purpose, just enough to get the pipeline (detect -> score severity ->
alert) working end-to-end tonight.

HOW IT WILL WORK (Member 2's contrastive version, same as sentiment_tagger.py):
A sentence-transformer model fine-tuned contrastively to separate
"routine business news" and "red-flag news" into distinguishable clusters
(see ROADMAP Week 6). At import time this module will embed labeled
routine/red-flag examples and build one or two centroids the same way
sentiment_tagger.py's _get_centroids() does; redflag_score() will become
cosine similarity to the red-flag centroid (or distance from the routine
centroid) instead of a keyword hit. Everything downstream - is_red_flag(),
early_warning_agent.py - keeps working unchanged because the 0.0-1.0
float contract doesn't change.

Two integration paths if this needs to change again later (same two paths
sentiment_tagger.py documents for its own handoff):
  (a) If given a function `redflag_score(text) -> float` directly ->
      replace the body of this function with a call to theirs.
  (b) If given a saved checkpoint -> add a `_MODEL_NAME` constant here
      (mirroring sentiment_tagger.py's) and lazy-load it in a `_get_model()`
      singleton the same way. Nothing else in this file or in
      early_warning_agent.py changes.

classify_flag_type() below is NOT part of the swap contract - it stays
rule-based even after Member 2's model lands, since the contrastive model
only separates routine vs red-flag, it doesn't sub-categorize *why*
something's a red flag. Category labels are still useful for the alert
output, so this keyword lookup stays as a permanent (non-swapped) helper.
"""

_REDFLAG_THRESHOLD = 0.5  # matches sentiment_tagger.py's DEFAULT_RELEVANCE_THRESHOLD-style
                           # convention of naming the constant instead of hardcoding it inline

# category -> keywords. Order matters for classify_flag_type() - first
# category with a keyword hit wins, so put more specific/severe categories
# first (fraud before generic "management change" wording, etc.)
_REDFLAG_KEYWORDS = {
    "fraud": ["fraud", "scam", "embezzlement", "accounting irregularit"],
    "litigation": ["lawsuit", "litigation", "sued", "sec probe", "investigation", "penalty", "fine imposed"],
    "credit_downgrade": ["downgrade", "downgraded", "rating cut", "credit rating"],
    "management_change": ["resignation", "resigned", "steps down", "stepped down", "ceo exit", "cfo exit", "quits as"],
    "regulatory": ["regulatory action", "banned", "suspended", "non-compliance"],
}


def redflag_score(text: str) -> float:
    """
    Return a 0.0-1.0 anomaly score for a piece of text (pass in
    "title. description" - see early_warning_agent.py).

    Placeholder: 1.0 if any red-flag keyword matches, 0.0 otherwise.
    Member 2's contrastive version replaces the body of this function only
    - see module docstring.
    """
    if not text or not text.strip():
        return 0.0

    lowered = text.lower()
    for keywords in _REDFLAG_KEYWORDS.values():
        if any(kw in lowered for kw in keywords):
            return 1.0
    return 0.0


def is_red_flag(text: str, threshold: float = _REDFLAG_THRESHOLD) -> bool:
    """Thin threshold wrapper around redflag_score() - kept separate so
    early_warning_agent.py can also use the raw score later (e.g. to rank
    alerts) without re-running detection."""
    return redflag_score(text) >= threshold


def classify_flag_type(text: str) -> str | None:
    """
    Which red-flag category a piece of text matches, or None if it isn't
    a red flag at all. Rule-based, stays rule-based after the contrastive
    swap (see module docstring) - the contrastive model doesn't replace
    this function.
    """
    if not text or not text.strip():
        return None

    lowered = text.lower()
    for category, keywords in _REDFLAG_KEYWORDS.items():
        if any(kw in lowered for kw in keywords):
            return category
    return None