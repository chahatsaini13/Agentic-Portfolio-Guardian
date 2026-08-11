"""
News filtering by thesis relevance (Week 3).

Replaces the Week 2 keyword-only filter with embedding-similarity
filtering. Deliberately kept as its own module (not stuffed into
investment_thesis_agent.py) so this is the ONE place that changes when
Member 2's real contrastive model is ready - just edit relevance_scorer.py,
this file and its callers don't need to know the difference.
"""

from src.agents.relevance_scorer import score_relevance

# Cosine similarity from a generic (non-fine-tuned) sentence embedding model
# tends to sit in a fairly narrow, mediocre-looking band even for genuinely
# relevant pairs - 0.35 is a starting point from eyeballing a few runs, not
# a principled number. Expect to retune this once you see real output
# tonight, and again once Member 2's model swaps in (a contrastively
# trained model should give better score separation, so the threshold may
# need to move).
DEFAULT_RELEVANCE_THRESHOLD = 0.5


def _article_text(article: dict) -> str:
    """Concatenate the fields worth embedding for a single article."""
    title = article.get("title") or ""
    description = article.get("description") or ""
    return f"{title}. {description}".strip()


def filter_news_by_relevance(
    thesis: str,
    news: list,
    threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
) -> list:
    """
    Score every article's relevance to `thesis` and keep only the ones
    scoring at or above `threshold`. Returns articles sorted by relevance
    (highest first), each annotated with its `relevance_score` so it's
    visible in the prompt and easy to sanity-check.
    """
    scored = []
    for article in news:
        text = _article_text(article)
        score = score_relevance(thesis, text)
        scored.append({**article, "relevance_score": round(score, 4)})

    # debug: print every article's score, kept or not, so threshold tuning
    # isn't a guessing game - remove or gate behind a flag once 0.35 (or
    # whatever replaces it) is actually validated
    scored_sorted = sorted(scored, key=lambda a: a["relevance_score"], reverse=True)
    print(f"  [debug] relevance scores for {len(scored_sorted)} article(s) (threshold={threshold}):")
    for a in scored_sorted:
        mark = "KEEP" if a["relevance_score"] >= threshold else "drop"
        title = (a.get("title") or "")[:60]
        print(f"    [{mark}] {a['relevance_score']:.4f}  {title}")

    kept = [a for a in scored if a["relevance_score"] >= threshold]
    kept.sort(key=lambda a: a["relevance_score"], reverse=True)
    return kept