"""
Evaluates the nearest-centroid redflag detector against the held-out
val_sentences.jsonl (17 sentences never used to build centroids).

For each val sentence: get redflag_score(), threshold at 0.5, compare
against the true label. Same approach as evaluate_sentiment_model.py,
adapted for binary routine/red_flag classification.

Usage:
    python scripts/evaluate_redflag_model.py
"""

import json
from pathlib import Path

import sys
sys.path.insert(0, '.')
from src.agents.redflag_detector_contrastive import redflag_score

VAL_PATH = Path("data/redflag/val_sentences.jsonl")
THRESHOLD = 0.5


def load_jsonl(path: Path) -> list:
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def main():
    val_rows = load_jsonl(VAL_PATH)
    print(f"Evaluating on {len(val_rows)} held-out sentences (threshold={THRESHOLD})\n")

    correct = 0
    results = []
    for r in val_rows:
        text = r["text"]
        true_label = r["label"]
        score = redflag_score(text)
        pred_label = "red_flag" if score >= THRESHOLD else "routine"
        is_correct = pred_label == true_label
        correct += is_correct
        results.append((text, true_label, pred_label, score, is_correct))

    accuracy = correct / len(val_rows)
    print(f"=== ACCURACY: {accuracy:.4f} ({correct}/{len(val_rows)}) ===\n")

    print("Per-sentence results:")
    for text, true_label, pred_label, score, is_correct in sorted(results, key=lambda x: x[3]):
        mark = "OK  " if is_correct else "WRONG"
        print(f"  [{mark}] score={score:.4f}  true={true_label:10s} pred={pred_label:10s}  {text[:60]}")

    # margin check - same spirit as ADR 0003's margin verification
    red_scores = [s for _, l, _, s, _ in results if l == "red_flag"]
    routine_scores = [s for _, l, _, s, _ in results if l == "routine"]
    if red_scores and routine_scores:
        print(f"\nMargin check: lowest red_flag score = {min(red_scores):.4f}, "
              f"highest routine score = {max(routine_scores):.4f}")
        gap = min(red_scores) - max(routine_scores)
        if gap > 0:
            print(f"  -> Clean separation, gap = {gap:.4f}")
        else:
            print(f"  -> WARNING: scores overlap by {-gap:.4f}")


if __name__ == "__main__":
    main()