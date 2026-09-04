"""
True VADER baseline accuracy on the sentiment val set - closes the open
item flagged at the end of ADR 0006 ("This VADER-specific number was not
computed in this session").

Runs vaderSentiment's polarity_scores() directly against
data/sentiment/val_sentences.jsonl (the same held-out val set
evaluate_sentiment_model.py uses), applies the same +-0.05 compound-score
cutoff the original VADER placeholder used (POSITIVE_THRESHOLD /
NEGATIVE_THRESHOLD = 0.05 per ADR 0005), and reports accuracy + a
confusion matrix in the same format as evaluate_sentiment_model.py - so
this number is directly comparable to the existing 65.36% (generic
embedding) / 83.71% (contrastive embedding) figures already in ADR 0006,
and gets appended to the SAME eval_results.jsonl file those two runs used.

Usage:
    python scripts/evaluate_vader_sentiment.py
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05


def load_jsonl(path: Path) -> list:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def classify_compound(compound: float) -> str:
    if compound >= POSITIVE_THRESHOLD:
        return "positive"
    elif compound <= NEGATIVE_THRESHOLD:
        return "negative"
    return "neutral"


def main():
    parser = argparse.ArgumentParser(description="VADER baseline accuracy on sentiment val set")
    parser.add_argument("--data-dir", default="data/sentiment")
    parser.add_argument("--val-file", default="val_sentences.jsonl")
    parser.add_argument("--label", default="vader")
    args = parser.parse_args()

    val_path = Path(args.data_dir) / args.val_file
    rows = load_jsonl(val_path)
    print(f"Loaded {len(rows)} val sentences from {val_path}")

    analyzer = SentimentIntensityAnalyzer()

    correct = 0
    confusion = defaultdict(lambda: defaultdict(int))
    for r in rows:
        scores = analyzer.polarity_scores(r["text"])
        pred = classify_compound(scores["compound"])
        true = r["label"]
        confusion[true][pred] += 1
        if pred == true:
            correct += 1

    accuracy = correct / len(rows)
    print(f"\n=== RESULTS ({args.label}) ===")
    print(f"Accuracy: {accuracy:.4f}  ({correct}/{len(rows)})")

    labels = sorted({r["label"] for r in rows})
    print("\nConfusion (rows=true, cols=predicted):")
    print("           " + "  ".join(f"{l:>10s}" for l in labels))
    for true_label in labels:
        row = "  ".join(f"{confusion[true_label][pred_label]:>10d}" for pred_label in labels)
        print(f"{true_label:>10s} {row}")

    # Appended to the SAME file evaluate_sentiment_model.py writes to, so
    # baseline / contrastive / vader all sit in one place for the ADR table.
    results_path = Path(args.data_dir) / "eval_results.jsonl"
    with open(results_path, "a") as f:
        f.write(json.dumps({"label": args.label, "model": "vaderSentiment", "accuracy": accuracy}) + "\n")
    print(f"\nAppended result to {results_path}")


if __name__ == "__main__":
    main()
