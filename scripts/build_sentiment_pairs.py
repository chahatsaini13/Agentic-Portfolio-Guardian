"""
Loads Financial PhraseBank and builds contrastive triplets for sentiment
fine-tuning: (anchor, positive, negative) where anchor+positive share a
sentiment label and negative is from a different label - same shape as
build_triplets.py's thesis/news triplets, adapted for 3-class sentiment
instead of binary relevance.

Expected input format (standard Financial PhraseBank release):
    <sentence text>@<label>
one per line, label in {positive, negative, neutral}, encoding latin-1.

If your file is a different format (e.g. CSV with sentence,label columns),
tell Claude and this loader gets a 2-line fix - don't hand-edit blindly.

Usage:
    python build_sentiment_pairs.py --file data/sentiment/Sentences_50Agree.txt
"""

import argparse
import json
import random
from pathlib import Path
from collections import defaultdict

random.seed(42)

OUT_DIR = Path("data/sentiment")
VAL_FRACTION = 0.2   # held-out sentences for eval, split BEFORE pair-building
                      # (so no sentence appears in both train and val - same
                      # leakage-prevention logic as ADR 0003's thesis-level split)


def load_phrasebank(path: str) -> list:
    """Returns list of {text, label} dicts. label in positive/negative/neutral."""
    rows = []
    with open(path, encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if not line or "@" not in line:
                continue
            text, label = line.rsplit("@", 1)
            rows.append({"text": text.strip(), "label": label.strip().lower()})
    return rows


def split_train_val(rows: list, val_fraction: float = VAL_FRACTION) -> tuple:
    """Split at the SENTENCE level, stratified by label so val isn't
    accidentally all-neutral or similar. This is the eval set for Step 3
    (contrastive vs baseline) - it must never touch training."""
    by_label = defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r)

    train, val = [], []
    for label, items in by_label.items():
        random.shuffle(items)
        n_val = max(1, round(len(items) * val_fraction))
        val.extend(items[:n_val])
        train.extend(items[n_val:])

    print(f"[split] train={len(train)}  val={len(val)}  "
          f"(per label: { {k: len(v) for k, v in by_label.items()} })")
    return train, val


def build_triplets(rows: list, triplets_per_anchor: int = 2) -> list:
    """For each sentence (anchor), sample `triplets_per_anchor` positives
    (same label) and `triplets_per_anchor` negatives (different label),
    same 'mix the difficulty spectrum' idea as build_triplets.py, except
    here there's no easy/hard distinction (PhraseBank has no company-name
    dimension) - negative_type is always 'cross_sentiment' for consistency
    with the existing triplet schema Member's training script expects."""
    by_label = defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r["text"])

    labels = list(by_label.keys())
    triplets = []
    for r in rows:
        anchor_text = r["text"]
        anchor_label = r["label"]

        same_label_pool = [t for t in by_label[anchor_label] if t != anchor_text]
        other_labels = [l for l in labels if l != anchor_label]

        if not same_label_pool or not other_labels:
            continue

        positives = random.sample(same_label_pool, k=min(triplets_per_anchor, len(same_label_pool)))
        for pos in positives:
            neg_label = random.choice(other_labels)
            neg = random.choice(by_label[neg_label])
            triplets.append({
                "anchor": anchor_text,
                "positive": pos,
                "negative": neg,
                "anchor_label": anchor_label,
                "negative_label": neg_label,
            })
    return triplets


def write_jsonl(path: Path, rows: list):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Build sentiment contrastive triplets")
    parser.add_argument("--file", required=True, help="Path to Financial PhraseBank file")
    parser.add_argument("--triplets-per-anchor", type=int, default=2)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_phrasebank(args.file)
    print(f"Loaded {len(rows)} sentences")
    label_counts = defaultdict(int)
    for r in rows:
        label_counts[r["label"]] += 1
    print(f"Label distribution: {dict(label_counts)}")

    train_rows, val_rows = split_train_val(rows)

    train_triplets = build_triplets(train_rows, args.triplets_per_anchor)
    write_jsonl(OUT_DIR / "train_triplets.jsonl", train_triplets)
    print(f"Wrote {len(train_triplets)} train triplets -> {OUT_DIR/'train_triplets.jsonl'}")

    # val_rows saved as flat sentences (not triplets) - Step 3 needs
    # (text, label) pairs to evaluate nearest-centroid accuracy, not triplets
    write_jsonl(OUT_DIR / "val_sentences.jsonl", val_rows)
    print(f"Wrote {len(val_rows)} val sentences -> {OUT_DIR/'val_sentences.jsonl'}")


if __name__ == "__main__":
    main()
