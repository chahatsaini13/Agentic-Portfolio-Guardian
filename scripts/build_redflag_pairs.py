"""
Builds contrastive triplets for red-flag detection from hand-labeled
routine vs red-flag news examples (data/redflag/raw_redflag_news.json).

Same shape/logic as scripts/build_sentiment_pairs.py, adapted for a
binary label set (routine / red_flag) instead of 3-class sentiment, and
reading from a hand-labeled JSON file instead of Financial PhraseBank.

Split is done at the SENTENCE level, stratified by label, so val is
never touched during training - same leakage-prevention logic as
build_sentiment_pairs.py / ADR 0003's thesis-level split.

Usage:
    python scripts/build_redflag_pairs.py --file data/redflag/raw_redflag_news.json
"""

import argparse
import json
import random
from pathlib import Path
from collections import defaultdict

random.seed(42)

OUT_DIR = Path("data/redflag")
VAL_FRACTION = 0.2  # held-out sentences for eval, split BEFORE pair-building


def load_labeled_json(path: str) -> list:
    """Reads {"routine": [...], "red_flag": [...]} -> list of {text, label} dicts."""
    with open(path) as f:
        data = json.load(f)

    rows = []
    for label, texts in data.items():
        for t in texts:
            rows.append({"text": t.strip(), "label": label.strip().lower()})
    return rows


def split_train_val(rows: list, val_fraction: float = VAL_FRACTION) -> tuple:
    """Split at sentence level, stratified by label so val isn't
    accidentally all one class."""
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
    """For each sentence (anchor), sample same-label positives and
    different-label negatives - same idea as build_sentiment_pairs.py,
    just binary here so 'other_labels' always has exactly one option."""
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
    parser = argparse.ArgumentParser(description="Build redflag contrastive triplets")
    parser.add_argument("--file", default="data/redflag/raw_redflag_news.json")
    parser.add_argument("--triplets-per-anchor", type=int, default=2)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_labeled_json(args.file)
    print(f"Loaded {len(rows)} labeled sentences")
    label_counts = defaultdict(int)
    for r in rows:
        label_counts[r["label"]] += 1
    print(f"Label distribution: {dict(label_counts)}")

    train_rows, val_rows = split_train_val(rows)

    train_triplets = build_triplets(train_rows, args.triplets_per_anchor)
    write_jsonl(OUT_DIR / "train_triplets.jsonl", train_triplets)
    print(f"Wrote {len(train_triplets)} train triplets -> {OUT_DIR/'train_triplets.jsonl'}")

    # val saved as flat sentences (not triplets) - needed for nearest-
    # centroid eval, same as build_sentiment_pairs.py's val_sentences.jsonl
    write_jsonl(OUT_DIR / "val_sentences.jsonl", val_rows)
    print(f"Wrote {len(val_rows)} val sentences -> {OUT_DIR/'val_sentences.jsonl'}")


if __name__ == "__main__":
    main()