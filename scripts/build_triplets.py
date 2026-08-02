"""
Converts data/contrastive/raw_theses_news.json into:
  1. triplets.jsonl        - {anchor, positive, negative} for contrastive training
  2. eval_pairs.jsonl      - {thesis, news, label} flat pairs for threshold tuning / accuracy eval
  3. train/val split of the triplets (by thesis id, so no thesis leaks across split)

Run:
    python scripts/build_triplets.py
"""

import json
import random
from pathlib import Path

random.seed(42)

RAW_PATH = Path("data/contrastive/raw_theses_news.json")
OUT_DIR = Path("data/contrastive")

VAL_FRACTION = 0.25  # held-out theses go entirely to val - no thesis appears in both splits


def load_raw() -> dict:
    with open(RAW_PATH) as f:
        return json.load(f)


def build_triplets(theses: list) -> list:
    """One triplet per (positive, negative) combination within a thesis.
    Mixing in both easy and hard negatives per thesis, not just hard ones -
    the model needs to see the full difficulty spectrum, not overfit to the
    hardest cases at the expense of the obvious ones."""
    triplets = []
    for t in theses:
        negatives = t["irrelevant_news_easy"] + t["irrelevant_news_hard"]
        for pos in t["relevant_news"]:
            for neg in negatives:
                triplets.append({
                    "thesis_id": t["id"],
                    "anchor": t["thesis"],
                    "positive": pos,
                    "negative": neg,
                })
    return triplets


def build_eval_pairs(theses: list) -> list:
    """Flat labeled pairs - one row per (thesis, news, label). This is what
    you'll use in Week 9 to compute accuracy/AUC against the keyword baseline,
    so build it now while the source data is fresh."""
    pairs = []
    for t in theses:
        for pos in t["relevant_news"]:
            pairs.append({"thesis_id": t["id"], "thesis": t["thesis"], "news": pos, "label": 1})
        for neg in t["irrelevant_news_easy"] + t["irrelevant_news_hard"]:
            pairs.append({"thesis_id": t["id"], "thesis": t["thesis"], "news": neg, "label": 0})
    return pairs


def split_by_thesis(theses: list) -> tuple:
    ids = [t["id"] for t in theses]
    random.shuffle(ids)
    n_val = max(1, round(len(ids) * VAL_FRACTION))
    val_ids = set(ids[:n_val])
    train_ids = set(ids[n_val:])
    return train_ids, val_ids


def write_jsonl(path: Path, rows: list):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main():
    raw = load_raw()
    theses = raw["theses"]

    train_ids, val_ids = split_by_thesis(theses)
    print(f"Train theses ({len(train_ids)}): {sorted(train_ids)}")
    print(f"Val theses   ({len(val_ids)}): {sorted(val_ids)}")

    all_triplets = build_triplets(theses)
    train_triplets = [tr for tr in all_triplets if tr["thesis_id"] in train_ids]
    val_triplets = [tr for tr in all_triplets if tr["thesis_id"] in val_ids]

    all_eval_pairs = build_eval_pairs(theses)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT_DIR / "train_triplets.jsonl", train_triplets)
    write_jsonl(OUT_DIR / "val_triplets.jsonl", val_triplets)
    write_jsonl(OUT_DIR / "eval_pairs.jsonl", all_eval_pairs)

    print(f"\nWrote {len(train_triplets)} train triplets -> {OUT_DIR/'train_triplets.jsonl'}")
    print(f"Wrote {len(val_triplets)} val triplets -> {OUT_DIR/'val_triplets.jsonl'}")
    print(f"Wrote {len(all_eval_pairs)} eval pairs -> {OUT_DIR/'eval_pairs.jsonl'}")


if __name__ == "__main__":
    main()
