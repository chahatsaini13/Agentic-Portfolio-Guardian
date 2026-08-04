"""
Converts data/contrastive/raw_theses_news.json into:
  1. triplets.jsonl        - {anchor, positive, negative} for contrastive training
  2. eval_pairs.jsonl      - {thesis, news, label} flat pairs for threshold tuning / accuracy eval
  3. train/val split of the triplets (by thesis id, so no thesis leaks across split)

Two split modes are supported (pick with --split-mode):
  - "fixed" (default, original behavior): one random ~75/25 split of the 8
    theses into train/val.
  - "loto": leave-one-thesis-out. Generates 8 separate train/val splits, each
    holding out exactly one thesis as val. Train + evaluate once per split,
    then average the val metric across all 8. With only 8 theses total, this
    is far more trustworthy than a single fixed split, whose val set is only
    2 thesis anchors (see teammate review notes).

Run:
    python scripts/build_triplets.py                    # fixed split
    python scripts/build_triplets.py --split-mode loto   # leave-one-thesis-out
"""

import argparse
import json
import random
from pathlib import Path

random.seed(42)

RAW_PATH = Path("data/contrastive/raw_theses_news.json")
OUT_DIR = Path("data/contrastive")

VAL_FRACTION = 0.25   # held-out theses go entirely to val - no thesis appears in both splits
SHUFFLE_SEED = 42      # seed used only for row shuffling before writing .jsonl files


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
    """Original fixed random split: ~75% of theses -> train, ~25% -> val.
    NOTE: with only 8 theses this means val is just 2 thesis anchors, so
    treat any val metric from this split as a rough sanity check, not a
    reliable number. Use --split-mode loto for a trustworthy estimate."""
    ids = [t["id"] for t in theses]
    random.shuffle(ids)
    n_val = max(1, round(len(ids) * VAL_FRACTION))
    val_ids = set(ids[:n_val])
    train_ids = set(ids[n_val:])
    return train_ids, val_ids


def generate_loto_splits(theses: list) -> list:
    """Leave-one-thesis-out: returns one split dict per thesis, each holding
    out exactly that one thesis as val and training on the other 7. Doing
    this for all 8 theses and averaging the resulting val metric uses every
    anchor as val exactly once, instead of only ever testing on the 2 theses
    a fixed split happened to pick."""
    all_ids = [t["id"] for t in theses]
    splits = []
    for held_out_id in all_ids:
        val_ids = {held_out_id}
        train_ids = set(all_ids) - val_ids
        splits.append({
            "held_out": held_out_id,
            "train_ids": train_ids,
            "val_ids": val_ids,
        })
    return splits


def shuffle_rows(rows: list, seed: int = SHUFFLE_SEED) -> list:
    """Shuffle a copy of rows using a dedicated Random instance (not the
    global `random` module) so this doesn't disturb the random state that
    split_by_thesis relies on, and stays reproducible regardless of what ran
    before it."""
    rng = random.Random(seed)
    shuffled = rows[:]
    rng.shuffle(shuffled)
    return shuffled


def write_jsonl(path: Path, rows: list, shuffle: bool = True):
    if shuffle:
        rows = shuffle_rows(rows)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def write_training_config(out_dir: Path):
    """Placeholder training config so hyperparameters are tracked from day
    one instead of getting hardcoded inside a training script later. Fill
    the TODOs in once fine-tuning actually starts."""
    config = {
        "model": "TODO - e.g. sentence-transformers/all-MiniLM-L6-v2",
        "loss_function": "TODO - e.g. TripletMarginLoss / MultipleNegativesRankingLoss",
        "margin": "TODO - only relevant for TripletMarginLoss",
        "epochs": "TODO",
        "learning_rate": "TODO",
        "batch_size": "TODO",
        "split_mode": "TODO - fixed | loto",
        "notes": "Fill in once fine-tuning starts; keep updated if hyperparameters change during experimentation."
    }
    path = out_dir / "training_config.json"
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Wrote training config placeholder -> {path}")


def run_fixed_split(theses: list, all_triplets: list):
    train_ids, val_ids = split_by_thesis(theses)
    print(f"[fixed split] Train theses ({len(train_ids)}): {sorted(train_ids)}")
    print(f"[fixed split] Val theses   ({len(val_ids)}): {sorted(val_ids)}")

    train_triplets = [tr for tr in all_triplets if tr["thesis_id"] in train_ids]
    val_triplets = [tr for tr in all_triplets if tr["thesis_id"] in val_ids]

    write_jsonl(OUT_DIR / "train_triplets.jsonl", train_triplets)
    write_jsonl(OUT_DIR / "val_triplets.jsonl", val_triplets)

    print(f"Wrote {len(train_triplets)} train triplets -> {OUT_DIR/'train_triplets.jsonl'}")
    print(f"Wrote {len(val_triplets)} val triplets -> {OUT_DIR/'val_triplets.jsonl'}")


def run_loto_splits(theses: list, all_triplets: list):
    splits = generate_loto_splits(theses)
    for split in splits:
        held_out = split["held_out"]
        train_triplets = [tr for tr in all_triplets if tr["thesis_id"] in split["train_ids"]]
        val_triplets = [tr for tr in all_triplets if tr["thesis_id"] in split["val_ids"]]

        train_path = OUT_DIR / f"loto_{held_out}_train_triplets.jsonl"
        val_path = OUT_DIR / f"loto_{held_out}_val_triplets.jsonl"
        write_jsonl(train_path, train_triplets)
        write_jsonl(val_path, val_triplets)

        print(f"[loto held_out={held_out}] {len(train_triplets)} train / {len(val_triplets)} val triplets "
              f"-> {train_path.name}, {val_path.name}")

    print(f"\nWrote {len(splits)} LOTO splits -> {OUT_DIR}/loto_<held_out_id>_{{train,val}}_triplets.jsonl")
    print("Train + evaluate once per split, then average the val metric across all splits.")


def main():
    parser = argparse.ArgumentParser(description="Build contrastive triplets from raw thesis/news data")
    parser.add_argument(
        "--split-mode",
        choices=["fixed", "loto"],
        default="fixed",
        help="fixed = single random ~75/25 split by thesis (default, original behavior). "
             "loto = leave-one-thesis-out, generates 8 splits to average a val metric over.",
    )
    args = parser.parse_args()

    raw = load_raw()
    theses = raw["theses"]

    all_triplets = build_triplets(theses)
    all_eval_pairs = build_eval_pairs(theses)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.split_mode == "fixed":
        run_fixed_split(theses, all_triplets)
    else:
        run_loto_splits(theses, all_triplets)

    write_jsonl(OUT_DIR / "eval_pairs.jsonl", all_eval_pairs)
    print(f"Wrote {len(all_eval_pairs)} eval pairs -> {OUT_DIR/'eval_pairs.jsonl'}")

    write_training_config(OUT_DIR)


if __name__ == "__main__":
    main()
