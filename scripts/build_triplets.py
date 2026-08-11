"""
Converts data/contrastive/raw_theses_news.json into:
  1. triplets.jsonl        - {anchor, positive, negative, negative_type} for contrastive training
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

NEW: every triplet now carries a "negative_type" field ("easy" or "hard"),
so evaluation can report accuracy separately for easy vs hard negatives -
not just a combined number. This matters because hard negatives (same
company, irrelevant news) are the whole reason we're doing contrastive
learning instead of keyword matching; a high combined accuracy that's
actually driven by easy negatives being trivial would be misleading.

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
    hardest cases at the expense of the obvious ones.

    Each triplet now records negative_type ("easy" or "hard") so downstream
    evaluation can break accuracy down by negative difficulty, not just
    report one blended number."""
    triplets = []
    for t in theses:
        easy_negs = t["irrelevant_news_easy"]
        hard_negs = t["irrelevant_news_hard"]
        for pos in t["relevant_news"]:
            for neg in easy_negs:
                triplets.append({
                    "thesis_id": t["id"],
                    "anchor": t["thesis"],
                    "positive": pos,
                    "negative": neg,
                    "negative_type": "easy",
                })
            for neg in hard_negs:
                triplets.append({
                    "thesis_id": t["id"],
                    "anchor": t["thesis"],
                    "positive": pos,
                    "negative": neg,
                    "negative_type": "hard",
                })
    return triplets


def build_eval_pairs(theses: list) -> list:
    """Flat labeled pairs - one row per (thesis, news, label). This is what
    you'll use in Week 9 to compute accuracy/AUC against the keyword baseline,
    so build it now while the source data is fresh.

    NOTE: this deliberately builds pairs from ALL theses, not just val ones -
    that's correct for building a general eval set, but means you MUST filter
    by a specific model's held-out thesis_ids before evaluating that model,
    or you'll be testing partly on data it already saw in training. See
    evaluate_relevance_model.py's --val-thesis-ids flag."""
    pairs = []
    for t in theses:
        for pos in t["relevant_news"]:
            pairs.append({"thesis_id": t["id"], "thesis": t["thesis"], "news": pos, "label": 1})
        for neg in t["irrelevant_news_easy"]:
            pairs.append({"thesis_id": t["id"], "thesis": t["thesis"], "news": neg, "label": 0,
                          "negative_type": "easy"})
        for neg in t["irrelevant_news_hard"]:
            pairs.append({"thesis_id": t["id"], "thesis": t["thesis"], "news": neg, "label": 0,
                          "negative_type": "hard"})
    return pairs


def split_by_thesis(theses: list, fixed_val_ids: list = None) -> tuple:
    """Fixed split: ~75% of theses -> train, ~25% -> val.
    If fixed_val_ids is given, use those exact thesis ids as val instead of
    random selection - needed when you want reproducible, specific val
    theses (e.g. to match an already-trained model you're now evaluating)."""
    ids = [t["id"] for t in theses]

    if fixed_val_ids:
        val_ids = set(fixed_val_ids)
        train_ids = set(ids) - val_ids
        return train_ids, val_ids

    random.shuffle(ids)
    n_val = max(1, round(len(ids) * VAL_FRACTION))
    val_ids = set(ids[:n_val])
    train_ids = set(ids[n_val:])
    return train_ids, val_ids


def generate_loto_splits(theses: list) -> list:
    """Leave-one-thesis-out: returns one split dict per thesis, each holding
    out exactly that one thesis as val and training on the other 7."""
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


def write_training_config(out_dir: Path, split_mode: str):
    config = {
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "loss_function": "MultipleNegativesRankingLoss",
        "margin": "n/a - MultipleNegativesRankingLoss doesn't use a margin",
        "epochs": 4,
        "learning_rate": "TODO - fill with actual value used",
        "batch_size": "TODO - fill with actual value used",
        "split_mode": split_mode,
        "notes": "negative_type field added to triplets for easy/hard breakdown."
    }
    path = out_dir / "training_config.json"
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Wrote training config -> {path}")


def run_fixed_split(theses: list, all_triplets: list, fixed_val_ids: list = None):
    train_ids, val_ids = split_by_thesis(theses, fixed_val_ids=fixed_val_ids)
    print(f"[fixed split] Train theses ({len(train_ids)}): {sorted(train_ids)}")
    print(f"[fixed split] Val theses   ({len(val_ids)}): {sorted(val_ids)}")

    train_triplets = [tr for tr in all_triplets if tr["thesis_id"] in train_ids]
    val_triplets = [tr for tr in all_triplets if tr["thesis_id"] in val_ids]

    write_jsonl(OUT_DIR / "train_triplets.jsonl", train_triplets)
    write_jsonl(OUT_DIR / "val_triplets.jsonl", val_triplets)

    print(f"Wrote {len(train_triplets)} train triplets -> {OUT_DIR/'train_triplets.jsonl'}")
    print(f"Wrote {len(val_triplets)} val triplets -> {OUT_DIR/'val_triplets.jsonl'}")
    return val_ids


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


def main():
    parser = argparse.ArgumentParser(description="Build contrastive triplets from raw thesis/news data")
    parser.add_argument("--split-mode", choices=["fixed", "loto"], default="fixed")
    parser.add_argument("--fixed-val-ids", nargs="+", default=None,
                         help="Explicit thesis ids to use as val in fixed mode "
                              "(e.g. t4_hdfcbank_retail t5_sunpharma_generics) - "
                              "use this to reproduce/match an already-trained model's split.")
    args = parser.parse_args()

    raw = load_raw()
    theses = raw["theses"]

    all_triplets = build_triplets(theses)
    all_eval_pairs = build_eval_pairs(theses)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.split_mode == "fixed":
        run_fixed_split(theses, all_triplets, fixed_val_ids=args.fixed_val_ids)
    else:
        run_loto_splits(theses, all_triplets)

    write_jsonl(OUT_DIR / "eval_pairs.jsonl", all_eval_pairs)
    print(f"Wrote {len(all_eval_pairs)} eval pairs -> {OUT_DIR/'eval_pairs.jsonl'}")

    write_training_config(OUT_DIR, args.split_mode)


if __name__ == "__main__":
    main()