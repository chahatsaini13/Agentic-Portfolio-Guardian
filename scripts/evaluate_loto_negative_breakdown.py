"""
LOTO per-fold accuracy broken down by negative_type (easy vs hard).

Companion to train_contrastive_model.py's --split-mode loto, which only
reports one blended triplet-accuracy number per fold (see ADR 0003's
"LOTO's easy/hard breakdown is not yet computed per-fold" known
limitation). Reuses load_triplets(), to_examples(), make_evaluator()
from train_contrastive_model.py directly - no duplicated logic.

Usage:
    python scripts/evaluate_loto_negative_breakdown.py
"""

import argparse
import json
import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer, losses
from torch.utils.data import DataLoader

sys.path.insert(0, 'scripts')
from train_contrastive_model import load_triplets, to_examples, make_evaluator


def train_one_fold(train_rows, model_name, epochs, batch_size, lr):
    """Same recipe as train_contrastive_model.py's train_one_run(), minus
    saving to disk - we only need the trained model object back so we can
    evaluate it ourselves on easy/hard subsets afterward."""
    model = SentenceTransformer(model_name)
    train_examples = to_examples(train_rows)
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model)
    warmup_steps = max(1, int(len(train_dataloader) * epochs * 0.1))

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        show_progress_bar=True,
    )
    return model


def eval_accuracy(model, rows, name):
    """Triplet accuracy on a filtered subset - same metric definition as
    train_contrastive_model.py, reused via make_evaluator()."""
    if not rows:
        return None
    evaluator = make_evaluator(rows, name=name)
    result = evaluator(model)
    if isinstance(result, dict):
        acc_key = next((k for k in result if "accuracy" in k), None)
        return float(result[acc_key]) if acc_key else float(next(iter(result.values())))
    return float(result)


def main():
    parser = argparse.ArgumentParser(description="LOTO per-fold easy/hard negative breakdown")
    parser.add_argument("--data-dir", default="data/contrastive")
    parser.add_argument("--model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--out-file", default="data/contrastive/loto_negative_breakdown.json")
    args = parser.parse_args()

    all_rows = load_triplets(Path(args.data_dir) / "train_triplets.jsonl") + \
               load_triplets(Path(args.data_dir) / "val_triplets.jsonl")
    thesis_ids = sorted({r["thesis_id"] for r in all_rows})
    print(f"[loto-breakdown] {len(thesis_ids)} theses: {thesis_ids}")
    print(f"[loto-breakdown] {len(all_rows)} total triplets\n")

    fold_results = []
    for held_out in thesis_ids:
        train_rows = [r for r in all_rows if r["thesis_id"] != held_out]
        val_rows = [r for r in all_rows if r["thesis_id"] == held_out]
        easy_rows = [r for r in val_rows if r.get("negative_type") == "easy"]
        hard_rows = [r for r in val_rows if r.get("negative_type") == "hard"]

        print(f"[fold held_out={held_out}] train={len(train_rows)} "
              f"val_easy={len(easy_rows)} val_hard={len(hard_rows)}")

        model = train_one_fold(train_rows, args.model_name, args.epochs, args.batch_size, args.lr)

        combined_acc = eval_accuracy(model, val_rows, name=f"{held_out}_combined")
        easy_acc = eval_accuracy(model, easy_rows, name=f"{held_out}_easy")
        hard_acc = eval_accuracy(model, hard_rows, name=f"{held_out}_hard")

        print(f"  combined={combined_acc}  easy={easy_acc}  hard={hard_acc}\n")

        fold_results.append({
            "held_out": held_out,
            "n_val_easy": len(easy_rows),
            "n_val_hard": len(hard_rows),
            "combined_accuracy": combined_acc,
            "easy_accuracy": easy_acc,
            "hard_accuracy": hard_acc,
        })

    def _avg(key):
        vals = [f[key] for f in fold_results if f[key] is not None]
        return sum(vals) / len(vals) if vals else None

    summary = {
        "folds": fold_results,
        "average_combined_accuracy": _avg("combined_accuracy"),
        "average_easy_accuracy": _avg("easy_accuracy"),
        "average_hard_accuracy": _avg("hard_accuracy"),
    }

    print("=" * 60)
    print("LOTO PER-FOLD EASY/HARD BREAKDOWN - SUMMARY")
    print("=" * 60)
    for f in fold_results:
        print(f"  {f['held_out']:30s} combined={f['combined_accuracy']}  "
              f"easy={f['easy_accuracy']}  hard={f['hard_accuracy']}")
    print(f"\n  AVERAGE combined: {summary['average_combined_accuracy']}")
    print(f"  AVERAGE easy:     {summary['average_easy_accuracy']}")
    print(f"  AVERAGE hard:     {summary['average_hard_accuracy']}")

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote full breakdown -> {out_path}")


if __name__ == "__main__":
    main()