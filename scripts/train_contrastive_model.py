"""
Fine-tunes a sentence-transformers model to score thesis <-> news relevance,
using (anchor=thesis, positive=relevant news, negative=irrelevant news) triplets.

Two modes:
  --split-mode fixed   uses train_triplets.jsonl / val_triplets.jsonl as-is
                        (the 6-thesis-train / 2-thesis-val split from
                        build_triplets.py)
  --split-mode loto     leave-one-thesis-out cross-validation. We don't have
                        separate loto_<id>_train/val files, so this mode
                        combines train_triplets.jsonl + val_triplets.jsonl
                        back into one pool, then for each of the 8 thesis
                        ids: hold that thesis out as val, train on the
                        other 7, and record validation triplet accuracy.
                        Reports the average across all 8 folds - a much
                        more trustworthy number than the single fixed
                        6/2 split, since 2 theses is a tiny, high-variance
                        val set.

Metric: triplet accuracy = % of triplets where the model scores
(anchor, positive) as more similar than (anchor, negative). This is what
sentence-transformers' built-in TripletEvaluator computes, and it's
directly interpretable ("did the model get the ranking right"), unlike
raw loss values.

Usage:
    python train_contrastive_model.py --split-mode fixed
    python train_contrastive_model.py --split-mode loto
"""

import argparse
import json
from pathlib import Path

from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import TripletEvaluator
from torch.utils.data import DataLoader


def load_triplets(path: Path) -> list:
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def to_examples(rows: list) -> list:
    return [InputExample(texts=[r["anchor"], r["positive"], r["negative"]]) for r in rows]


def make_evaluator(rows: list, name: str) -> TripletEvaluator:
    """TripletEvaluator wants three parallel lists, not InputExamples."""
    anchors = [r["anchor"] for r in rows]
    positives = [r["positive"] for r in rows]
    negatives = [r["negative"] for r in rows]
    return TripletEvaluator(anchors, positives, negatives, name=name, batch_size=16)


def train_one_run(train_rows: list, val_rows: list, model_name: str, epochs: int,
                   batch_size: int, lr: float, output_dir: str, save_model: bool) -> float:
    """Trains one model on train_rows, evaluates on val_rows each epoch,
    returns the best triplet accuracy seen. If save_model, keeps the
    best-performing checkpoint on disk at output_dir (sentence-transformers'
    save_best_model handles this automatically when an evaluator is passed)."""
    model = SentenceTransformer(model_name)
    train_examples = to_examples(train_rows)
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    evaluator = make_evaluator(val_rows, name="val")
    warmup_steps = max(1, int(len(train_dataloader) * epochs * 0.1))

    fit_kwargs = dict(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        evaluator=evaluator,
        evaluation_steps=len(train_dataloader),  # evaluate once per epoch
        show_progress_bar=True,
    )
    if save_model:
        fit_kwargs["output_path"] = output_dir
        fit_kwargs["save_best_model"] = True

    model.fit(**fit_kwargs)

    # Re-run the evaluator once more explicitly to get a clean final number
    # (fit() logs to a CSV under output_dir, but we want the value in hand).
    # Newer sentence-transformers versions return a dict (e.g.
    # {"val_cosine_accuracy": 1.0}) instead of a bare float - handle both.
    result = evaluator(model)
    if isinstance(result, dict):
        # key is usually "<name>_cosine_accuracy" - find it robustly
        acc_key = next((k for k in result if "accuracy" in k), None)
        final_acc = float(result[acc_key]) if acc_key else float(next(iter(result.values())))
    else:
        final_acc = float(result)
    return final_acc


def run_fixed(args):
    train_rows = load_triplets(Path(args.data_dir) / "train_triplets.jsonl")
    val_rows = load_triplets(Path(args.data_dir) / "val_triplets.jsonl")
    print(f"[fixed] train triplets: {len(train_rows)} | val triplets: {len(val_rows)}")

    acc = train_one_run(
        train_rows, val_rows, args.model_name, args.epochs, args.batch_size,
        args.lr, args.output_dir, save_model=True,
    )
    print(f"\n[fixed] final val triplet accuracy: {acc:.4f}")
    print(f"[fixed] model saved to: {args.output_dir}")


def run_loto(args):
    all_rows = load_triplets(Path(args.data_dir) / "train_triplets.jsonl") + \
               load_triplets(Path(args.data_dir) / "val_triplets.jsonl")
    thesis_ids = sorted({r["thesis_id"] for r in all_rows})
    print(f"[loto] {len(thesis_ids)} theses found: {thesis_ids}")

    fold_accuracies = {}
    for held_out in thesis_ids:
        train_rows = [r for r in all_rows if r["thesis_id"] != held_out]
        val_rows = [r for r in all_rows if r["thesis_id"] == held_out]
        print(f"\n[loto] fold: held-out={held_out} "
              f"(train={len(train_rows)}, val={len(val_rows)})")

        # Don't save every fold's model to disk - these are just for the
        # cross-validated metric, not for deployment.
        acc = train_one_run(
            train_rows, val_rows, args.model_name, args.epochs, args.batch_size,
            args.lr, output_dir=None, save_model=False,
        )
        fold_accuracies[held_out] = acc
        print(f"[loto] fold {held_out}: triplet accuracy = {acc:.4f}")

    avg_acc = sum(fold_accuracies.values()) / len(fold_accuracies)
    print("\n" + "=" * 50)
    print("LOTO RESULTS (per fold)")
    print("=" * 50)
    for tid, acc in fold_accuracies.items():
        print(f"  {tid:30s} {acc:.4f}")
    print(f"\n  AVERAGE across {len(fold_accuracies)} folds: {avg_acc:.4f}")

    if args.save_final_model:
        print(f"\n[loto] retraining on ALL {len(all_rows)} triplets (all 8 theses) "
              f"for the deployable checkpoint at {args.output_dir}.")
        print("[loto] note: this final model's quality is estimated by the "
              "LOTO average above, not re-validated in-sample.")
        # No held-out val set left, so just reuse a small slice as a sanity
        # evaluator (not a real val set - purely to let save_best_model work).
        sanity_val = all_rows[: max(8, len(all_rows) // 20)]
        train_one_run(
            all_rows, sanity_val, args.model_name, args.epochs, args.batch_size,
            args.lr, args.output_dir, save_model=True,
        )
        print(f"[loto] final model saved to: {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train contrastive thesis-relevance model")
    parser.add_argument("--split-mode", choices=["fixed", "loto"], default="fixed")
    parser.add_argument("--data-dir", default=".", help="dir containing train/val jsonl files")
    parser.add_argument("--model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--output-dir", default="models/contrastive_relevance_v1")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--save-final-model", action="store_true", default=True,
                         help="in loto mode, also train+save a deployable model on all data")
    args = parser.parse_args()

    if args.split_mode == "fixed":
        run_fixed(args)
    else:
        run_loto(args)


if __name__ == "__main__":
    main()
