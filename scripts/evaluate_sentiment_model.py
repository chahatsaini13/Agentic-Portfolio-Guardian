"""
Evaluates a sentiment embedding model (baseline or fine-tuned) using
nearest-centroid classification: build one centroid embedding per label
from a sample of train sentences, then classify val_sentences.jsonl by
which centroid each val sentence is closest to (cosine similarity).

Run once with the baseline model name and once with the fine-tuned path
to get a side-by-side comparison, same "before/after" pattern as
evaluate_relevance_model.py.

Usage:
    python evaluate_sentiment_model.py --model sentence-transformers/all-MiniLM-L6-v2 --label baseline
    python evaluate_sentiment_model.py --model models/contrastive_sentiment_v1 --label contrastive
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from sentence_transformers import SentenceTransformer, util


def load_jsonl(path: Path) -> list:
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def build_centroids(model, train_rows: list, per_label_sample: int = 200) -> dict:
    """One mean-pooled embedding per label, computed from a sample of
    train sentences (not all 3876 - keeps this step fast, and centroid
    quality saturates well before using every sentence)."""
    by_label = defaultdict(list)
    for r in train_rows:
        by_label[r["anchor_label"]].append(r["anchor"])
        by_label[r["negative_label"]].append(r["negative"])

    centroids = {}
    for label, texts in by_label.items():
        sample = texts[:per_label_sample]
        embeddings = model.encode(sample, convert_to_tensor=True, show_progress_bar=False)
        centroids[label] = embeddings.mean(dim=0)
    return centroids


def classify(model, centroids: dict, text: str) -> str:
    emb = model.encode(text, convert_to_tensor=True, show_progress_bar=False)
    scores = {label: util.cos_sim(emb, c).item() for label, c in centroids.items()}
    return max(scores, key=scores.get)


def main():
    parser = argparse.ArgumentParser(description="Nearest-centroid sentiment eval")
    parser.add_argument("--model", required=True, help="model name or local path")
    parser.add_argument("--label", required=True, help="display name for this run, e.g. baseline/contrastive")
    parser.add_argument("--data-dir", default="data/sentiment")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model = SentenceTransformer(args.model)

    train_rows = load_jsonl(Path(args.data_dir) / "train_triplets.jsonl")
    val_rows = load_jsonl(Path(args.data_dir) / "val_sentences.jsonl")
    print(f"Building centroids from train triplets, evaluating on {len(val_rows)} val sentences")

    centroids = build_centroids(model, train_rows)
    print(f"Centroid labels: {list(centroids.keys())}")

    correct = 0
    confusion = defaultdict(lambda: defaultdict(int))
    for r in val_rows:
        pred = classify(model, centroids, r["text"])
        true = r["label"]
        confusion[true][pred] += 1
        if pred == true:
            correct += 1

    accuracy = correct / len(val_rows)
    print(f"\n=== RESULTS ({args.label}) ===")
    print(f"Accuracy: {accuracy:.4f}  ({correct}/{len(val_rows)})")
    print("\nConfusion (rows=true, cols=predicted):")
    labels = sorted(centroids.keys())
    print("           " + "  ".join(f"{l:>10s}" for l in labels))
    for true_label in labels:
        row = "  ".join(f"{confusion[true_label][pred_label]:>10d}" for pred_label in labels)
        print(f"{true_label:>10s} {row}")

    # append to a running results file so both runs (baseline + contrastive)
    # are easy to compare afterward without re-running anything
    results_path = Path(args.data_dir) / "eval_results.jsonl"
    with open(results_path, "a") as f:
        f.write(json.dumps({"label": args.label, "model": args.model, "accuracy": accuracy}) + "\n")
    print(f"\nAppended result to {results_path}")


if __name__ == "__main__":
    main()
