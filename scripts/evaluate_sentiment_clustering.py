"""
Silhouette score + cluster purity comparison for sentiment embeddings -
baseline (generic) vs contrastive (fine-tuned) - on
data/sentiment/val_sentences.jsonl.

Companion to evaluate_sentiment_model.py's nearest-centroid comparison;
this script instead measures embedding-space cluster quality directly:
  - silhouette_score against the TRUE labels (no clustering needed - just
    "are the 3 known sentiment classes well-separated in this embedding
    space")
  - purity from actual unsupervised KMeans clustering into 3 clusters,
    checked against the true labels

Run once per model, same --label convention as evaluate_sentiment_model.py,
so results are easy to diff side by side:

Usage:
    python scripts/evaluate_sentiment_clustering.py \
        --model sentence-transformers/all-MiniLM-L6-v2 --label baseline
    python scripts/evaluate_sentiment_clustering.py \
        --model models/contrastive_sentiment_v1 --label contrastive
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def load_jsonl(path: Path) -> list:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compute_purity(true_labels: list, cluster_ids) -> float:
    """Standard clustering purity: for every cluster, take the most common
    true label inside it, sum those counts, divide by total points.
    1.0 = every cluster is perfectly single-label; low = clusters mix
    labels together."""
    total_correct = 0
    for cluster_id in set(cluster_ids):
        idx_in_cluster = [i for i, c in enumerate(cluster_ids) if c == cluster_id]
        if not idx_in_cluster:
            continue
        labels_in_cluster = [true_labels[i] for i in idx_in_cluster]
        _, count = Counter(labels_in_cluster).most_common(1)[0]
        total_correct += count
    return total_correct / len(true_labels)


def main():
    parser = argparse.ArgumentParser(description="Silhouette + purity eval for sentiment embeddings")
    parser.add_argument("--model", required=True, help="model name or local path")
    parser.add_argument("--label", required=True, help="display name for this run, e.g. baseline/contrastive")
    parser.add_argument("--data-dir", default="data/sentiment")
    parser.add_argument("--val-file", default="val_sentences.jsonl")
    parser.add_argument("--n-clusters", type=int, default=3,
                         help="clusters for the unsupervised purity measure - 3 to match "
                              "positive/negative/neutral")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    val_path = Path(args.data_dir) / args.val_file
    rows = load_jsonl(val_path)
    print(f"Loaded {len(rows)} val sentences from {val_path}")

    texts = [r["text"] for r in rows]
    true_labels = [r["label"] for r in rows]
    print(f"Label distribution: {dict(Counter(true_labels))}")

    print(f"Loading model: {args.model}")
    model = SentenceTransformer(args.model)

    print("Encoding val sentences ...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # Silhouette against TRUE labels - measures how separated the 3 known
    # sentiment classes already are in this embedding space, no clustering
    # step involved.
    sil_true = silhouette_score(embeddings, true_labels, metric="cosine")

    # Purity from actual unsupervised KMeans clustering into n_clusters.
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=args.random_state, n_init=10)
    cluster_ids = kmeans.fit_predict(embeddings)
    purity = compute_purity(true_labels, cluster_ids)

    # Silhouette on the KMeans clusters themselves too - shows whether the
    # unsupervised clusters are well-formed as clusters, separate from
    # whether they happen to line up with the true labels.
    sil_kmeans = silhouette_score(embeddings, cluster_ids, metric="cosine")

    print(f"\n=== RESULTS ({args.label}) ===")
    print(f"Model:                            {args.model}")
    print(f"N sentences:                      {len(rows)}")
    print(f"Silhouette (vs true labels):      {sil_true:.4f}")
    print(f"Silhouette (vs KMeans clusters):  {sil_kmeans:.4f}")
    print(f"Purity (KMeans k={args.n_clusters} vs true labels): {purity:.4f}")

    results_path = Path(args.data_dir) / "clustering_eval_results.jsonl"
    with open(results_path, "a") as f:
        f.write(json.dumps({
            "label": args.label,
            "model": args.model,
            "n_sentences": len(rows),
            "silhouette_vs_true_labels": sil_true,
            "silhouette_vs_kmeans_clusters": sil_kmeans,
            "purity_kmeans": purity,
            "n_clusters": args.n_clusters,
        }) + "\n")
    print(f"\nAppended result to {results_path}")


if __name__ == "__main__":
    main()
