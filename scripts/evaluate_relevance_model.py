"""
Loads a saved sentence-transformers checkpoint and scores it against
eval_pairs.jsonl (flat thesis/news/label rows - NOT used in training).

For each (thesis, news) pair we compute cosine similarity between their
embeddings. If similarity >= threshold, we predict "relevant" (1),
otherwise "irrelevant" (0). We compare that prediction against the
ground-truth label and report accuracy/precision/recall/F1.

This is what you'll use in Week 9 to compare the contrastive model against
the keyword-only baseline.

Usage:
    python evaluate_relevance_model.py --model-path models/contrastive_relevance_v1 \
        --eval-file eval_pairs.jsonl --threshold 0.5
"""

import argparse
import json

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def load_eval_pairs(path: str) -> list:
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser(description="Evaluate relevance model on eval_pairs.jsonl")
    parser.add_argument("--model-path", required=True,
                         help="path to a saved SentenceTransformer checkpoint, "
                              "or a model name like 'sentence-transformers/all-MiniLM-L6-v2' "
                              "for the untrained baseline")
    parser.add_argument("--eval-file", default="eval_pairs.jsonl")
    parser.add_argument("--threshold", type=float, default=0.5,
                         help="cosine similarity cutoff above which a pair counts as relevant")
    args = parser.parse_args()

    print(f"Loading model from: {args.model_path}")
    model = SentenceTransformer(args.model_path)

    rows = load_eval_pairs(args.eval_file)
    print(f"Loaded {len(rows)} eval pairs")

    theses = [r["thesis"] for r in rows]
    news = [r["news"] for r in rows]
    labels = [r["label"] for r in rows]

    thesis_emb = model.encode(theses, convert_to_tensor=True, show_progress_bar=True)
    news_emb = model.encode(news, convert_to_tensor=True, show_progress_bar=True)

    sims = [cos_sim(thesis_emb[i], news_emb[i]).item() for i in range(len(rows))]
    preds = [1 if s >= args.threshold else 0 for s in sims]

    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)

    print("\n" + "=" * 50)
    print(f"EVAL RESULTS  (threshold={args.threshold})")
    print("=" * 50)
    print(f"  accuracy:  {acc:.4f}")
    print(f"  precision: {prec:.4f}")
    print(f"  recall:    {rec:.4f}")
    print(f"  f1:        {f1:.4f}")
    print(f"  n pairs:   {len(rows)}  (positives={sum(labels)}, negatives={len(labels)-sum(labels)})")

    # Also dump a few sample errors - useful for sanity-checking the threshold,
    # not just the aggregate metric.
    print("\nSample misclassifications:")
    shown = 0
    for r, s, p in zip(rows, sims, preds):
        if p != r["label"] and shown < 5:
            print(f"  [{r['label']}->pred {p}, sim={s:.3f}] {r['thesis'][:50]}... | {r['news'][:60]}...")
            shown += 1


if __name__ == "__main__":
    main()
