"""
Loads a saved sentence-transformers checkpoint and scores it against
eval_pairs.jsonl (flat thesis/news/label rows - NOT used in training).

For each (thesis, news) pair we compute cosine similarity between their
embeddings. If similarity >= threshold, we predict "relevant" (1),
otherwise "irrelevant" (0). We compare that prediction against the
ground-truth label and report accuracy/precision/recall/F1.

IMPORTANT - held-out filtering:
eval_pairs.jsonl is built from ALL 8 theses (see build_triplets.py), not
just whichever 2 were held out as val for a given model. If you evaluate a
model on the full eval_pairs.jsonl, you're partly testing it on data it
already saw during training - the result will look better than it should.

Use --val-thesis-ids to restrict evaluation to only the thesis_ids that
were genuinely held out for the model you're testing (e.g. for the
fixed-split model trained on 6 theses with t4/t5 held out, pass
--val-thesis-ids t4_hdfcbank_retail t5_sunpharma_generics).

Usage:
    python evaluate_relevance_model.py --model-path models/contrastive_relevance_v1 \
        --eval-file data/contrastive/eval_pairs.jsonl \
        --val-thesis-ids t4_hdfcbank_retail t5_sunpharma_generics \
        --threshold 0.5 --breakdown-by-negative-type
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


def filter_eval_pairs(rows: list, val_thesis_ids: list) -> list:
    """Keep only rows whose thesis_id is in val_thesis_ids - i.e. only
    theses the model being evaluated never saw during training. Raises if
    a given id doesn't match anything, so a typo fails loudly instead of
    silently returning an empty/wrong-looking result."""
    val_ids = set(val_thesis_ids)
    all_ids_present = {r["thesis_id"] for r in rows}
    missing = val_ids - all_ids_present
    if missing:
        raise ValueError(
            f"These --val-thesis-ids weren't found in {rows[0].get('thesis_id', '?')!r}-style "
            f"thesis_id values in the eval file: {sorted(missing)}. "
            f"Available ids: {sorted(all_ids_present)}"
        )
    filtered = [r for r in rows if r["thesis_id"] in val_ids]
    return filtered


def compute_metrics(labels: list, preds: list) -> dict:
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "n": len(labels),
        "positives": sum(labels),
        "negatives": len(labels) - sum(labels),
    }


def print_metrics(title: str, m: dict):
    print(f"\n{title}")
    print(f"  n pairs:   {m['n']}  (positives={m['positives']}, negatives={m['negatives']})")
    print(f"  accuracy:  {m['accuracy']:.4f}")
    print(f"  precision: {m['precision']:.4f}")
    print(f"  recall:    {m['recall']:.4f}")
    print(f"  f1:        {m['f1']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate relevance model on eval_pairs.jsonl")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--eval-file", default="data/contrastive/eval_pairs.jsonl")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--val-thesis-ids", nargs="+", default=None,
                         help="Restrict evaluation to only these thesis_ids "
                              "(the ones genuinely held out from this model's training). "
                              "Strongly recommended - without this you may be evaluating "
                              "partly on data the model already saw.")
    parser.add_argument("--breakdown-by-negative-type", action="store_true",
                         help="Also report accuracy separately for easy vs hard negatives "
                              "(requires eval_pairs.jsonl to have the negative_type field - "
                              "regenerate with the updated build_triplets.py if missing).")
    parser.add_argument("--show-all-scores", action="store_true",
                         help="Print every pair's raw cosine similarity score, correct or not - "
                              "useful for checking the MARGIN between relevant and irrelevant "
                              "pairs, not just the pass/fail count. A perfect accuracy score with "
                              "razor-thin margins (e.g. relevant=0.52, irrelevant=0.48) is much "
                              "less trustworthy than one with wide separation (e.g. 0.85 vs 0.20), "
                              "especially on a small sample.")
    args = parser.parse_args()

    print(f"Loading model from: {args.model_path}")
    model = SentenceTransformer(args.model_path)

    rows = load_eval_pairs(args.eval_file)
    print(f"Loaded {len(rows)} eval pairs total")

    if args.val_thesis_ids:
        rows = filter_eval_pairs(rows, args.val_thesis_ids)
        print(f"Filtered to {len(rows)} pairs for held-out thesis_ids: {args.val_thesis_ids}")
    else:
        print("[warn] No --val-thesis-ids given - evaluating on ALL theses. "
              "This may include data the model saw during training. "
              "Pass --val-thesis-ids to get a genuinely held-out number.")

    theses = [r["thesis"] for r in rows]
    news = [r["news"] for r in rows]
    labels = [r["label"] for r in rows]

    thesis_emb = model.encode(theses, convert_to_tensor=True, show_progress_bar=True)
    news_emb = model.encode(news, convert_to_tensor=True, show_progress_bar=True)

    sims = [cos_sim(thesis_emb[i], news_emb[i]).item() for i in range(len(rows))]
    preds = [1 if s >= args.threshold else 0 for s in sims]

    overall = compute_metrics(labels, preds)
    print_metrics(f"OVERALL RESULTS (threshold={args.threshold})", overall)

    if args.breakdown_by_negative_type:
        has_neg_type = any("negative_type" in r for r in rows)
        if not has_neg_type:
            print("\n[warn] --breakdown-by-negative-type requested but eval_pairs.jsonl rows "
                  "have no negative_type field. Regenerate eval_pairs.jsonl with the updated "
                  "build_triplets.py first.")
        else:
            # positives don't have a negative_type (label=1 rows) - only negatives do.
            # For the breakdown we compare: all positives (always included) + easy negatives,
            # vs all positives + hard negatives, so precision/recall stay meaningful.
            pos_idx = [i for i, r in enumerate(rows) if r["label"] == 1]
            easy_idx = [i for i, r in enumerate(rows) if r.get("negative_type") == "easy"]
            hard_idx = [i for i, r in enumerate(rows) if r.get("negative_type") == "hard"]

            for name, neg_idx in [("EASY NEGATIVES", easy_idx), ("HARD NEGATIVES", hard_idx)]:
                idx = sorted(pos_idx + neg_idx)
                sub_labels = [labels[i] for i in idx]
                sub_preds = [preds[i] for i in idx]
                m = compute_metrics(sub_labels, sub_preds)
                print_metrics(f"{name} ONLY (positives + {name.lower()})", m)

    print("\nSample misclassifications:")
    shown = 0
    for r, s, p in zip(rows, sims, preds):
        if p != r["label"] and shown < 5:
            neg_type = r.get("negative_type", "")
            print(f"  [{r['label']}->pred {p}, sim={s:.3f}{', ' + neg_type if neg_type else ''}] "
                  f"{r['thesis'][:50]}... | {r['news'][:60]}...")
            shown += 1
    if shown == 0:
        print("  (none)")

    if args.show_all_scores:
        print(f"\nAll {len(rows)} pair scores (sorted by similarity, so you can see the margin "
              f"around threshold={args.threshold}):")
        combined = list(zip(rows, sims, preds))
        combined.sort(key=lambda x: x[1])  # sort by similarity ascending
        for r, s, p in combined:
            label_str = "RELEVANT " if r["label"] == 1 else f"IRRELEVANT({r.get('negative_type','?'):4s})"
            correct = "correct" if p == r["label"] else "WRONG  "
            print(f"  sim={s:.4f}  [{label_str}]  {correct}  {r['thesis'][:40]}... | {r['news'][:55]}...")

        relevant_scores = [s for r, s in zip(rows, sims) if r["label"] == 1]
        irrelevant_scores = [s for r, s in zip(rows, sims) if r["label"] == 0]
        if relevant_scores and irrelevant_scores:
            min_rel = min(relevant_scores)
            max_irrel = max(irrelevant_scores)
            print(f"\nMargin check: lowest relevant score = {min_rel:.4f}, "
                  f"highest irrelevant score = {max_irrel:.4f}")
            gap = min_rel - max_irrel
            if gap > 0:
                print(f"  -> Clean separation, gap = {gap:.4f} (relevant and irrelevant scores don't overlap)")
            else:
                print(f"  -> WARNING: scores OVERLAP by {-gap:.4f} - threshold={args.threshold} happens to "
                      f"land in a gap, but this is fragile, not a robust separation")


if __name__ == "__main__":
    main()