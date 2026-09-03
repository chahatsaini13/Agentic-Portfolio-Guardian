"""
Evaluates the restored keyword+embedding cross-check (ADR 0009) against
the same held-out val_sentences.jsonl (17 sentences) evaluate_redflag_model.py
used for the embedding-only version, so the two numbers are directly
comparable - same data, same threshold, only the detection logic changed.

Usage:
    python scripts/evaluate_redflag_signal_agreement.py
"""

import json
from pathlib import Path
from collections import Counter

import sys
sys.path.insert(0, '.')
from src.agents.redflag_detector import check_signal_agreement, redflag_score, keyword_redflag_score

VAL_PATH = Path("data/redflag/val_sentences.jsonl")


def load_jsonl(path: Path) -> list:
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def main():
    val_rows = load_jsonl(VAL_PATH)
    print(f"Evaluating on {len(val_rows)} held-out sentences\n")

    correct_embedding_only = 0
    correct_combined = 0
    agreement_counts = Counter()
    results = []

    for r in val_rows:
        text = r["text"]
        true_label = r["label"]

        agreement = check_signal_agreement(text)
        pred_combined = "red_flag" if agreement["is_red_flag"] else "routine"
        pred_embedding_only = "red_flag" if agreement["embedding_flagged"] else "routine"

        correct_embedding_only += (pred_embedding_only == true_label)
        correct_combined += (pred_combined == true_label)
        agreement_counts[agreement["signal_agreement"]] += 1

        results.append((text, true_label, pred_embedding_only, pred_combined,
                         agreement["embedding_score"], agreement["signal_agreement"]))

    n = len(val_rows)
    print(f"=== ACCURACY (embedding-only, same logic as ADR 0008): "
          f"{correct_embedding_only / n:.4f} ({correct_embedding_only}/{n}) ===")
    print(f"=== ACCURACY (combined, embedding OR keyword - ADR 0009): "
          f"{correct_combined / n:.4f} ({correct_combined}/{n}) ===\n")

    print("Signal agreement breakdown across all 17 val sentences:")
    for label in ["both", "embedding_only", "keyword_only", "neither"]:
        print(f"  {label:16s} {agreement_counts.get(label, 0)}")

    print("\nPer-sentence results:")
    for text, true_label, pred_emb, pred_comb, emb_score, agreement in sorted(results, key=lambda x: x[4]):
        emb_mark = "OK  " if pred_emb == true_label else "WRONG"
        comb_mark = "OK  " if pred_comb == true_label else "WRONG"
        changed = " <-- combined differs from embedding-only" if pred_emb != pred_comb else ""
        print(f"  emb_score={emb_score:.4f}  true={true_label:10s}  "
              f"embedding-only=[{emb_mark}]  combined=[{comb_mark}]  agreement={agreement:16s}  "
              f"{text[:55]}{changed}")

    # Did the combined logic flip any prediction relative to embedding-only?
    flips = [r for r in results if r[2] != r[3]]
    print(f"\nPredictions changed by adding the keyword cross-check: {len(flips)}")
    for text, true_label, pred_emb, pred_comb, emb_score, agreement in flips:
        direction = "routine->red_flag" if pred_comb == "red_flag" else "red_flag->routine"
        correctness = "IMPROVED" if pred_comb == true_label and pred_emb != true_label else (
            "WORSENED" if pred_emb == true_label and pred_comb != true_label else "NEUTRAL")
        print(f"  [{correctness}] {direction}  true={true_label}  {text[:55]}")


if __name__ == "__main__":
    main()