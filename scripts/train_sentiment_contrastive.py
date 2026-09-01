"""
Fine-tunes a sentence-transformers model to cluster financial text by
sentiment (SimCSE-style contrastive learning), using
data/sentiment/train_triplets.jsonl (anchor=sentence, positive=same-
sentiment sentence, negative=different-sentiment sentence).

Same loss/approach as scripts/train_contrastive_model.py (Week 3's
relevance model) - MultipleNegativesRankingLoss pulls anchor+positive
embeddings together and pushes anchor+negative apart, in-batch, without
needing a separate classifier head. Reusing the same recipe here means
Member 1's swap-in pattern (relevance_scorer.py's _get_model()) works
identically for sentiment.

Usage:
    python train_sentiment_contrastive.py --epochs 2 --batch-size 16
"""

import argparse
from pathlib import Path
import json

from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader


def load_triplets(path: Path) -> list:
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def to_examples(rows: list) -> list:
    return [InputExample(texts=[r["anchor"], r["positive"], r["negative"]]) for r in rows]


def main():
    parser = argparse.ArgumentParser(description="Train contrastive sentiment embedding model")
    parser.add_argument("--data-dir", default="data/sentiment")
    parser.add_argument("--model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--output-dir", default="models/contrastive_sentiment_v1")
    parser.add_argument("--epochs", type=int, default=2,
                         help="2 epochs to keep this realistic for a 3-hour session - "
                              "Week 3's relevance model used 4 on a much smaller dataset (192 "
                              "triplets vs 7752 here), so 2 epochs here is comparable total "
                              "gradient steps, not a corner-cut.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    train_rows = load_triplets(Path(args.data_dir) / "train_triplets.jsonl")
    print(f"Loaded {len(train_rows)} training triplets")

    model = SentenceTransformer(args.model_name)
    train_examples = to_examples(train_rows)
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    warmup_steps = max(1, int(len(train_dataloader) * args.epochs * 0.1))

    print(f"Training for {args.epochs} epoch(s), batch_size={args.batch_size}, "
          f"{len(train_dataloader)} batches/epoch ...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        show_progress_bar=True,
        output_path=args.output_dir,
    )

    print(f"\nDone. Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
