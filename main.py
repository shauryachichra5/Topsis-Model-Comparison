"""
Ranks pretrained text-summarization models on SAMSUM dialogues using
TOPSIS multi-criteria ranking over BLEU, BERTScore, ROUGE-1/2/L and
redundancy.

Usage:
    python main.py
    python main.py --sample-size 50
"""

import argparse

from src.evaluate import evaluate_models, DEFAULT_MODELS
from src.topsis import topsis

# Equal weight on every metric by default; redundancy is "lower is better",
# everything else is "higher is better".
DEFAULT_WEIGHTS = [1, 1, 1, 1, 1, 1]
DEFAULT_IMPACTS = [1, 1, 1, 1, 1, -1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-size", type=int, default=100,
        help="Number of SAMSUM dialogues to evaluate each model on.",
    )
    args = parser.parse_args()

    results_df = evaluate_models(models=DEFAULT_MODELS, sample_size=args.sample_size)

    metric_cols = [c for c in results_df.columns if c != "model"]
    scores, ranks = topsis(results_df[metric_cols], DEFAULT_WEIGHTS, DEFAULT_IMPACTS)

    results_df["topsis_score"] = scores.round(3)
    results_df["rank"] = ranks

    results_df = results_df.sort_values("rank")
    print("\nFinal Ranking:\n")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
