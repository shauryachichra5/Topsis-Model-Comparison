# TOPSIS Model Comparison

Ranks pretrained text-summarization models objectively, using **TOPSIS**
(Technique for Order of Preference by Similarity to Ideal Solution) — a
multi-criteria decision-making algorithm — instead of eyeballing a table of
metrics.

## The problem

Comparing summarization models isn't a single-number comparison. Model A
might have the best ROUGE score but also the highest redundancy (repetitive
text); Model B might be balanced across everything but best at nothing.
Picking "the best model" from a table of 5-6 competing metrics is exactly
the kind of multi-criteria ranking problem TOPSIS is built for.

## How it works

1. **Evaluate** each candidate model on a sample of the [SAMSUM](https://huggingface.co/datasets/samsum)
   dialogue-summarization dataset, scoring its generated summaries against
   human reference summaries on:
   - **BLEU** — n-gram precision overlap with the reference
   - **BERTScore (F1)** — semantic similarity using contextual embeddings
   - **ROUGE-1 / ROUGE-2 / ROUGE-L** — unigram, bigram, and longest-common-subsequence overlap
   - **Redundancy** — how repetitive the model's own generated summaries are (lower is better)
2. **Rank** the resulting model x metric matrix with TOPSIS: normalize each
   metric, apply weights, find the "ideal best" and "ideal worst" hypothetical
   models, then score every real model by its relative closeness to the ideal.

## Project structure

\`\`\`
src/
  topsis.py     # Generic TOPSIS algorithm — no dependency on summarization,
                 # reusable for ranking any alternatives x criteria matrix.
  metrics.py    # BLEU / BERTScore / ROUGE / redundancy computation.
  evaluate.py   # Loads SAMSUM, runs each model, builds the results dataframe.
main.py          # CLI entry point: runs evaluation, then TOPSIS ranking.
\`\`\`

\`topsis.py\` is deliberately decoupled from the summarization-specific code —
it only ever sees a plain numeric matrix, weights, and +/- impacts, so it
could rank laptops, vendors, or anything else with the same interface.

## Setup

\`\`\`bash
pip install -r requirements.txt
\`\`\`

## Usage

\`\`\`bash
python main.py                      # default: 100 sampled dialogues
python main.py --sample-size 50     # smaller sample, faster run
\`\`\`

## Customizing the ranking

Weights and impacts are set in \`main.py\`:

\`\`\`python
DEFAULT_WEIGHTS = [1, 1, 1, 1, 1, 1]       # equal importance per metric
DEFAULT_IMPACTS = [1, 1, 1, 1, 1, -1]      # +1 = higher is better, -1 = lower is better
\`\`\`

Adjust weights to prioritize, e.g., semantic quality (BERTScore) over raw
n-gram overlap (BLEU) for your use case.


Link For Kaggle notebook: https://www.kaggle.com/code/shauryachichra5/text-summarization-model-comparison-using-topsis
