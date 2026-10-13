"""
Runs each candidate summarization model over a sample of the SAMSUM dataset
and collects quality metrics for each, ready to be fed into TOPSIS.
"""

import warnings

import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import pipeline

from src.metrics import calc_metrics

warnings.filterwarnings("ignore")

DEFAULT_MODELS = [
    "philschmid/bart-large-cnn-samsum",
    "facebook/bart-large-cnn",
    "philschmid/distilbart-cnn-12-6-samsum",
    "knkarthick/MEETING_SUMMARY",
    "google/bigbird-pegasus-large-arxiv",
]


def load_sample(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Load n random dialogues (with reference summaries) from the SAMSUM test set."""
    dataset = load_dataset("samsum")
    df = dataset["test"].to_pandas()
    return df.sample(n=n, replace=False, random_state=seed).reset_index(drop=True)


def run_model(model_name: str, df: pd.DataFrame) -> dict:
    """Generate summaries with one model and score them against references."""
    pipe = pipeline("summarization", model=model_name)

    predictions = [
        pipe(dialogue, max_length=130, min_length=30, truncation=True)[0]["summary_text"]
        for dialogue in df["dialogue"]
    ]

    return calc_metrics(df["summary"].tolist(), predictions)


def evaluate_models(models: list = None, sample_size: int = 100) -> pd.DataFrame:
    """
    Run every model in `models` over a sample of SAMSUM and return a
    dataframe of per-model metrics, ready for TOPSIS ranking.
    """
    models = models or DEFAULT_MODELS
    df = load_sample(n=sample_size)

    rows = []
    for model_name in models:
        print(f"Evaluating: {model_name}")
        metrics = run_model(model_name, df)
        rows.append({"model": model_name, **{k: np.round(v, 3) for k, v in metrics.items()}})

    return pd.DataFrame(rows)
