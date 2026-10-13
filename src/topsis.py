"""
TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution).

A general-purpose multi-criteria ranking algorithm. Given a matrix of
alternatives x criteria, it ranks alternatives by how close they are to an
"ideal" solution (best value on every criterion) and how far from a
"worst" solution (worst value on every criterion).

This module is intentionally generic and has no dependency on the
summarization use case in evaluate.py -- it can rank anything, given a
numeric matrix, weights per criterion, and whether each criterion should
be maximized ("+") or minimized ("-").
"""

import numpy as np
import pandas as pd


def normalize(matrix: np.ndarray) -> np.ndarray:
    """Vector-normalize each column so columns are comparable regardless of scale."""
    return matrix / np.sqrt(np.sum(matrix ** 2, axis=0))


def weighted_normalize(norm_matrix: np.ndarray, weights: list) -> np.ndarray:
    """Apply per-criterion weights to the normalized matrix."""
    return norm_matrix * weights


def ideal_best_worst(weighted_norm_matrix: np.ndarray, impacts: list):
    """
    Compute the ideal-best and ideal-worst rows.

    impacts: +1 for criteria where higher is better, -1 where lower is better.
    """
    ideal_best = np.max(weighted_norm_matrix, axis=0) * impacts
    ideal_worst = np.min(weighted_norm_matrix, axis=0) * impacts
    return ideal_best, ideal_worst


def euclidean_distances(weighted_norm_matrix, ideal_best, ideal_worst):
    """Distance of every alternative from the ideal-best and ideal-worst rows."""
    dist_to_best = np.sqrt(np.sum((weighted_norm_matrix - ideal_best) ** 2, axis=1))
    dist_to_worst = np.sqrt(np.sum((weighted_norm_matrix - ideal_worst) ** 2, axis=1))
    return dist_to_best, dist_to_worst


def performance_score(dist_to_best, dist_to_worst):
    """
    Relative closeness to the ideal solution, in [0, 1].
    Higher score = better alternative.
    """
    return dist_to_worst / (dist_to_best + dist_to_worst)


def topsis(matrix: pd.DataFrame, weights: list, impacts: list):
    """
    Run full TOPSIS ranking.

    Args:
        matrix: rows = alternatives, columns = criteria (numeric only).
        weights: importance per criterion (same order as columns).
        impacts: +1 (maximize) or -1 (minimize) per criterion.

    Returns:
        (scores, ranks) -- ranks are 1 = best.
    """
    values = matrix.to_numpy(dtype=float)

    norm_matrix = normalize(values)
    weighted_norm_matrix = weighted_normalize(norm_matrix, weights)
    ideal_best, ideal_worst = ideal_best_worst(weighted_norm_matrix, impacts)
    dist_to_best, dist_to_worst = euclidean_distances(
        weighted_norm_matrix, ideal_best, ideal_worst
    )
    scores = performance_score(dist_to_best, dist_to_worst)

    # rank 1 = highest score
    ranks = pd.Series(scores).rank(ascending=False, method="min").astype(int).to_numpy()

    return scores, ranks
