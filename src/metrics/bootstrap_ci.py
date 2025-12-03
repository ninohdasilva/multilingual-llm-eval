"""Bootstrap confidence intervals for metrics."""

import numpy as np
from typing import List, Tuple


def bootstrap_ci(
    values: List[float], n_resamples: int = 1000, confidence: float = 0.95
) -> Tuple[float, float, float, float]:
    """
    Compute bootstrap confidence interval for a metric.

    Args:
        values: List of per-sentence metric values
        n_resamples: Number of bootstrap resamples (default: 1000)
        confidence: Confidence level (default: 0.95 for 95% CI)

    Returns:
        Tuple of (mean, std, lo95, hi95)
    """
    if len(values) == 0:
        return 0.0, 0.0, 0.0, 0.0

    values_array = np.array(values)
    mean = np.mean(values_array)
    std = np.std(values_array, ddof=1)  # Sample std

    # Bootstrap resampling
    n = len(values_array)
    bootstrap_means = []

    for _ in range(n_resamples):
        # Resample with replacement
        resample = np.random.choice(values_array, size=n, replace=True)
        bootstrap_means.append(np.mean(resample))

    bootstrap_means = np.array(bootstrap_means)

    # Compute confidence interval
    alpha = 1.0 - confidence
    lo95 = np.percentile(bootstrap_means, 100 * (alpha / 2))
    hi95 = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

    return mean, std, lo95, hi95


def compute_metric_with_ci(
    per_sentence_values: List[float],
) -> Tuple[float, float, float, float]:
    """
    Compute metric statistics with bootstrap CI.

    Convenience function that calls bootstrap_ci with default parameters.

    Args:
        per_sentence_values: List of per-sentence metric values

    Returns:
        Tuple of (mean, std, lo95, hi95)
    """
    return bootstrap_ci(per_sentence_values, n_resamples=1000, confidence=0.95)

