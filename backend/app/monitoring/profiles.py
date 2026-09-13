from typing import Any

import numpy as np
from numpy.typing import NDArray

FEATURE_NAMES = ("feature_1", "feature_2", "feature_3")


def build_reference_profiles(
    features: NDArray[np.float64], *, bin_count: int = 10
) -> list[dict[str, Any]]:
    """Create compact, JSON-safe summaries from the rows used to fit a model."""
    if features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES):
        raise ValueError("Expected a two-dimensional, three-feature matrix.")
    profiles: list[dict[str, Any]] = []
    for index, feature_name in enumerate(FEATURE_NAMES):
        values = np.asarray(features[:, index], dtype=np.float64)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            raise ValueError(f"{feature_name} has no finite training values.")
        counts, bins = np.histogram(finite, bins=bin_count)
        profiles.append(
            {
                "feature_name": feature_name,
                "sample_count": int(finite.size),
                "mean": float(np.mean(finite)),
                "std": float(np.std(finite)),
                "min": float(np.min(finite)),
                "max": float(np.max(finite)),
                "median": float(np.median(finite)),
                "q25": float(np.quantile(finite, 0.25)),
                "q75": float(np.quantile(finite, 0.75)),
                "histogram_bins": bins.tolist(),
                "histogram_counts": counts.tolist(),
            }
        )
    return profiles

