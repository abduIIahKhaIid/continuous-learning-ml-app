from typing import Literal, Protocol

import numpy as np

DriftStatus = Literal["stable", "warning", "critical", "insufficient_data"]


class HistogramProfile(Protocol):
    histogram_bins: list[float]
    histogram_counts: list[int]


def calculate_psi(
    profile: HistogramProfile,
    current_values: list[float],
    *,
    epsilon: float = 1e-6,
) -> float:
    """Calculate PSI with reference bins and smoothing for empty bins."""
    bins = np.asarray(profile.histogram_bins, dtype=np.float64).copy()
    reference_counts = np.asarray(
        profile.histogram_counts, dtype=np.float64
    )
    current = np.asarray(current_values, dtype=np.float64)
    current = current[np.isfinite(current)]
    if bins.size < 2 or reference_counts.size != bins.size - 1:
        raise ValueError("Reference histogram is invalid.")
    if reference_counts.sum() <= 0 or current.size == 0:
        return 0.0
    bins[0], bins[-1] = -np.inf, np.inf
    current_counts, _ = np.histogram(current, bins=bins)
    expected = np.clip(reference_counts / reference_counts.sum(), epsilon, None)
    actual = np.clip(current_counts / current_counts.sum(), epsilon, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def classify_psi(
    psi: float, *, warning_threshold: float, critical_threshold: float
) -> DriftStatus:
    if psi >= critical_threshold:
        return "critical"
    if psi >= warning_threshold:
        return "warning"
    return "stable"


def worst_status(statuses: list[DriftStatus]) -> DriftStatus:
    order = {
        "insufficient_data": 0,
        "stable": 1,
        "warning": 2,
        "critical": 3,
    }
    usable = [status for status in statuses if status != "insufficient_data"]
    if not usable:
        return "insufficient_data"
    return max(usable, key=order.__getitem__)

