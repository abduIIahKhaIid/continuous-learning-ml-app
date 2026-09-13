from dataclasses import asdict, dataclass

from app.ml.evaluator import EvaluationMetrics


@dataclass(frozen=True)
class PromotionDecision:
    promote: bool
    reason: str


def metrics_to_dict(metrics: EvaluationMetrics) -> dict[str, object]:
    return asdict(metrics)


def decide_promotion(
    *,
    candidate: EvaluationMetrics,
    active: EvaluationMetrics | None,
    primary_metric: str,
    minimum_improvement: float,
    minimum_acceptable_f1: float,
) -> PromotionDecision:
    if candidate.f1_score < minimum_acceptable_f1:
        return PromotionDecision(
            promote=False,
            reason=(
                f"Candidate F1 {candidate.f1_score:.4f} is below the "
                f"minimum {minimum_acceptable_f1:.4f}."
            ),
        )

    candidate_value = getattr(candidate, primary_metric, None)
    if candidate_value is None:
        return PromotionDecision(
            promote=False,
            reason=(
                f"Candidate primary metric '{primary_metric}' is unavailable."
            ),
        )

    if active is None:
        return PromotionDecision(
            promote=True,
            reason="No active model exists and the candidate meets quality requirements.",
        )

    active_value = getattr(active, primary_metric, None)
    if active_value is None:
        return PromotionDecision(
            promote=False,
            reason=(
                f"Active-model primary metric '{primary_metric}' is unavailable "
                "for a fair comparison."
            ),
        )

    required_value = active_value + minimum_improvement
    if candidate_value >= required_value:
        return PromotionDecision(
            promote=True,
            reason=(
                f"Candidate {primary_metric} {candidate_value:.4f} meets the "
                f"required {required_value:.4f}."
            ),
        )
    return PromotionDecision(
        promote=False,
        reason=(
            f"Candidate {primary_metric} {candidate_value:.4f} does not meet "
            f"the required {required_value:.4f}."
        ),
    )
