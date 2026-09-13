from app.core.config import get_settings
from app.database.init_db import init_db
from app.database.session import SessionLocal
from app.ml.config import TrainingConfig
from app.ml.service import run_initial_training


def _format_metric(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.4f}"


def main() -> int:
    settings = get_settings()
    config = TrainingConfig.from_settings(settings)
    init_db()

    print("Training started")
    try:
        with SessionLocal() as session:
            summary = run_initial_training(session, config)
    except Exception as error:
        print(f"Training failed: {error}")
        return 1

    print(f"Labelled samples: {summary.training_sample_count}")
    print(f"Train samples: {summary.train_sample_count}")
    print(f"Test samples: {summary.test_sample_count}")
    print("Model: LogisticRegression")
    print(f"Accuracy: {_format_metric(summary.metrics.accuracy)}")
    print(f"Precision: {_format_metric(summary.metrics.precision)}")
    print(f"Recall: {_format_metric(summary.metrics.recall)}")
    print(f"F1: {_format_metric(summary.metrics.f1_score)}")
    print(f"ROC-AUC: {_format_metric(summary.metrics.roc_auc)}")
    print(f"Saved model: {summary.model_path}")
    print(f"Model version: {summary.model_version}")
    print(f"Training batch: {summary.training_batch_id}")
    print("Training completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
