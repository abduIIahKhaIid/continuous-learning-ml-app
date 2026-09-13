from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.main import app
from app.models.prediction import Prediction
from app.models.sample import Sample
from app.models.training_run import TrainingRun
from app.repositories.data import SqlAlchemyDataRepository
from app.repositories.prediction import SqlAlchemyPredictionRepository
from app.schemas.prediction import PredictionFeedbackRequest
from app.services.feedback import FeedbackDatabaseError, FeedbackService

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
async def feedback_client(
    db_session: Session,
) -> AsyncIterator[AsyncClient]:
    async def override_get_db() -> AsyncIterator[Session]:
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            yield client
    finally:
        app.dependency_overrides.clear()


def create_prediction(
    session: Session,
    *,
    predicted_class: int = 1,
    feature_1: float = 1.0,
) -> Prediction:
    if session.get(TrainingRun, 1) is None:
        session.add(
            TrainingRun(
                training_batch_id="feedback_batch",
                model_version="model_feedback",
                algorithm="LogisticRegression",
                training_sample_count=20,
                random_state=42,
                status="completed",
            )
        )
        session.flush()
    prediction = Prediction(
        feature_1=feature_1,
        feature_2=2.0,
        feature_3=3.0,
        predicted_class=predicted_class,
        prediction_probability=0.75,
        model_version="model_feedback",
    )
    session.add(prediction)
    session.commit()
    return prediction


async def submit_feedback(
    client: AsyncClient,
    prediction_id: int,
    actual_label: object,
):
    return await client.patch(
        f"/api/predictions/{prediction_id}/feedback",
        json={"actual_label": actual_label},
    )


async def test_successful_feedback_is_stored_and_creates_verified_sample(
    feedback_client: AsyncClient,
    db_session: Session,
) -> None:
    prediction = create_prediction(db_session, predicted_class=1)

    response = await submit_feedback(feedback_client, prediction.id, 0)

    assert response.status_code == 200
    assert response.json() == {
        "prediction_id": prediction.id,
        "predicted_class": 1,
        "actual_label": 0,
        "was_correct": False,
        "feedback_received": True,
        "model_version": "model_feedback",
        "updated_at": response.json()["updated_at"],
    }

    db_session.expire_all()
    stored_prediction = db_session.get(Prediction, prediction.id)
    sample = db_session.scalar(
        select(Sample).where(Sample.source_prediction_id == prediction.id)
    )
    assert stored_prediction is not None
    assert stored_prediction.actual_label == 0
    assert stored_prediction.feedback_received is True
    assert sample is not None
    assert (sample.feature_1, sample.feature_2, sample.feature_3) == (
        prediction.feature_1,
        prediction.feature_2,
        prediction.feature_3,
    )
    assert sample.label == 0
    assert sample.label != prediction.predicted_class
    assert sample.source_prediction_id == prediction.id
    assert sample.used_for_training is False
    assert sample.training_batch_id is None
    assert sample.model_version is None


@pytest.mark.parametrize(
    ("predicted_class", "actual_label", "was_correct"),
    [(1, 1, True), (1, 0, False)],
)
async def test_feedback_reports_observed_correctness(
    feedback_client: AsyncClient,
    db_session: Session,
    predicted_class: int,
    actual_label: int,
    was_correct: bool,
) -> None:
    prediction = create_prediction(
        db_session, predicted_class=predicted_class
    )

    response = await submit_feedback(
        feedback_client, prediction.id, actual_label
    )

    assert response.status_code == 200
    assert response.json()["was_correct"] is was_correct


async def test_nonexistent_prediction_feedback_returns_404(
    feedback_client: AsyncClient,
) -> None:
    response = await submit_feedback(feedback_client, 999, 1)

    assert response.status_code == 404
    assert response.json() == {"detail": "Prediction not found."}


@pytest.mark.parametrize("invalid_label", [-1, 2, 0.5, "1", True, None])
async def test_invalid_actual_label_is_rejected(
    feedback_client: AsyncClient,
    db_session: Session,
    invalid_label: object,
) -> None:
    prediction = create_prediction(db_session)

    response = await submit_feedback(
        feedback_client, prediction.id, invalid_label
    )

    assert response.status_code == 422
    db_session.expire_all()
    assert db_session.get(Prediction, prediction.id).feedback_received is False


async def test_duplicate_feedback_is_immutable_and_creates_one_sample(
    feedback_client: AsyncClient,
    db_session: Session,
) -> None:
    prediction = create_prediction(db_session)
    first = await submit_feedback(feedback_client, prediction.id, 1)
    second = await submit_feedback(feedback_client, prediction.id, 0)

    assert first.status_code == 200
    assert second.status_code == 409
    assert second.json() == {
        "detail": (
            "Ground-truth feedback has already been submitted for this "
            "prediction."
        )
    }
    db_session.expire_all()
    assert db_session.get(Prediction, prediction.id).actual_label == 1
    assert db_session.scalar(select(func.count(Sample.id))) == 1


async def test_source_prediction_database_uniqueness_is_enforced(
    feedback_client: AsyncClient,
    db_session: Session,
) -> None:
    prediction = create_prediction(db_session)
    assert (await submit_feedback(feedback_client, prediction.id, 1)).status_code == 200
    db_session.add(
        Sample(
            feature_1=1,
            feature_2=2,
            feature_3=3,
            label=0,
            source_prediction_id=prediction.id,
        )
    )

    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()


def test_source_prediction_foreign_key_is_enforced(
    db_session: Session,
) -> None:
    db_session.add(
        Sample(
            feature_1=1,
            feature_2=2,
            feature_3=3,
            label=0,
            source_prediction_id=999,
        )
    )

    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()


async def test_feedback_status_before_and_after_submission(
    feedback_client: AsyncClient,
    db_session: Session,
) -> None:
    prediction = create_prediction(db_session, predicted_class=0)

    pending = await feedback_client.get(
        f"/api/predictions/{prediction.id}/feedback"
    )
    assert pending.status_code == 200
    assert pending.json() == {
        "prediction_id": prediction.id,
        "predicted_class": 0,
        "actual_label": None,
        "feedback_received": False,
        "was_correct": None,
        "model_version": "model_feedback",
    }

    # The shared test session has a read transaction; close it as a real
    # request-scoped session would be closed before the following request.
    db_session.rollback()
    assert (await submit_feedback(feedback_client, prediction.id, 0)).status_code == 200
    saved = await feedback_client.get(
        f"/api/predictions/{prediction.id}/feedback"
    )
    assert saved.json()["actual_label"] == 0
    assert saved.json()["feedback_received"] is True
    assert saved.json()["was_correct"] is True


async def test_feedback_summary_uses_only_verified_ground_truth(
    feedback_client: AsyncClient,
    db_session: Session,
) -> None:
    correct = create_prediction(db_session, predicted_class=1)
    incorrect = create_prediction(db_session, predicted_class=1)
    create_prediction(db_session, predicted_class=0)
    assert (await submit_feedback(feedback_client, correct.id, 1)).status_code == 200
    assert (await submit_feedback(feedback_client, incorrect.id, 0)).status_code == 200

    response = await feedback_client.get("/api/predictions/feedback/summary")

    assert response.status_code == 200
    assert response.json() == {
        "total_predictions": 3,
        "feedback_received": 2,
        "feedback_pending": 1,
        "correct_predictions": 1,
        "incorrect_predictions": 1,
        "verified_accuracy": 0.5,
    }


async def test_training_data_filters_labelled_and_unused_records(
    feedback_client: AsyncClient,
    db_session: Session,
) -> None:
    db_session.add_all(
        [
            Sample(feature_1=1, feature_2=1, feature_3=1, label=None),
            Sample(
                feature_1=2,
                feature_2=2,
                feature_3=2,
                label=0,
                used_for_training=True,
            ),
            Sample(feature_1=3, feature_2=3, feature_3=3, label=1),
        ]
    )
    db_session.commit()

    response = await feedback_client.get(
        "/api/training-data",
        params={"labelled_only": True, "unused_only": True},
    )

    assert response.status_code == 200
    assert [row["label"] for row in response.json()] == [1]
    assert response.json()[0]["used_for_training"] is False
    assert response.json()[0]["source_prediction_id"] is None


async def test_corrupt_prediction_features_are_not_added_to_training_data(
    feedback_client: AsyncClient,
    db_session: Session,
) -> None:
    prediction = create_prediction(db_session, feature_1=float("inf"))

    response = await submit_feedback(feedback_client, prediction.id, 1)

    assert response.status_code == 422
    db_session.expire_all()
    assert db_session.get(Prediction, prediction.id).feedback_received is False
    assert db_session.scalar(select(func.count(Sample.id))) == 0


def test_feedback_transaction_rolls_back_when_sample_creation_fails(
    db_session: Session,
) -> None:
    prediction = create_prediction(db_session)

    class FailingDataRepository(SqlAlchemyDataRepository):
        def create_verified(self, **_: object) -> Sample:
            raise SQLAlchemyError("simulated sample write failure")

    service = FeedbackService(
        session=db_session,
        prediction_repository=SqlAlchemyPredictionRepository(db_session),
        data_repository=FailingDataRepository(db_session),
    )

    with pytest.raises(FeedbackDatabaseError):
        service.submit(
            prediction.id,
            PredictionFeedbackRequest(actual_label=1),
        )

    db_session.expire_all()
    stored = db_session.get(Prediction, prediction.id)
    assert stored is not None
    assert stored.actual_label is None
    assert stored.feedback_received is False
    assert db_session.scalar(select(func.count(Sample.id))) == 0
