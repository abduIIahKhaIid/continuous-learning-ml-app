from celery import Celery

from app.core.config import get_settings

settings = get_settings()

celery_app = Celery(
    "continuous_learning_ml",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["app.workers.tasks"],
)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    task_default_queue=settings.training_queue_name,
    task_routes={
        "app.workers.tasks.run_retraining_task": {
            "queue": settings.training_queue_name
        }
    },
    task_always_eager=settings.celery_task_always_eager,
    task_store_eager_result=settings.celery_task_always_eager,
    task_soft_time_limit=settings.training_task_soft_time_limit_seconds,
    task_time_limit=settings.training_task_time_limit_seconds,
    broker_connection_retry_on_startup=True,
    broker_connection_timeout=2,
    task_publish_retry=False,
)
