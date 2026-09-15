from dataclasses import dataclass

from celery import Celery
from redis import Redis


@dataclass(frozen=True)
class WorkerHealth:
    redis_available: bool
    celery_worker_available: bool


class InfrastructureHealthService:
    def __init__(
        self,
        *,
        redis_client: Redis,
        celery_app: Celery,
        timeout_seconds: float,
    ) -> None:
        self._redis = redis_client
        self._celery = celery_app
        self._timeout = timeout_seconds

    def check(self) -> WorkerHealth:
        try:
            redis_available = bool(self._redis.ping())
        except Exception:
            redis_available = False
        finally:
            self._redis.close()
        try:
            replies = self._celery.control.inspect(
                timeout=self._timeout
            ).ping()
            worker_available = bool(replies)
        except Exception:
            worker_available = False
        return WorkerHealth(
            redis_available=redis_available,
            celery_worker_available=worker_available,
        )
