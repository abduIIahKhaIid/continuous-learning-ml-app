from dataclasses import dataclass
from threading import Lock
from time import monotonic

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
        cache_seconds: float = 5.0,
    ) -> None:
        self._redis = redis_client
        self._celery = celery_app
        self._timeout = timeout_seconds
        self._cache_seconds = cache_seconds
        self._cached: WorkerHealth | None = None
        self._cache_expires_at = 0.0
        self._lock = Lock()

    def check(self) -> WorkerHealth:
        now = monotonic()
        if self._cached is not None and now < self._cache_expires_at:
            return self._cached
        with self._lock:
            now = monotonic()
            if self._cached is not None and now < self._cache_expires_at:
                return self._cached
            health = self._check_dependencies()
            self._cached = health
            self._cache_expires_at = now + self._cache_seconds
            return health

    def _check_dependencies(self) -> WorkerHealth:
        try:
            redis_available = bool(self._redis.ping())
        except Exception:
            redis_available = False
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
