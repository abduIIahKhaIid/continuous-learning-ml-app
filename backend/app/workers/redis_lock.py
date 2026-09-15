from uuid import uuid4

from redis import Redis

RELEASE_IF_OWNER_SCRIPT = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
else
    return 0
end
"""


class RedisTrainingLock:
    """An expiring Redis lock that only its unique owner may release."""

    def __init__(
        self,
        client: Redis,
        *,
        key: str = "ml:training:lock",
        timeout_seconds: int,
        token: str | None = None,
    ) -> None:
        self._client = client
        self.key = key
        self.timeout_seconds = timeout_seconds
        self.token = token or uuid4().hex
        self.acquired = False

    def acquire(self) -> bool:
        self.acquired = bool(
            self._client.set(
                self.key,
                self.token,
                nx=True,
                ex=self.timeout_seconds,
            )
        )
        return self.acquired

    def release(self) -> bool:
        if not self.acquired:
            return False
        released = bool(
            self._client.eval(
                RELEASE_IF_OWNER_SCRIPT, 1, self.key, self.token
            )
        )
        self.acquired = False
        return released

