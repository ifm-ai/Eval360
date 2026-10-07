import asyncio


class RateLimiter:
    """Async rate limiter (token bucket algorithm).

    Allows up to ``requests_per_minute`` requests per minute on average.
    The bucket starts full. Each :meth:`acquire` call consumes one permit,
    blocking until one is available.
    """

    def __init__(self, requests_per_minute: int):
        self._requests_per_minute = requests_per_minute
        self._rate = requests_per_minute / 60.0  # permits per second
        self._capacity = float(requests_per_minute)
        self._permits = float(requests_per_minute)  # start full
        self._last_refill: float | None = None
        self._condition = asyncio.Condition()

    def _refill(self, now: float) -> None:
        if self._last_refill is None:
            self._last_refill = now
            return
        elapsed = now - self._last_refill
        self._permits = min(self._capacity, self._permits + elapsed * self._rate)
        self._last_refill = now

    async def acquire(self) -> None:
        """Block until a permit is available, then consume one."""
        async with self._condition:
            while True:
                now = asyncio.get_running_loop().time()
                self._refill(now)
                if self._permits >= 1.0:
                    self._permits -= 1.0
                    return
                wait_time = (1.0 - self._permits) / self._rate
                try:
                    await asyncio.wait_for(
                        self._condition.wait(),
                        timeout=wait_time,
                    )
                except asyncio.TimeoutError:
                    pass

    async def refund(self) -> None:
        """Return a permit that was reserved but never used for a request."""
        async with self._condition:
            now = asyncio.get_running_loop().time()
            self._refill(now)
            self._permits = min(self._capacity, self._permits + 1.0)
            self._condition.notify_all()

    @property
    def requests_per_minute(self) -> int:
        """Configured quota used to detect conflicting registrations."""
        return self._requests_per_minute
