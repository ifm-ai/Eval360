"""
Tests for RateLimiter (scheduler/rate_limiter.py).
"""
import asyncio

import pytest

from scheduler.rate_limiter import RateLimiter


class TestRateLimiter:
    @pytest.mark.asyncio
    async def test_immediate_acquire_within_capacity(self):
        """First N requests within capacity should complete without blocking."""
        limiter = RateLimiter(requests_per_minute=60)
        start = asyncio.get_event_loop().time()
        for _ in range(10):
            await limiter.acquire()
        elapsed = asyncio.get_event_loop().time() - start
        assert elapsed < 0.5, f"Should not block for 10 requests at 60 RPM, got {elapsed:.2f}s"

    @pytest.mark.asyncio
    async def test_capacity_starts_full(self):
        """Limiter starts with full capacity bucket."""
        limiter = RateLimiter(requests_per_minute=5)
        start = asyncio.get_event_loop().time()
        for _ in range(5):
            await limiter.acquire()
        elapsed = asyncio.get_event_loop().time() - start
        assert elapsed < 0.5, f"5 requests at 5 RPM capacity should not block, got {elapsed:.2f}s"

    @pytest.mark.asyncio
    async def test_blocks_when_capacity_exceeded(self):
        """After exhausting capacity, next request should block."""
        limiter = RateLimiter(requests_per_minute=60)
        # Exhaust permits
        for _ in range(60):
            await limiter.acquire()
        # The 61st should block (need to wait for refill)
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(limiter.acquire(), timeout=0.5)

    @pytest.mark.asyncio
    async def test_refills_over_time(self):
        """Permits refill at the correct rate."""
        # 120 RPM = 2 per second
        limiter = RateLimiter(requests_per_minute=120)
        # Drain all permits
        for _ in range(120):
            await limiter.acquire()
        # Wait 0.6 seconds → should get ~1 permit (2 per second * 0.6 = 1.2 permits)
        await asyncio.sleep(0.6)
        # Should be able to acquire at least 1 more without timeout
        await asyncio.wait_for(limiter.acquire(), timeout=0.2)

    @pytest.mark.asyncio
    async def test_concurrent_acquires_serialized(self):
        """Multiple concurrent acquires complete correctly."""
        limiter = RateLimiter(requests_per_minute=600)
        results = []
        async def acquire_one(i):
            await limiter.acquire()
            results.append(i)
        await asyncio.gather(*[acquire_one(i) for i in range(20)])
        assert len(results) == 20

    @pytest.mark.asyncio
    async def test_refund_restores_an_unused_permit(self):
        limiter = RateLimiter(requests_per_minute=1)
        await limiter.acquire()

        await limiter.refund()

        await asyncio.wait_for(limiter.acquire(), timeout=0.1)

    @pytest.mark.asyncio
    async def test_refund_wakes_exactly_one_waiter_without_delay(self):
        limiter = RateLimiter(requests_per_minute=1)
        await limiter.acquire()

        first_waiter = asyncio.create_task(limiter.acquire())
        await asyncio.sleep(0)
        refund_task = asyncio.create_task(limiter.refund())
        second_waiter = None
        try:
            completed, _ = await asyncio.wait(
                {refund_task},
                timeout=0.1,
            )
            assert refund_task in completed, (
                "an unused permit refund must not queue behind a refill wait"
            )

            await asyncio.wait_for(first_waiter, timeout=0.1)
            second_waiter = asyncio.create_task(limiter.acquire())
            completed, _ = await asyncio.wait(
                {second_waiter},
                timeout=0.01,
            )
            assert second_waiter not in completed
        finally:
            pending = [first_waiter, refund_task]
            if second_waiter is not None:
                pending.append(second_waiter)
            for task in pending:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
