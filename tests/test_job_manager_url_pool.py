"""
Tests for the JobManager URL pool (scheduler/job.py).

JobManager is the in-memory pool of live VLLM endpoint URLs.  The scheduler's
handle_job_update drives state changes; OpenAIConnection calls get_live_url()
per request.

Covers:
  - register_live_url: adds URL and wakes waiters
  - get_live_url: blocks until a URL is available, returns one at random
  - evict_url: removes a URL; after last URL is evicted get_live_url blocks again
  - is_url_live: reflects the current pool state
  - get_live_urls: snapshot of the pool
  - edge cases: duplicate registration, unknown model/URL, multi-model isolation
"""

import asyncio
import pytest

from scheduler.job import JobManager

URL_A = "http://node-a:8000"
URL_B = "http://node-b:8000"
MODEL = "my-model"


# ---------------------------------------------------------------------------
# register_live_url
# ---------------------------------------------------------------------------

class TestRegisterLiveUrl:

    @pytest.mark.asyncio
    async def test_registers_url(self):
        jm = JobManager()
        await jm.register_live_url(MODEL, URL_A)
        assert URL_A in jm.get_live_urls(MODEL)

    @pytest.mark.asyncio
    async def test_registering_same_url_twice_does_not_duplicate(self):
        jm = JobManager()
        await jm.register_live_url(MODEL, URL_A)
        await jm.register_live_url(MODEL, URL_A)
        assert jm.get_live_urls(MODEL).count(URL_A) == 1

    @pytest.mark.asyncio
    async def test_register_two_urls(self):
        jm = JobManager()
        await jm.register_live_url(MODEL, URL_A)
        await jm.register_live_url(MODEL, URL_B)
        urls = jm.get_live_urls(MODEL)
        assert URL_A in urls
        assert URL_B in urls

    @pytest.mark.asyncio
    async def test_register_wakes_blocked_get_live_url(self):
        """get_live_url blocks when pool is empty; registering a URL unblocks it."""
        jm = JobManager()

        async def delayed_register():
            await asyncio.sleep(0.01)
            await jm.register_live_url(MODEL, URL_A)

        asyncio.create_task(delayed_register())
        url = await asyncio.wait_for(jm.get_live_url(MODEL), timeout=1.0)
        assert url == URL_A


# ---------------------------------------------------------------------------
# get_live_url
# ---------------------------------------------------------------------------

class TestGetLiveUrl:

    @pytest.mark.asyncio
    async def test_returns_url_when_already_registered(self):
        jm = JobManager()
        await jm.register_live_url(MODEL, URL_A)
        url = await jm.get_live_url(MODEL)
        assert url == URL_A

    @pytest.mark.asyncio
    async def test_blocks_indefinitely_when_pool_is_empty(self):
        """get_live_url should not return when no URL is registered."""
        jm = JobManager()
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(jm.get_live_url(MODEL), timeout=0.05)

    @pytest.mark.asyncio
    async def test_returns_one_of_the_registered_urls(self):
        jm = JobManager()
        await jm.register_live_url(MODEL, URL_A)
        await jm.register_live_url(MODEL, URL_B)
        url = await jm.get_live_url(MODEL)
        assert url in (URL_A, URL_B)

    @pytest.mark.asyncio
    async def test_multiple_waiters_all_unblocked_on_register(self):
        """Multiple concurrent get_live_url calls should all return once a URL is registered."""
        jm = JobManager()

        async def get():
            return await jm.get_live_url(MODEL)

        tasks = [asyncio.create_task(get()) for _ in range(5)]
        await asyncio.sleep(0.01)
        await jm.register_live_url(MODEL, URL_A)

        results = await asyncio.gather(*tasks)
        assert all(r == URL_A for r in results)

    @pytest.mark.asyncio
    async def test_blocks_again_after_all_urls_evicted(self):
        """After evicting the last URL, get_live_url must block until a new one arrives."""
        jm = JobManager()
        await jm.register_live_url(MODEL, URL_A)
        await jm.evict_url(MODEL, URL_A)

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(jm.get_live_url(MODEL), timeout=0.05)

    @pytest.mark.asyncio
    async def test_returns_url_after_re_register_following_eviction(self):
        jm = JobManager()
        await jm.register_live_url(MODEL, URL_A)
        await jm.evict_url(MODEL, URL_A)

        async def delayed_register():
            await asyncio.sleep(0.01)
            await jm.register_live_url(MODEL, URL_B)

        asyncio.create_task(delayed_register())
        url = await asyncio.wait_for(jm.get_live_url(MODEL), timeout=1.0)
        assert url == URL_B


# ---------------------------------------------------------------------------
# evict_url
# ---------------------------------------------------------------------------

class TestEvictUrl:

    @pytest.mark.asyncio
    async def test_removes_url_from_pool(self):
        jm = JobManager()
        await jm.register_live_url(MODEL, URL_A)
        await jm.evict_url(MODEL, URL_A)
        assert URL_A not in jm.get_live_urls(MODEL)

    @pytest.mark.asyncio
    async def test_evicting_one_of_two_leaves_the_other(self):
        jm = JobManager()
        await jm.register_live_url(MODEL, URL_A)
        await jm.register_live_url(MODEL, URL_B)
        await jm.evict_url(MODEL, URL_A)
        urls = jm.get_live_urls(MODEL)
        assert URL_A not in urls
        assert URL_B in urls

    @pytest.mark.asyncio
    async def test_evict_unknown_model_is_noop(self):
        """Evicting a URL for a model that was never registered must not raise."""
        jm = JobManager()
        await jm.evict_url("nonexistent-model", URL_A)  # should not raise

    @pytest.mark.asyncio
    async def test_evict_unknown_url_is_noop(self):
        """Evicting a URL that was never registered must not raise."""
        jm = JobManager()
        await jm.register_live_url(MODEL, URL_A)
        await jm.evict_url(MODEL, URL_B)  # URL_B was never registered
        assert jm.get_live_urls(MODEL) == [URL_A]

    @pytest.mark.asyncio
    async def test_evict_twice_is_noop(self):
        """Evicting the same URL twice must not raise."""
        jm = JobManager()
        await jm.register_live_url(MODEL, URL_A)
        await jm.evict_url(MODEL, URL_A)
        await jm.evict_url(MODEL, URL_A)  # second evict — must not raise


# ---------------------------------------------------------------------------
# is_url_live
# ---------------------------------------------------------------------------

class TestIsUrlLive:

    @pytest.mark.asyncio
    async def test_true_for_registered_url(self):
        jm = JobManager()
        await jm.register_live_url(MODEL, URL_A)
        assert jm.is_url_live(URL_A) is True

    @pytest.mark.asyncio
    async def test_false_for_never_registered_url(self):
        jm = JobManager()
        assert jm.is_url_live(URL_A) is False

    @pytest.mark.asyncio
    async def test_false_after_eviction(self):
        jm = JobManager()
        await jm.register_live_url(MODEL, URL_A)
        await jm.evict_url(MODEL, URL_A)
        assert jm.is_url_live(URL_A) is False

    @pytest.mark.asyncio
    async def test_url_live_across_models(self):
        """is_url_live checks all model pools; a URL registered under any model is live."""
        jm = JobManager()
        await jm.register_live_url("model-X", URL_A)
        assert jm.is_url_live(URL_A) is True

    @pytest.mark.asyncio
    async def test_one_url_evicted_other_still_live(self):
        jm = JobManager()
        await jm.register_live_url(MODEL, URL_A)
        await jm.register_live_url(MODEL, URL_B)
        await jm.evict_url(MODEL, URL_A)
        assert jm.is_url_live(URL_A) is False
        assert jm.is_url_live(URL_B) is True


# ---------------------------------------------------------------------------
# Multi-model isolation
# ---------------------------------------------------------------------------

class TestMultiModelIsolation:

    @pytest.mark.asyncio
    async def test_models_have_independent_pools(self):
        """Registering a URL for one model must not affect another model's pool."""
        jm = JobManager()
        await jm.register_live_url("model-A", URL_A)

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(jm.get_live_url("model-B"), timeout=0.05)

    @pytest.mark.asyncio
    async def test_evicting_from_one_model_does_not_affect_other(self):
        jm = JobManager()
        await jm.register_live_url("model-A", URL_A)
        await jm.register_live_url("model-B", URL_A)  # same URL, different model
        await jm.evict_url("model-A", URL_A)

        # model-B still has URL_A live
        url = await asyncio.wait_for(jm.get_live_url("model-B"), timeout=0.1)
        assert url == URL_A


# ---------------------------------------------------------------------------
# get_live_urls snapshot
# ---------------------------------------------------------------------------

class TestGetLiveUrls:

    @pytest.mark.asyncio
    async def test_empty_when_no_urls_registered(self):
        jm = JobManager()
        assert jm.get_live_urls(MODEL) == []

    @pytest.mark.asyncio
    async def test_returns_all_registered_urls(self):
        jm = JobManager()
        await jm.register_live_url(MODEL, URL_A)
        await jm.register_live_url(MODEL, URL_B)
        urls = jm.get_live_urls(MODEL)
        assert set(urls) == {URL_A, URL_B}

    @pytest.mark.asyncio
    async def test_returns_copy_not_internal_list(self):
        """Mutating the returned list must not affect the internal state."""
        jm = JobManager()
        await jm.register_live_url(MODEL, URL_A)
        urls = jm.get_live_urls(MODEL)
        urls.clear()
        assert jm.get_live_urls(MODEL) == [URL_A]
