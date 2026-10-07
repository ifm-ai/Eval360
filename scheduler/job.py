import asyncio
import logging
import random

logger = logging.getLogger("JobManager")


class JobManager:
    """In-memory pool of live VLLM endpoint URLs, one pool per model.

    The scheduler's handle_job_update loop drives state changes:
      - register_live_url() when a job passes its health check
      - evict_url() when a job disappears or times out

    OpenAIConnection calls get_live_url() to obtain a URL for each request
    and is_url_live() to distinguish node-death failures from real errors.
    """

    def __init__(self):
        # model_name -> list of live URL strings
        self._live_urls: dict[str, list[str]] = {}
        # model_name -> asyncio.Condition
        self._conditions: dict[str, asyncio.Condition] = {}

    def _ensure_model(self, model_name: str):
        if model_name not in self._conditions:
            self._conditions[model_name] = asyncio.Condition()
            self._live_urls[model_name] = []

    async def get_live_url(self, model_name: str) -> str:
        """Block until at least one live URL is available, then return one at random."""
        self._ensure_model(model_name)
        async with self._conditions[model_name]:
            await self._conditions[model_name].wait_for(
                lambda: len(self._live_urls[model_name]) > 0
            )
            return random.choice(self._live_urls[model_name])

    async def register_live_url(self, model_name: str, url: str):
        """Add url to the live pool and wake any get_live_url() waiters."""
        self._ensure_model(model_name)
        async with self._conditions[model_name]:
            if url not in self._live_urls[model_name]:
                self._live_urls[model_name].append(url)
                logger.info(f"Registered live URL for {model_name}: {url}")
            self._conditions[model_name].notify_all()

    async def evict_url(self, model_name: str, url: str):
        """Remove url from the live pool."""
        if model_name not in self._live_urls:
            return
        async with self._conditions[model_name]:
            try:
                self._live_urls[model_name].remove(url)
                logger.info(f"Evicted URL for {model_name}: {url}")
            except ValueError:
                pass

    def is_url_live(self, url: str) -> bool:
        """Return True if url is currently in any model's live pool."""
        for urls in self._live_urls.values():
            if url in urls:
                return True
        return False

    def get_live_urls(self, model_name: str) -> list[str]:
        """Return a snapshot of the current live URL list for a model."""
        return list(self._live_urls.get(model_name, []))
