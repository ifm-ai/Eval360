import asyncio
import logging
logger = logging.getLogger("GradingManager")
logging.basicConfig(level=logging.INFO)


class GradingManager:
    def __init__(self):
        self._queue = asyncio.Queue()

    async def enqueue(self, event, iterator):
        await self._queue.put((event, iterator))
        logger.info(f"Enqueuing {event.uuid} to grade {event.model} on {event.task_uuid}")

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self._queue.get()
