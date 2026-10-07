import asyncio
import errno
import functools
import glob
import re

from enum import Enum
from pathlib import PurePosixPath, Path

from watchdog.events import FileSystemEventHandler
#  PollingObserver is needed for network filesystems
from watchdog.observers.polling import PollingObserver as Observer

import logging
logger = logging.getLogger("FSManager")
logging.basicConfig(level=logging.INFO)


class DirType(Enum):
    MODELSPEC = 1
    MODELINSTANCE = 2
    DATA = 3


_GLOB_CHARS = re.compile(r"[*?\[]")


def _glob_translate(path: str, recursive: bool = True, include_hidden: bool = False) -> str:
    translate = getattr(glob, "translate", None)
    if translate is not None:
        return translate(path, recursive=recursive, include_hidden=include_hidden)

    def translate_segment(segment: str) -> str:
        result = []
        i = 0
        while i < len(segment):
            char = segment[i]
            if char == "*":
                result.append("[^/]*")
            elif char == "?":
                result.append("[^/]")
            elif char == "[":
                end = segment.find("]", i + 1)
                if end == -1:
                    result.append(re.escape(char))
                else:
                    content = segment[i + 1:end]
                    if content.startswith("!"):
                        content = "^" + re.escape(content[1:])
                    else:
                        content = re.escape(content)
                    result.append(f"[{content}]")
                    i = end
            else:
                result.append(re.escape(char))
            i += 1
        return "".join(result)

    parts = path.split("/")
    regex = ["^"]
    if parts and parts[0] == "":
        regex.append("/")
        parts = parts[1:]

    for index, part in enumerate(parts):
        if recursive and part == "**":
            if index == len(parts) - 1:
                regex.append(".*")
            else:
                regex.append("(?:[^/]+/)*")
            continue
        regex.append(translate_segment(part))
        if index != len(parts) - 1:
            regex.append("/")

    regex.append("$")
    return "".join(regex)


def _glob_constant_prefix(glob_path: str) -> str:
    if not glob_path.startswith("/"):
        raise ValueError(f"glob must be absolute: {glob_path}")

    parts = PurePosixPath(glob_path).parts

    prefix_parts = []
    for part in parts:
        if _GLOB_CHARS.search(part):
            break
        prefix_parts.append(part)

    # If the glob starts with '/', parts[0] is '/'
    return str(PurePosixPath(*prefix_parts))


async def _mkdir_shared_fs_safe(path, retries=10, delay=1.0):
    last_exc = OSError(errno.ESTALE, "Retries exhausted without a successful mkdir")
    for _ in range(retries):
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
            return
        except OSError as e:
            if e.errno != errno.ESTALE:
                raise
            last_exc = e
            await asyncio.sleep(delay)
    raise last_exc


class FSManager:
    def __init__(self, model_directory, dataset_directory, loop=None):
        self._loop = loop or asyncio.get_event_loop()
        self._queue = asyncio.Queue()
        self._observer = Observer(timeout=5.0)
        # TODO: handle final / in directory paths
        if model_directory:
            self.register_path(model_directory, DirType.MODELSPEC, scan_existing=True)
        if dataset_directory:
            self.register_path(dataset_directory, DirType.DATA, scan_existing=True)
        if model_directory or dataset_directory:
            self._observer.start()

    def register_path(self, path, dirtype: DirType, info=None, recursive=True, scan_existing=False):
        prefix = _glob_constant_prefix(path)
        if prefix != path:
            glob_re = re.compile(_glob_translate(path, recursive=True, include_hidden=True))
            watch_path = prefix
        else:
            glob_re = None
            watch_path = path

        class Handler(FileSystemEventHandler):
            def on_any_event(inner_self, event):
                # Hand off to asyncio thread-safely
                if glob_re:
                    match = glob_re.match(event.src_path)
                    if not match:
                        return
                self._loop.call_soon_threadsafe(
                    self._queue.put_nowait, (dirtype, event, info)
                )

        handler = Handler()
        #  TODO: handle when watch_path doesnt exist yet
        logger.info(f"registering watcher for {path} on {watch_path} with type {dirtype}")
        self._loop.call_soon_threadsafe(
            functools.partial(self._observer.schedule, handler, watch_path, recursive=recursive))

        if scan_existing:
            from watchdog.events import FileCreatedEvent
            watch_dir = Path(watch_path)
            if watch_dir.is_dir():
                existing = watch_dir.rglob("*") if recursive else watch_dir.iterdir()
                for f in existing:
                    if not f.is_file():
                        continue
                    src = str(f)
                    if glob_re and not glob_re.match(src):
                        continue
                    logger.info(f"enqueuing existing file {src} as {dirtype}")
                    self._queue.put_nowait((dirtype, FileCreatedEvent(src), info))

    # TODO: factor out code with above
    async def ensures_exists_register_path(self, path, dirtype: DirType, register_existing, info=None, recursive=True):
        prefix = _glob_constant_prefix(path)
        if prefix != path:
            glob_re = re.compile(_glob_translate(path, recursive=True))
            watch_path = prefix
        else:
            glob_re = None
            watch_path = path
        await _mkdir_shared_fs_safe(watch_path)

        class Handler(FileSystemEventHandler):
            def on_any_event(inner_self, event):
                # Hand off to asyncio thread-safely
                if glob_re:
                    match = glob_re.match(event.src_path)
                    # TODO: make sure we dont catch foo/not_done.txt
                    if not match or not event.src_path.endswith("done.txt"):
                        return
                self._loop.call_soon_threadsafe(
                    self._queue.put_nowait, (dirtype, event, info)
                )

        handler = Handler()
        if register_existing:
            path = Path(watch_path)
            if glob_re:
                # TODO: factor out str
                done_files = [str(f) for f in path.rglob("*") if glob_re.match(str(f)) and str(f).endswith("done.txt")]
            else:
                done_files = [str(f) for f in path.rglob("*") if str(f).endswith("done.txt")]
            logger.info(f"done files: {done_files}")
            for done_file in done_files:
                # TODO: make this a proper event notification object instead of done_file str
                await self._queue.put((dirtype, done_file, info))

        #  TODO: handle when watch_path doesnt exist yet
        logger.info(f"registering watcher for {path} on {watch_path} with type {dirtype}")
        self._loop.call_soon_threadsafe(
            functools.partial(self._observer.schedule, handler, watch_path, recursive=recursive))

    async def stop(self):
        self._observer.stop()
        self._observer.join()

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return await self._queue.get()
        except asyncio.CancelledError:
            raise StopAsyncIteration
