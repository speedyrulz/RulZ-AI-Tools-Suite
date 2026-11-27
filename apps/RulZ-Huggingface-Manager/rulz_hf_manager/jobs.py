"""Background jobs (uploads, downloads, moves...) with progress shown in the Activity drawer."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from nicegui import background_tasks, run

from .hub import friendly_error

log = logging.getLogger('rulz_hf_manager')


@dataclass
class Job:
    id: int
    title: str
    icon: str = 'sync'
    status: str = 'queued'  # queued | running | done | error
    progress: float | None = None
    message: str = ''
    detail: str = ''
    error: str = ''
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    result: Any = None

    def set(self, progress: float | None = None, message: str | None = None, detail: str | None = None) -> None:
        if progress is not None:
            self.progress = progress
        if message is not None:
            self.message = message
        if detail is not None:
            self.detail = detail
        manager.bump()

    @property
    def elapsed(self) -> float:
        if not self.started:
            return 0.0
        return (self.finished or time.time()) - self.started

    @property
    def active(self) -> bool:
        return self.status in {'queued', 'running'}


async def _own_thread(fn: Callable[[Job], Any], job: Job) -> Any:
    """Run ``fn(job)`` in a thread of its own. An unlimited job (a follow that lasts as long as a transfer, hours) must
    not sit in NiceGUI's shared io_bound pool: a few of them would leave no worker for the rest of the app."""
    loop = asyncio.get_running_loop()
    future = loop.create_future()

    def settle(result: Any, error: BaseException | None) -> None:
        if future.done():
            return
        if error is not None:
            future.set_exception(error)
        else:
            future.set_result(result)

    def target() -> None:
        try:
            result = fn(job)
        except BaseException as e:  # noqa: BLE001  (handed to the awaiting coroutine)
            loop.call_soon_threadsafe(settle, None, e)
        else:
            loop.call_soon_threadsafe(settle, result, None)

    threading.Thread(target=target, name=f'job-{job.id}', daemon=True).start()
    return await future


class JobManager:
    def __init__(self, max_concurrent: int = 3) -> None:
        self.jobs: list[Job] = []
        self.version = 0
        self._next_id = 1
        self._sem: asyncio.Semaphore | None = None
        self._max = max_concurrent

    def bump(self) -> None:
        self.version += 1

    @property
    def active_count(self) -> int:
        return sum(1 for j in self.jobs if j.active)

    def clear_finished(self) -> None:
        self.jobs = [j for j in self.jobs if j.active]
        self.bump()

    def submit(self, title: str, fn: Callable[[Job], Any], *, icon: str = 'sync',
               on_success: Callable[[Job], Any] | None = None,
               on_error: Callable[[Job], Any] | None = None, limited: bool = True) -> Job:
        """Run ``fn(job)`` in a worker thread. Callbacks run on the event loop afterwards."""
        job = Job(id=self._next_id, title=title, icon=icon)
        self._next_id += 1
        self.jobs.insert(0, job)
        del self.jobs[200:]
        self.bump()
        background_tasks.create(self._run(job, fn, on_success, on_error, limited), name=f'job-{job.id}')
        return job

    async def _run(self, job: Job, fn, on_success, on_error, limited: bool = True) -> None:
        if self._sem is None:
            self._sem = asyncio.Semaphore(self._max)
        # limited=False: long-lived jobs (a follow that watches the instance) must not hold one of the slots
        async with self._sem if limited else contextlib.nullcontext():
            job.status = 'running'
            job.started = time.time()
            self.bump()
            try:
                job.result = await (run.io_bound(fn, job) if limited else _own_thread(fn, job))
                job.status = 'done'
                job.progress = 1.0
                callback = on_success
            except Exception as e:  # noqa: BLE001
                log.warning('Job "%s" failed:\n%s', job.title, traceback.format_exc())
                job.status = 'error'
                job.error = friendly_error(e)
                callback = on_error
            finally:
                job.finished = time.time()
                self.bump()
        if callback:
            try:
                result = callback(job)
                if inspect.isawaitable(result):
                    await result
            except Exception:  # the page may be gone; never let callbacks crash the job runner
                log.debug('Job callback failed', exc_info=True)


manager = JobManager()
