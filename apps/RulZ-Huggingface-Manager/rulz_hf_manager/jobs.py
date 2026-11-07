"""Background jobs (uploads, downloads, moves...) with progress shown in the Activity drawer."""

from __future__ import annotations

import asyncio
import inspect
import logging
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
               on_error: Callable[[Job], Any] | None = None) -> Job:
        """Run ``fn(job)`` in a worker thread. Callbacks run on the event loop afterwards."""
        job = Job(id=self._next_id, title=title, icon=icon)
        self._next_id += 1
        self.jobs.insert(0, job)
        del self.jobs[200:]
        self.bump()
        background_tasks.create(self._run(job, fn, on_success, on_error), name=f'job-{job.id}')
        return job

    async def _run(self, job: Job, fn, on_success, on_error) -> None:
        if self._sem is None:
            self._sem = asyncio.Semaphore(self._max)
        async with self._sem:
            job.status = 'running'
            job.started = time.time()
            self.bump()
            try:
                job.result = await run.io_bound(fn, job)
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
