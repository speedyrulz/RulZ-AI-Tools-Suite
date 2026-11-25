"""Running ComfyUI instances.

Each instance is one ``wsl.exe`` process on Windows with ComfyUI attached to it
inside WSL. The two live and die together: when ComfyUI exits, ``wsl.exe``
exits, and when ``wsl.exe`` is ended, WSL hangs up the session and ComfyUI ends.

Stopping is done the gentle way first: Ctrl+C (SIGINT) to ComfyUI's process
group, the same as pressing Ctrl+C in a terminal. Only if that does not work in
time are SIGTERM, SIGKILL and finally ending ``wsl.exe`` used.
"""
from __future__ import annotations

from enum import Enum

from PySide6.QtCore import QObject, QProcess, QTimer, Signal
from PySide6.QtNetwork import QAbstractSocket, QTcpSocket

from . import wsl
from .command import (LOG_PREFIX, PID_PATTERN, READY_PATTERN, build_launch_script,
                      build_signal_script, instance_url, launch_signature)
from .model import Profile
from .options import Catalog
from .textstream import LineSplitter, TextStream


class State(str, Enum):
    STOPPED = "Stopped"
    QUEUED = "Queued"
    STARTING = "Starting"
    RUNNING = "Running"
    STOPPING = "Stopping"
    FAILED = "Failed"


ACTIVE_STATES = (State.STARTING, State.RUNNING, State.STOPPING)

_TERM_GRACE_MS = 5000
_KILL_GRACE_MS = 2000


class Instance(QObject):
    state_changed = Signal(str)   # profile id
    output = Signal(str, str)     # profile id, text
    ready = Signal(str)           # profile id

    def __init__(self, profile_id: str, job=None, parent: QObject | None = None):
        super().__init__(parent)
        self.profile_id = profile_id
        self.state = State.STOPPED
        self.profile: Profile | None = None   # the settings this instance was started with
        self.catalog: Catalog | None = None   # the option list it was started with
        self.signature = ""
        self.url = ""
        self.linux_pid = 0
        self._linux_start = ""   # kernel start time of that process; identifies it for good
        self.exit_code: int | None = None
        self._job = job
        self._proc: QProcess | None = None
        self._script = ""
        self._stream = TextStream()
        self._lines = LineSplitter()
        self._stop_requested = False
        self._stop_stage = 0
        self._stop_timeout_ms = 10000
        self._stop_timer = QTimer(self)
        self._stop_timer.setSingleShot(True)
        self._stop_timer.timeout.connect(self._advance_stop)

    # -- public -------------------------------------------------------------
    @property
    def active(self) -> bool:
        return self.state in ACTIVE_STATES

    def start(self, profile: Profile, catalog: Catalog, stop_timeout: int = 10) -> bool:
        if self.active:
            return False
        self.profile = profile.snapshot()
        self.catalog = catalog
        self.signature = launch_signature(self.profile, catalog)
        self.url = instance_url(self.profile)
        self._script = build_launch_script(self.profile, catalog)
        self._stop_timeout_ms = max(1, int(stop_timeout)) * 1000
        self._stop_requested = False
        self._stop_stage = 0
        self.linux_pid = 0
        self._linux_start = ""
        self.exit_code = None
        self._stream = TextStream()
        self._lines = LineSplitter()

        where = self.profile.distro or "the default WSL distro"
        self._say(f"Starting '{self.profile.name}' in {where} ...")

        proc = QProcess(self)
        proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        proc.setProcessEnvironment(wsl.process_environment())
        proc.started.connect(self._on_started)
        proc.readyReadStandardOutput.connect(self._on_output)
        proc.finished.connect(self._on_finished)
        proc.errorOccurred.connect(self._on_error)
        self._proc = proc
        self._set_state(State.STARTING)
        proc.start(wsl.WSL_EXE, wsl.bash_stdin_args(self.profile.distro, self.profile.user))
        return True

    def stop(self) -> None:
        """Ask ComfyUI to stop. Escalates by itself if it does not."""
        if self.state not in (State.STARTING, State.RUNNING) or self._proc is None:
            return
        self._stop_requested = True
        self._stop_stage = 0
        self._set_state(State.STOPPING)
        if self.linux_pid:
            self._say("Stopping (Ctrl+C) ...")
            self._signal("INT")
            self._stop_timer.start(self._stop_timeout_ms)
        else:
            # ComfyUI has not started yet; ending wsl.exe ends the script.
            self._stop_stage = 3
            self._end_windows_process()
            self._stop_timer.start(_KILL_GRACE_MS)

    def force_stop(self) -> None:
        """Skip the waiting: end the process tree now."""
        if self._proc is None:
            return
        self._stop_requested = True
        if self.state in (State.STARTING, State.RUNNING):
            self._set_state(State.STOPPING)
        if self.linux_pid:
            self._signal("KILL")
        self._stop_stage = 3
        self._end_windows_process()
        self._stop_timer.start(_KILL_GRACE_MS)

    def mark_ready(self) -> None:
        """Called when the server answers on its port (or printed its ready line)."""
        if self.state == State.STARTING and self.linux_pid:
            self._set_state(State.RUNNING)
            self.ready.emit(self.profile_id)

    # -- process events -----------------------------------------------------
    def _on_started(self):
        proc = self._proc
        if proc is None:
            return
        if self._stop_requested:
            # Stop was pressed before the process was up: do not start ComfyUI at all.
            proc.closeWriteChannel()
            proc.kill()
            return
        if self._job is None or not self._job.assign(proc.processId()):
            self._say("Note: this instance could not be tied to the launcher. If the launcher "
                      "crashes, stop the instance by hand.")
        proc.write(self._script.encode("utf-8"))
        proc.closeWriteChannel()

    def _on_output(self):
        if self._proc is None:
            return
        text = self._stream.decode(self._proc.readAllStandardOutput().data()).replace("\x00", "")
        if not text:
            return
        for line in self._lines.feed(text):
            if not self.linux_pid:
                match = PID_PATTERN.match(line)
                if match:
                    self.linux_pid = int(match.group(1))
                    self._linux_start = match.group(2) or ""
                    continue
            if self.state == State.STARTING and READY_PATTERN.search(line):
                self.mark_ready()
        self.output.emit(self.profile_id, text)

    def _on_error(self, error):
        if error == QProcess.ProcessError.FailedToStart and self._proc is not None:
            self._say("ERROR: wsl.exe could not be started. Is WSL installed?")
            self._cleanup()
            self.exit_code = -1
            self._set_state(State.FAILED)

    def _on_finished(self, code, status):
        if self._proc is None:
            return
        self._on_output()
        crashed = status != QProcess.ExitStatus.NormalExit
        self.exit_code = -1 if crashed else code
        requested = self._stop_requested
        self._cleanup()
        if requested:
            self._say("Stopped.")
            self._set_state(State.STOPPED)
        elif code == 0 and not crashed:
            self._say("ComfyUI exited.")
            self._set_state(State.STOPPED)
        else:
            self._say(f"ComfyUI ended unexpectedly (exit code {self.exit_code}).")
            self._set_state(State.FAILED)

    # -- internals ----------------------------------------------------------
    def _cleanup(self):
        self._stop_timer.stop()
        proc, self._proc = self._proc, None
        self.linux_pid = 0
        self._linux_start = ""
        if proc is not None:
            proc.deleteLater()

    def _set_state(self, state: State):
        if state != self.state:
            self.state = state
            self.state_changed.emit(self.profile_id)

    def _say(self, message: str):
        self.output.emit(self.profile_id, f"\n{LOG_PREFIX}{message}\n")

    def _signal(self, name: str):
        """Send a signal to ComfyUI's process group inside WSL.

        The signal is sent by a second, short wsl.exe call, so it arrives a
        moment later. To be sure it can never reach a different program, the
        script checks the process's start time first: if the instance has ended
        meanwhile and its number was given to another process, nothing is sent.
        Without a known start time no signal is sent at all, and stopping falls
        back to ending wsl.exe.
        """
        if self._proc is None or not self.linux_pid or not self._linux_start or self.profile is None:
            return
        script = build_signal_script(self.linux_pid, self._linux_start, name)
        wsl.run_bash(script, self.profile.distro, self.profile.user, timeout_ms=8000, parent=self)

    def _advance_stop(self):
        if self._proc is None:
            return
        self._stop_stage += 1
        if self._stop_stage == 1:
            self._say("Still running; sending SIGTERM ...")
            self._signal("TERM")
            self._stop_timer.start(_TERM_GRACE_MS)
        elif self._stop_stage == 2:
            self._say("Still running; sending SIGKILL ...")
            self._signal("KILL")
            self._stop_timer.start(_KILL_GRACE_MS)
        else:
            self._end_windows_process()

    def _end_windows_process(self):
        """End our wsl.exe. WSL then hangs up the Linux session attached to it.

        QProcess ends the process through the handle it holds, so this can never
        hit another program (a "taskkill" by process number could, if Windows
        gave the number to a new process in between).
        """
        proc = self._proc
        if proc is not None:
            proc.kill()


class InstanceManager(QObject):
    state_changed = Signal(str)
    output = Signal(str, str)
    ready = Signal(str)

    def __init__(self, job=None, parent: QObject | None = None):
        super().__init__(parent)
        self._job = job
        self._instances: dict[str, Instance] = {}
        self._queue: list[tuple[Profile, Catalog, int]] = []
        self._restart: dict[str, tuple[Profile, Catalog, int]] = {}
        self._stagger_ms = 0
        self._queue_timer = QTimer(self)
        self._queue_timer.setSingleShot(True)
        self._queue_timer.timeout.connect(self._start_next)

    # -- queries ------------------------------------------------------------
    def instance(self, profile_id: str) -> Instance | None:
        return self._instances.get(profile_id)

    def state(self, profile_id: str) -> State:
        if any(p.id == profile_id for p, _c, _t in self._queue) or profile_id in self._restart:
            inst = self._instances.get(profile_id)
            if inst is None or not inst.active:
                return State.QUEUED
        inst = self._instances.get(profile_id)
        return inst.state if inst else State.STOPPED

    def is_busy(self, profile_id: str) -> bool:
        return self.state(profile_id) not in (State.STOPPED, State.FAILED)

    def active_profiles(self) -> list[Profile]:
        """The settings of everything that is running, starting, stopping or queued."""
        profiles = [i.profile for i in self._instances.values() if i.active and i.profile]
        profiles += [p for p, _c, _t in self._queue]
        # A restart that waits for the old process to end counts with its new settings.
        waiting = {p.id: p for p, _c, _t in self._restart.values()}
        profiles = [waiting.pop(p.id, p) for p in profiles]
        return profiles + list(waiting.values())

    def active_ids(self) -> list[str]:
        return [pid for pid, inst in self._instances.items() if inst.active]

    def any_active(self) -> bool:
        return bool(self._queue) or bool(self._restart) or any(i.active for i in self._instances.values())

    # -- actions ------------------------------------------------------------
    def start(self, profile: Profile, catalog: Catalog, stop_timeout: int = 10) -> bool:
        inst = self._instances.get(profile.id)
        if inst is None:
            inst = Instance(profile.id, self._job, self)
            inst.state_changed.connect(self._on_state_changed)
            inst.output.connect(self.output)
            inst.ready.connect(self.ready)
            self._instances[profile.id] = inst
        return inst.start(profile, catalog, stop_timeout)

    def start_many(self, items: list[tuple[Profile, Catalog]], stagger_seconds: float,
                   stop_timeout: int = 10) -> None:
        """Start several profiles, with a pause between them."""
        for profile, catalog in items:
            if self.is_busy(profile.id):
                continue
            self._queue.append((profile.snapshot(), catalog, stop_timeout))
            self.state_changed.emit(profile.id)
        self._stagger_ms = int(max(0.0, stagger_seconds) * 1000)
        if not self._queue_timer.isActive():
            self._start_next()

    def stop(self, profile_id: str) -> None:
        self._cancel_restart(profile_id)
        if self._unqueue(profile_id):
            return
        inst = self._instances.get(profile_id)
        if inst:
            inst.stop()

    def force_stop(self, profile_id: str) -> None:
        self._cancel_restart(profile_id)
        self._unqueue(profile_id)
        inst = self._instances.get(profile_id)
        if inst:
            inst.force_stop()

    def stop_all(self) -> None:
        self._clear_waiting()
        for inst in list(self._instances.values()):
            inst.stop()

    def force_stop_all(self) -> None:
        self._clear_waiting()
        for inst in list(self._instances.values()):
            inst.force_stop()

    def restart(self, profile: Profile, catalog: Catalog, stop_timeout: int = 10) -> None:
        inst = self._instances.get(profile.id)
        if inst is None or not inst.active:
            self.start(profile, catalog, stop_timeout)
            return
        self._restart[profile.id] = (profile.snapshot(), catalog, stop_timeout)
        inst.stop()

    # -- internals ----------------------------------------------------------
    def _cancel_restart(self, profile_id: str) -> None:
        if self._restart.pop(profile_id, None) is not None:
            self.state_changed.emit(profile_id)

    def _clear_waiting(self) -> None:
        """Forget everything that has not started yet: the queue and pending restarts."""
        waiting = [p.id for p, _c, _t in self._queue] + list(self._restart)
        self._queue.clear()
        self._restart.clear()
        self._queue_timer.stop()
        for profile_id in waiting:
            self.state_changed.emit(profile_id)

    def _run_restart(self, profile_id: str) -> None:
        """Second half of a restart. Does nothing if the restart was cancelled meanwhile."""
        inst = self._instances.get(profile_id)
        if profile_id not in self._restart or (inst is not None and inst.active):
            return
        profile, catalog, stop_timeout = self._restart.pop(profile_id)
        if not self.start(profile, catalog, stop_timeout):
            self.state_changed.emit(profile_id)

    def _unqueue(self, profile_id: str) -> bool:
        before = len(self._queue)
        self._queue = [item for item in self._queue if item[0].id != profile_id]
        if len(self._queue) != before:
            self.state_changed.emit(profile_id)
            return True
        return False

    def _start_next(self):
        if not self._queue:
            return
        profile, catalog, stop_timeout = self._queue.pop(0)
        if not self.start(profile, catalog, stop_timeout):
            self.state_changed.emit(profile.id)
        if self._queue:
            self._queue_timer.start(self._stagger_ms)

    def _on_state_changed(self, profile_id: str):
        inst = self._instances.get(profile_id)
        if inst is not None and not inst.active and profile_id in self._restart:
            # The old process has ended. Start again from the event loop, a moment
            # later. The request stays registered until then, so Stop can cancel it.
            QTimer.singleShot(300, self, lambda: self._run_restart(profile_id))
        self.state_changed.emit(profile_id)


class PortWatcher(QObject):
    """Checks now and then which of the watched local ports accept connections.

    Used to notice a server that was started outside the launcher, and as a
    second way to see that our own server has finished starting.
    """
    changed = Signal(int, bool)   # port, is open

    def __init__(self, interval_ms: int = 4000, parent: QObject | None = None):
        super().__init__(parent)
        self._ports: set[int] = set()
        self._open: dict[int, bool] = {}
        self._pending: dict[int, QTcpSocket] = {}
        self._timer = QTimer(self)
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self.check_now)

    def start(self):
        self._timer.start()
        self.check_now()

    def stop(self):
        self._timer.stop()

    def set_ports(self, ports) -> None:
        self._ports = {int(p) for p in ports if 1 <= int(p) <= 65535}
        for port in list(self._open):
            if port not in self._ports:
                del self._open[port]

    def is_open(self, port: int) -> bool:
        return self._open.get(port, False)

    def forget(self, port: int) -> None:
        """Drop what is known about a port (its server just ended) and look again."""
        self._open.pop(port, None)
        stale = self._pending.pop(port, None)
        if stale is not None:
            stale.abort()
            stale.deleteLater()
        QTimer.singleShot(400, self, self.check_now)

    def check_now(self):
        for port in self._ports:
            if port in self._pending:
                continue
            socket = QTcpSocket(self)
            self._pending[port] = socket
            socket.connected.connect(lambda port=port: self._result(port, True))
            socket.errorOccurred.connect(lambda _e, port=port: self._result(port, False))
            QTimer.singleShot(1500, socket, lambda port=port: self._result(port, False))
            socket.connectToHost("127.0.0.1", port)

    def _result(self, port: int, is_open: bool):
        socket = self._pending.pop(port, None)
        if socket is None:
            return
        if socket.state() != QAbstractSocket.SocketState.UnconnectedState:
            socket.abort()
        socket.deleteLater()
        if port not in self._ports:
            return
        if self._open.get(port) != is_open:
            self._open[port] = is_open
            self.changed.emit(port, is_open)
