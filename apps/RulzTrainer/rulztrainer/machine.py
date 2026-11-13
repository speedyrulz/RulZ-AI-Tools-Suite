"""What the WebUI shows about the machine it runs on, and the one thing it may do to a rented one.

- disks(): free space of the drives the models and the runs are on. A rental's disk has a fixed size and a run that
  fills it dies in the middle of a save, so the Monitor shows it and warns early.
- memory(): system memory, the container's own limit where there is one (psutil alone reports the host's).
- vast_instance(): the vast.ai instance this is, from the variables vast.ai sets in every container, or None.
- bucket_download(): how far the model download of scripts/vast/download_models.sh is.
- AutoStop: "stop this instance when the runs are done". Armed from the Monitor, kept in output/_autostop.json so a
  restarted WebUI still knows, and disarmed before the stop is sent so a restarted instance does not stop again.

Nothing here imports torch: the WebUI calls it from its polling timers.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUTOSTOP_FILE = ROOT / "output" / "_autostop.json"
GB = 1024 ** 3
DISK_WARN_GB = 15.0  # a few checkpoints and their samples
DISK_CRITICAL_GB = 5.0  # the next save may not fit
STOP_GRACE_SECONDS = 180.0  # nothing running for this long before the instance stops: time to cancel
VAST_API = "https://console.vast.ai/api/v0"
_ENV_FILE = Path("/etc/environment")  # vast.ai's images write the container's variables here as well


def disk_level(free_gb: float) -> str:
    """"ok", "warning" or "critical" for this much free space."""
    return "critical" if free_gb < DISK_CRITICAL_GB else "warning" if free_gb < DISK_WARN_GB else "ok"


def disks(paths: dict[str, Path | str]) -> list[dict]:
    """Free space per drive for the named folders, e.g. {"runs": output, "models": models}: folders on one drive
    share a row ("runs, models"). A folder that does not exist yet counts as its nearest existing parent. Sizes in
    GB."""
    rows: dict[object, dict] = {}
    for label, path in paths.items():
        p = Path(path).resolve()
        while not p.exists() and p.parent != p:
            p = p.parent
        try:
            usage = shutil.disk_usage(p)
            key = os.stat(p).st_dev or (usage.total, usage.free)
        except OSError:
            continue
        row = rows.setdefault(key, {"labels": [], "path": str(p), "total": usage.total / GB,
                                    "used": (usage.total - usage.free) / GB, "free": usage.free / GB})
        row["labels"].append(label)
    out = []
    for row in rows.values():
        row["label"] = ", ".join(row.pop("labels"))
        row["level"] = disk_level(row["free"])
        out.append(row)
    return out


def _cgroup_number(*names: str) -> int | None:
    for name in names:
        try:
            text = Path(name).read_text(encoding="ascii").strip()
        except (OSError, UnicodeDecodeError):
            continue
        if text.isdigit():
            return int(text)
    return None


def memory() -> dict | None:
    """{"total", "used"} in GB, or None without psutil. Inside a container with a memory limit, the limit and the
    container's own use."""
    try:
        import psutil

        vm = psutil.virtual_memory()
    except Exception:  # noqa: BLE001
        return None
    total, used = vm.total, vm.total - vm.available
    limit = _cgroup_number("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes")
    if limit and limit < total:
        mine = _cgroup_number("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes")
        total, used = limit, min(limit, mine if mine is not None else used)
    return {"total": total / GB, "used": used / GB}


CPU_MAX_V2 = Path("/sys/fs/cgroup/cpu.max")
CPU_QUOTA_V1 = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
CPU_PERIOD_V1 = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS")  # a thread count of the user's own: left as it is


def _read(path: Path) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None


def cpu_quota(v2: Path = CPU_MAX_V2, v1_quota: Path = CPU_QUOTA_V1, v1_period: Path = CPU_PERIOD_V1) -> float | None:
    """How many CPUs this container may use (its cgroup quota over its period), or None when it has no limit or the
    files are not there. A vast.ai container can show every CPU of the host (os.cpu_count, nproc: 24) while its quota
    is half of them (11.5): a program that starts a thread per CPU it sees then spends most of its time throttled.
    cgroup v2 cpu.max is "<quota> <period>" ("max <period>": no limit); cgroup v1 has the two numbers in their own
    files (quota -1: no limit)."""
    def ratio(quota: str, period: str) -> float | None:
        try:
            q, p = float(quota), float(period)
        except (TypeError, ValueError):
            return None
        return q / p if q > 0 and p > 0 else None

    text = _read(v2)
    if text is not None:
        parts = text.split()
        if len(parts) == 2 and parts[0] == "max":
            return None
        return ratio(*parts) if len(parts) == 2 else None
    return ratio(_read(v1_quota), _read(v1_period))


def cpu_thread_cap(threads: int, quota: float | None, env=None) -> int | None:
    """The CPU thread count to use instead of `threads` (PyTorch's, one per CPU the machine shows) when the
    container's quota is smaller, else None: nothing to change, no quota, or the user set OMP_NUM_THREADS /
    MKL_NUM_THREADS themselves."""
    env = os.environ if env is None else env
    if quota is None or any(str(env.get(name) or "").strip() for name in THREAD_ENV):
        return None
    cap = max(1, int(quota))
    return cap if threads > cap else None


def _container_env(env=None, env_file: Path = _ENV_FILE) -> dict[str, str]:
    """The process's variables, with vast.ai's from /etc/environment for the ones a service did not inherit."""
    values = dict(os.environ if env is None else env)
    try:
        lines = env_file.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return values
    for line in lines:
        name, sep, value = line.strip().removeprefix("export ").partition("=")
        if sep and name in ("CONTAINER_ID", "CONTAINER_API_KEY", "VAST_CONTAINERLABEL", "PUBLIC_IPADDR") \
                and not values.get(name):
            values[name] = value.strip().strip("'\"")
    return values


def vast_instance(env=None, env_file: Path = _ENV_FILE) -> dict | None:
    """{"id", "label", "can_stop"} when this is a vast.ai instance, else None. can_stop: the instance's own API key
    is there (it may start, stop and destroy this instance and nothing else)."""
    values = _container_env(env, env_file)
    label = values.get("VAST_CONTAINERLABEL") or ""
    raw = values.get("CONTAINER_ID") or label.removeprefix("C.")
    if not str(raw).isdigit():
        return None
    return {"id": int(raw), "label": label or f"C.{raw}", "can_stop": bool(values.get("CONTAINER_API_KEY")),
            "ip": values.get("PUBLIC_IPADDR") or ""}


def stop_instance(env=None, env_file: Path = _ENV_FILE, opener=urllib.request.urlopen) -> tuple[bool, str]:
    """Stop this vast.ai instance with its own API key: the GPU's billing ends, the disk (and its storage charge)
    stays, and Start on vast.ai brings it back. (ok, message)."""
    values = _container_env(env, env_file)
    info = vast_instance(values, env_file)
    key = values.get("CONTAINER_API_KEY")
    if info is None or not key:
        return False, "This is not a vast.ai instance with its own API key (CONTAINER_ID, CONTAINER_API_KEY)."
    request = urllib.request.Request(
        f"{VAST_API}/instances/{info['id']}/", data=json.dumps({"state": "stopped"}).encode("utf-8"), method="PUT",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json", "Accept": "application/json"})
    try:
        with opener(request, timeout=30) as response:
            body = response.read().decode("utf-8", errors="replace")
        try:
            answer = json.loads(body)
        except json.JSONDecodeError:
            answer = {}
        if answer.get("success") is False:
            raise OSError(str(answer.get("msg") or answer.get("error") or body)[:300])
        return True, f"Stop sent to vast.ai for instance {info['id']}."
    except (urllib.error.URLError, OSError, ValueError) as e:
        api_error = f"{type(e).__name__}: {e}"
    cli = shutil.which("vastai")  # the same request by the CLI, where it is installed
    if cli:
        try:
            done = subprocess.run([cli, "stop", "instance", str(info["id"]), "--api-key", key], capture_output=True,
                                  text=True, timeout=60)
            if done.returncode == 0:
                return True, f"Stop sent to vast.ai for instance {info['id']} (vastai CLI)."
            api_error += f"; vastai CLI: {(done.stderr or done.stdout).strip()[:200]}"
        except (OSError, subprocess.SubprocessError) as e:
            api_error += f"; vastai CLI: {e}"
    return False, f"Could not stop instance {info['id']}: {api_error}"


_bucket_cache: dict = {"at": 0.0, "key": None, "value": None}


def bucket_download(models_dir: Path | str, max_age: float = 10.0) -> dict | None:
    """The first-boot model download (scripts/vast/download_models.sh): None when there was none, else {"state":
    "running" | "done" | "failed", "gb": what is on disk so far, "line": the log's last line}. The folder is measured
    at most every max_age seconds."""
    models = Path(models_dir)
    log = models / "bucket_download.log"
    if not log.is_file():
        return None
    now = time.time()
    if _bucket_cache["key"] == str(models) and now - _bucket_cache["at"] < max_age:
        return _bucket_cache["value"]
    try:
        lines = [x for x in log.read_text(encoding="utf-8", errors="replace").splitlines() if x.strip()]
    except OSError:
        lines = []
    tail = lines[-1].strip() if lines else ""
    if (models / "bucket" / ".download_complete").is_file():
        state = "done"
    elif any(x.startswith("[ERROR]") for x in lines[-3:]):
        state = "failed"
    else:
        state = "running"
    size = 0
    if state == "running":
        for folder, _, names in os.walk(models / "bucket"):
            for name in names:
                try:
                    size += os.path.getsize(os.path.join(folder, name))
                except OSError:
                    pass
    value = {"state": state, "gb": size / GB, "line": tail[:200]}
    _bucket_cache.update(at=now, key=str(models), value=value)
    return value


class AutoStop:
    """Stop the rented instance once the runs are done.

    Armed, it waits until a run has been seen running, then for nothing to run and nothing to wait in the queue for
    STOP_GRACE_SECONDS on end (a failed or stopped last run counts as done: the instance would only sit there billing).
    Runs left waiting in a paused queue keep it from stopping: a pause is someone's decision to look first. tick()
    is called by the WebUI's queue thread, so it works with no browser open."""

    def __init__(self, path: Path = AUTOSTOP_FILE, grace: float = STOP_GRACE_SECONDS, stop=stop_instance):
        self.path, self.grace, self.stop = Path(path), grace, stop
        self.state = {"armed": False, "seen_running": False, "idle_since": None, "last": ""}
        try:
            saved = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(saved, dict):
                self.state.update({k: saved[k] for k in self.state if k in saved})
        except (OSError, json.JSONDecodeError):
            pass
        self.state["idle_since"] = None  # the wait starts again in a restarted WebUI

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.state), encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError:
            pass

    @property
    def armed(self) -> bool:
        return bool(self.state["armed"])

    def arm(self, running: bool = False) -> None:
        self.state.update(armed=True, seen_running=bool(running), idle_since=None, last="")
        self._save()

    def disarm(self, note: str = "") -> None:
        self.state.update(armed=False, seen_running=False, idle_since=None, last=note)
        self._save()

    def seconds_left(self, now: float | None = None) -> float | None:
        """Seconds until the stop while the wait is on, else None."""
        since = self.state["idle_since"]
        if not self.armed or since is None:
            return None
        return max(0.0, self.grace - ((time.time() if now is None else now) - since))

    def tick(self, running: bool, waiting: int, now: float | None = None) -> tuple[bool, str] | None:
        """One look at the queue. Returns stop()'s (ok, message) at the moment it stops the instance, else None."""
        if not self.armed:
            return None
        now = time.time() if now is None else now
        if running:
            if not self.state["seen_running"] or self.state["idle_since"] is not None:
                self.state.update(seen_running=True, idle_since=None)
                self._save()
            return None
        if waiting or not self.state["seen_running"]:
            if self.state["idle_since"] is not None:
                self.state["idle_since"] = None
            return None
        if self.state["idle_since"] is None:
            self.state["idle_since"] = now
            return None
        if now - self.state["idle_since"] < self.grace:
            return None
        self.disarm("stopping the instance")  # first: the instance must not stop again right after its next start
        ok, message = self.stop()
        self.state["last"] = message
        self._save()
        return ok, message
