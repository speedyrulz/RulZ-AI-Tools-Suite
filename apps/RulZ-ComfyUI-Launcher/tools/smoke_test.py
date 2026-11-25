"""Smoke test of starting and stopping instances, without a window and without ComfyUI.

It puts a small fake main.py into /tmp/rcl-smoke inside WSL and drives the real
process code against it: a stop before ComfyUI is up, a restart that is
cancelled, wsl.exe ended from outside, a forced stop, and a normal stop. After
each case it checks that no process is left inside WSL. Ports 8291 to 8293 are
used; nothing touches a real ComfyUI or a GPU.

Run from the project folder:  python tools/smoke_test.py [distro]
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6.QtCore import QCoreApplication, QTimer

from rulz_comfy_launcher import wsl
from rulz_comfy_launcher.instance import InstanceManager, State
from rulz_comfy_launcher.model import Profile
from rulz_comfy_launcher.options import builtin_catalog
from rulz_comfy_launcher.winjob import JobObject

FAKE = r'''
import sys, time, signal, os, socket, subprocess
port = int(sys.argv[sys.argv.index("--port") + 1])
if "--stubborn" in sys.argv:
    signal.signal(signal.SIGINT, lambda *a: print("ignoring SIGINT", flush=True))
    signal.signal(signal.SIGTERM, lambda *a: print("ignoring SIGTERM", flush=True))
child = subprocess.Popen(["sleep", "4%d" % port])   # a child process, to check group signalling
s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind(("127.0.0.1", port)); s.listen(5)
print("To see the GUI go to: http://127.0.0.1:%d" % port, flush=True)
try:
    while True:
        time.sleep(0.5)
except KeyboardInterrupt:
    print("\nStopped server", flush=True)
'''
SETUP = f"""
rm -rf /tmp/rcl-smoke && mkdir -p /tmp/rcl-smoke/venv/bin && cd /tmp/rcl-smoke || exit 1
cat > main.py <<'RCL_EOF'
{FAKE}
RCL_EOF
ln -s "$(command -v python3)" venv/bin/python
echo setup-ok
"""
CHECK = "pgrep -af 'rcl-smoke|main.py --port 829|sleep 4829|sleep 77' | grep -v pgrep || echo NONE\n"

app = QCoreApplication(sys.argv)
manager = InstanceManager(JobObject())
catalog = builtin_catalog()
manager.output.connect(lambda pid, text: sys.stdout.write(text.replace("\r", "~")))
log = []
manager.state_changed.connect(lambda pid: (log.append((pid, manager.state(pid).value)),
                                           print(f"<<{pid} -> {manager.state(pid).value}>>")))

D = sys.argv[1] if len(sys.argv) > 1 else ""   # "" = the default WSL distro
a = Profile(id="aaa", name="A", distro=D, comfy_dir="/tmp/rcl-smoke", port=8291)
slow = Profile(id="slo", name="Slow pre-command", distro=D, comfy_dir="/tmp/rcl-smoke", port=8293)
slow.pre_commands = "sleep 77"
stub = Profile(id="stb", name="Stubborn", distro=D, comfy_dir="/tmp/rcl-smoke", port=8292, extra_args="--stubborn")
results = {}
steps = []


def step(delay, fn):
    steps.append((delay, fn))


def run():
    if steps:
        delay, fn = steps.pop(0)
        QTimer.singleShot(delay, lambda: (fn(), run()))


def linux(label):
    c = wsl.run_bash(CHECK, D, parent=app)
    c.finished.connect(lambda code, out, err: (results.__setitem__(label, out.strip()),
                                               print(f"[{label}] linux processes: {out.strip()}")))


def expect(label, cond):
    results[label] = "ok" if cond else "FAIL"
    print(f"[{label}] {'ok' if cond else 'FAIL'}")


gap_stop = {"armed": False, "done": False}


def on_state(pid):
    # Press Stop exactly in the gap between "old process ended" and "new one starts".
    if gap_stop["armed"] and pid == "aaa" and manager.state("aaa") == State.QUEUED and not manager.instance("aaa").active:
        gap_stop["armed"] = False
        gap_stop["done"] = True
        manager.stop("aaa")


manager.state_changed.connect(on_state)


def kill_outer():
    pid = manager.instance("aaa")._proc.processId()
    out = subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True, text=True)
    print("killed only the outer wsl.exe:", out.stdout.strip() or out.stderr.strip())


def begin(code, out, err):
    print("setup:", out.strip())
    # 1. stop while the script is still in the pre-commands (ComfyUI's pid not known yet)
    step(0, lambda: manager.start(slow, catalog, 3))
    step(2500, lambda: (expect("1 pid unknown before stop", manager.instance("slo").linux_pid == 0), manager.stop("slo")))
    step(2500, lambda: expect("1 state after stop", manager.state("slo") == State.STOPPED))
    step(0, lambda: linux("1 nothing left"))
    # 2. restart, then Stop inside the restart gap -> must stay stopped
    step(1500, lambda: manager.start(a, catalog, 3))
    step(3500, lambda: (expect("2 running", manager.state("aaa") == State.RUNNING),
                        gap_stop.__setitem__("armed", True), manager.restart(a, catalog, 3)))
    step(4000, lambda: (expect("2 stop landed in the gap", gap_stop["done"]),
                        expect("2 stays stopped", manager.state("aaa") == State.STOPPED and not manager.any_active())))
    step(0, lambda: linux("2 nothing left"))
    # 3. the Windows wsl.exe is ended from outside (what the job object does when the launcher dies)
    step(1500, lambda: manager.start(a, catalog, 3))
    step(3500, lambda: (expect("3 running", manager.state("aaa") == State.RUNNING), kill_outer()))
    step(3000, lambda: expect("3 reported as failed", manager.state("aaa") == State.FAILED))
    step(0, lambda: linux("3 nothing left"))
    # 4. force stop of an instance that ignores SIGINT and SIGTERM
    step(1500, lambda: manager.start(stub, catalog, 30))
    step(3500, lambda: (expect("4 running", manager.state("stb") == State.RUNNING), manager.stop("stb")))
    step(1500, lambda: (expect("4 still stopping", manager.state("stb") == State.STOPPING), manager.force_stop("stb")))
    step(3000, lambda: expect("4 stopped after force", manager.state("stb") == State.STOPPED))
    step(0, lambda: linux("4 nothing left"))
    # 5. normal stop reaches child processes too (process group)
    step(1500, lambda: manager.start(a, catalog, 3))
    step(3500, lambda: manager.stop("aaa"))
    step(3000, lambda: expect("5 stopped", manager.state("aaa") == State.STOPPED))
    step(0, lambda: linux("5 nothing left"))
    step(2500, finish)
    run()


def finish():
    wsl.run_bash("rm -rf /tmp/rcl-smoke\n", D, parent=app).finished.connect(lambda *a: None)
    print("\nRESULTS")
    for key, value in results.items():
        print(f"  {key}: {value}")
    QTimer.singleShot(1500, app.quit)


wsl.run_bash(SETUP, D, parent=app).finished.connect(begin)
QTimer.singleShot(110000, app.quit)
sys.exit(app.exec())
