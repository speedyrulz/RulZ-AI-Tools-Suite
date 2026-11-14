r"""RulzTrainer WebUI on a made-up two-instance vast.ai fleet, for screenshots only.

    set RULZ_VAST_HOME=R:\vast-home        (a fresh, empty folder)
    <venv>\python.exe -B rulz_trainer_fake_fleet.py <clean copy of apps\RulzTrainer> 7899

Swaps rulz_fleet.Fleet for one over tests/test_vast_fleet.py's fake machines (no ssh, no vast.ai), with demo run
names and a loss / likeness curve, then runs webui.py. Fleet state lives in RULZ_VAST_HOME (a fresh folder).
"""
import math
import os
import runpy
import sys
import types
from pathlib import Path

APP = Path(sys.argv[1]).resolve()
HOME = Path(os.environ["RULZ_VAST_HOME"])
PORT = sys.argv[2] if len(sys.argv) > 2 else "7899"

sys.path[:0] = [str(APP), str(APP / "scripts" / "vast"), str(APP / "tests")]
stub = types.ModuleType("pytest")  # the test module imports pytest only for its fixtures
stub.fixture = lambda *a, **k: (a[0] if a and callable(a[0]) else (lambda f: f))
stub.mark = types.SimpleNamespace(parametrize=lambda *a, **k: (lambda f: f), skipif=lambda *a, **k: (lambda f: f))
sys.modules.setdefault("pytest", stub)

import rulz_fleet  # noqa: E402
import test_vast_fleet as tf  # noqa: E402


class DemoMachine(tf.Machine):
    def snapshot(self):
        snap = super().snapshot()
        if self.running:
            job = self.jobs[self.running]
            step = job["step"]
            losses = [[s, round(0.21 + 0.33 * math.exp(-s / 450) + 0.012 * math.sin(s / 23), 4)]
                      for s in range(10, step + 1, 10)]
            probes = [{"step": s, "combined_ema": round(48 - 38 * math.exp(-s / 420), 1)} for s in range(200, step + 1, 200)]
            snap["series"] = {"losses": losses, "probes": probes}
            snap["job"].update(eta=11 * 60, stage="training", step_time=0.53,
                               last_probe={"step": 1200, "face_ema": 73.6, "body_ema": 48.2, "combined_ema": 47.0,
                                           "best": 47.0, "best_step": 1200})
        return snap


tf.Machine = DemoMachine
world = tf.World.__new__(tf.World)
world.machines = {"61204417": DemoMachine(61204417, ("character_set",)), "61207733": DemoMachine(61207733, ("character_set",))}
world.listed = lambda: tf.listing(world.machines.values())
world.fetched, world.sent, world.clock = [], [], tf.Clock()
HOME.mkdir(parents=True, exist_ok=True)
world.fleet = world.new_fleet(HOME)
world.fleet.discover()

m2 = world.machines["61207733"]
m2.run("style_adamw", total=2000, step=2000, elapsed=3900.0, avg_loss=0.2110)
m2.finish("done", done=True)
world.tick()
m1 = world.machines["61204417"]
m1.run("character_keel_r32", total=2000, step=1240, elapsed=2480.0, avg_loss=0.2075)
world.tick()
world.fleet.queue_edit("pause")  # paused, so the made-up queue stays put
world.fleet.enqueue("character_muon", {"steps": 2000, "optimizer": "muon"})
world.fleet.enqueue("style_prodigy", {"steps": 1500, "optimizer": "prodigy"})
world.fleet.enqueue("character_r16", {"steps": 1500, "rank": 16}, instance=61207733)

rulz_fleet.Fleet = lambda *a, **k: world.fleet
sys.argv = [str(APP / "webui.py"), "--port", PORT]
os.chdir(APP)
runpy.run_path(str(APP / "webui.py"), run_name="__main__")
