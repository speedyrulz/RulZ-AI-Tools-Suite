"""The block audit (blockaudit.py): grouping of the trained blocks, the paired credit maths, the log line, the saved state
and its resume, the config's refusals, switching a group's adapter off for every kind (lora, lokr, oft, dora), and that
the audit's images are not read as probe images."""
import json
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import blockaudit as BA  # noqa: E402
from rulztrainer import lora as L  # noqa: E402
from rulztrainer import runreport as RR  # noqa: E402
from rulztrainer.config import TrainConfig  # noqa: E402


# ---------------------------------------------------------------- grouping
def test_split_even_is_contiguous_and_uneven_parts_put_the_longer_ones_first():
    assert BA.split_even(list(range(10)), 3) == [[0, 1, 2, 3], [4, 5, 6], [7, 8, 9]]
    assert BA.split_even(list(range(6)), 3) == [[0, 1], [2, 3], [4, 5]]
    assert BA.split_even([4, 9, 10], 6) == [[4], [9], [10]]  # fewer blocks than groups: one each
    assert BA.split_even([], 4) == []


def test_span_text_is_compact():
    assert BA.span_text([0, 1, 2, 5, 7, 8]) == "0-2,5,7-8" and BA.span_text([3]) == "3" and BA.span_text([]) == "none"


def test_groups_cut_the_trained_blocks_and_modules_in_no_block_form_other():
    block_of = {f"layers.{i}.qkv": i for i in range(10, 17)}  # only blocks 10-16 are trained
    block_of.update({f"layers.{i}.proj": i for i in range(10, 17)})
    block_of["text_embedder.proj"] = None
    groups = BA.build_groups(block_of, 3)
    assert [g["name"] for g in groups] == ["g1", "g2", "g3", "other"]
    assert [g["blocks"] for g in groups[:3]] == [[10, 11, 12], [13, 14], [15, 16]]
    assert groups[0]["span"] == "10-12" and groups[0]["modules"] == ["layers.10.proj", "layers.10.qkv",
                                                                     "layers.11.proj", "layers.11.qkv",
                                                                     "layers.12.proj", "layers.12.qkv"]
    assert groups[3]["modules"] == ["text_embedder.proj"] and groups[3]["blocks"] == []
    assert sum(len(g["modules"]) for g in groups) == len(block_of)  # every module in exactly one group
    assert BA.build_groups({"head": None}, 4) == [{"name": "other", "blocks": [], "span": "no block",
                                                   "modules": ["head"]}]


class Tiny(nn.Module):
    def __init__(self, n=5):
        super().__init__()
        self.layers = nn.ModuleList([nn.ModuleDict({"qkv": nn.Linear(8, 8), "proj": nn.Linear(8, 8)})
                                     for _ in range(n)])
        self.head = nn.Linear(8, 8)

    def forward(self, x):
        for blk in self.layers:
            x = blk["proj"](blk["qkv"](x))
        return self.head(x)


def test_block_map_finds_each_module_by_identity_whatever_the_names():
    model = Tiny()
    names = [f"layers.{i}.{k}" for i in range(5) for k in ("qkv", "proj")] + ["head"]
    injected = L.inject_lora(model, 4, 4.0, [], names=names)
    mapping = BA.block_map(list(model.layers), injected)
    assert mapping["layers.3.qkv"] == 3 and mapping["layers.0.proj"] == 0 and mapping["head"] is None
    groups = BA.build_groups(mapping, 2)
    assert [g["blocks"] for g in groups[:2]] == [[0, 1, 2], [3, 4]] and groups[2]["name"] == "other"


# ---------------------------------------------------------------- credit maths
def test_credit_is_paired_by_seed_and_skips_seeds_scored_in_only_one():
    normal = {"face": [60.0, 58.0, 62.0, None], "body": [40.0, 42.0], "gold": []}
    ablated = {"face": [50.0, 55.0, None, 30.0], "body": [41.0, 40.0]}
    c = BA.credit(normal, ablated)
    assert c["face"]["n"] == 2 and c["face"]["mean"] == pytest.approx(6.5)  # (10 + 3) / 2: seeds 3 and 4 unpaired
    assert c["face"]["sd"] == pytest.approx(4.95, abs=0.01)
    assert c["face"]["normal"] == pytest.approx(59.0) and c["face"]["ablated"] == pytest.approx(52.5)
    assert c["body"]["mean"] == pytest.approx(0.5) and "gold" not in c


def test_credit_on_the_normalized_scale_and_with_one_pair():
    c = BA.credit({"face": [60.0]}, {"face": [50.0]}, norm=lambda kind, v: v / 2)
    assert c["face"]["mean"] == 10.0 and c["face"]["sd"] is None and c["face"]["norm"] == 5.0
    assert BA.credit({"face": [None]}, {"face": [1.0]}) == {}


def test_frame_credit_is_the_drift_added_by_switching_the_group_off():
    assert BA.frame_credit(0.1, 0.35) == {"normal": 0.1, "ablated": 0.35, "credit": 0.25}
    assert BA.frame_credit(None, 0.3) is None


def test_spearman_ties_and_stability():
    assert BA.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0
    assert BA.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0
    assert BA.spearman([1, 2], [1, 2]) is None and BA.spearman([1, 1, 1], [1, 2, 3]) is None
    assert BA.spearman([1, 2, 2, 4], [1, 3, 2, 5]) == pytest.approx(0.9487, abs=1e-3)
    rows = [{"step": s, "groups": [{"name": f"g{i}", "credit": {"face": {"mean": v}}} for i, v in enumerate(vals)]}
            for s, vals in ((100, [5, 3, 1]), (200, [6, 4, 2]), (300, [1, 4, 6]))]
    assert BA.stability(rows) == [(200, 1.0), (300, -1.0)]


def _row(step=600):
    return {"step": step, "seeds": 2, "renders": 8, "seconds": 41.0, "groups": [
        {"name": "g1", "span": "0-9", "blocks": list(range(10)),
         "credit": {"face": {"mean": 4.1, "sd": 1.2, "n": 2}, "body": {"mean": 0.34, "sd": None, "n": 1}},
         "frame": {"normal": 0.0, "ablated": 0.25, "credit": 0.25}},
        {"name": "other", "span": "no block", "blocks": [], "credit": {"face": {"mean": -0.4, "sd": 0.2, "n": 2}},
         "frame": None}]}


def test_describe_line():
    line = BA.describe(_row())
    assert line.startswith("g1 blocks 0-9 face +4.1±1.2 body +0.3 frame drift +25 points | other (no block) face -0.4±0.2")
    assert line.endswith("(8 renders, 41s, 2 seeds)")


# ---------------------------------------------------------------- state and resume
def test_ticks_count_probes_after_the_baseline_and_every_nth_is_due(tmp_path):
    a = BA.BlockAudit(tmp_path, every=2)
    a.set_groups(BA.build_groups({"layers.0.q": 0, "layers.1.q": 1}, 2))
    assert [a.tick(s) for s in (0, 100, 100, 200, 300, 400)] == [False, False, False, True, False, True]
    assert a.probes_seen == 4


def test_record_state_round_trip_and_resume(tmp_path):
    groups = BA.build_groups({"layers.0.q": 0, "layers.1.q": 1}, 2)
    a = BA.BlockAudit(tmp_path, every=1)
    a.set_groups(groups)
    assert a.tick(100) and a.tick(200)
    a.record(_row(100))
    a.record(_row(200))
    assert [r["step"] for r in BA.read_log(tmp_path)] == [100, 200]
    assert json.loads((tmp_path / BA.STATE_FILE).read_text(encoding="utf-8"))["probes_seen"] == 2
    b = BA.BlockAudit(tmp_path, every=1, resume=True)
    assert b.set_groups(groups) is True
    assert b.probes_seen == 2 and b.last_step == 200 and [r["step"] for r in b.history] == [100, 200]
    assert not b.tick(200) and b.tick(300)  # the step already seen is not counted twice
    c = BA.BlockAudit(tmp_path, every=1, resume=True)  # another cut: the saved audits are not this run's
    assert c.set_groups(BA.build_groups({"layers.0.q": 0, "layers.1.q": 1, "layers.2.q": 2}, 3)) is False
    assert c.history == [] and c.probes_seen == 0
    assert BA.read_log(tmp_path) == []  # the other cut's rows are not mixed into this run's history
    BA.BlockAudit(tmp_path, every=1)  # a new run clears the old files
    assert BA.read(tmp_path) is None and BA.read_log(tmp_path) == []


def test_resume_from_an_earlier_checkpoint_drops_the_later_audits(tmp_path):
    groups = BA.build_groups({"layers.0.q": 0, "layers.1.q": 1}, 2)
    a = BA.BlockAudit(tmp_path, every=1)
    a.set_groups(groups)
    for s in (100, 200, 300):
        assert a.tick(s)
        a.record(_row(s))
    b = BA.BlockAudit(tmp_path, every=1, resume=True)
    assert b.set_groups(groups, 200) is True
    assert [r["step"] for r in b.history] == [100, 200] and b.last_step is None and b.probes_seen == 0
    assert [r["step"] for r in BA.read_log(tmp_path)] == [100, 200]
    c = BA.BlockAudit(tmp_path, every=1, resume=True)  # resuming at the latest step keeps everything
    assert c.set_groups(groups, 300) is True and len(c.history) == 3 and c.probes_seen == 3


def test_audit_image_names():
    assert BA.is_audit_image("run_" + BA.image_tag(2, 600) + "_face_0.png")
    assert not BA.is_audit_image("run_step000600_face_0.png")


def test_a_failure_turns_the_audits_off(tmp_path):
    a = BA.BlockAudit(tmp_path, every=1)
    assert a.tick(100)
    a.fail(100, RuntimeError("out of memory"))
    assert a.failed and not a.tick(200)


# ---------------------------------------------------------------- the config
def _cfg(**kw):
    base = dict(dataset_dir=".", probe_every=100, face_refs=["a.png"], trigger_word="tok", block_audit_every=2)
    return TrainConfig(**{**base, **kw})


def test_the_defaults_are_off_and_a_good_config_passes():
    d = TrainConfig()
    assert (d.block_audit_every, d.block_audit_groups, d.block_audit_seeds) == (0, 6, 0)
    d.validate_block_audit()
    _cfg(block_audit_groups=16, block_audit_seeds=2).validate_block_audit()
    _cfg(block_audit_every=0, probe_every=0).validate_block_audit()  # off needs nothing


@pytest.mark.parametrize("kw, match", [
    (dict(block_audit_every=-1), "block_audit_every must be 0"),
    (dict(block_audit_groups=1), "block_audit_groups must be 2 to 16"),
    (dict(block_audit_groups=17), "block_audit_groups must be 2 to 16"),
    (dict(block_audit_seeds=17), "block_audit_seeds must be 0"),
    (dict(probe_every=0), "set probe_every > 0 and face_refs"),
    (dict(face_refs=[]), "set probe_every > 0 and face_refs"),
    (dict(noise_files=["a.safetensors"]), "noise_files measures"),
    (dict(soup_files=["a.safetensors", "b.safetensors"]), "soup_files averages"),
    (dict(finetune=True), "no adapter to switch off"),
    (dict(race=True), "takes turns"),
    (dict(probe_async=True), "probe_async scores weights"),
    (dict(compile_model=True), "compiled block guards"),
])
def test_the_config_refuses_what_cannot_be_paired(kw, match):
    with pytest.raises(ValueError, match=match):
        _cfg(**kw).validate_block_audit()


def test_compile_auto_and_a_sampling_model_are_allowed():
    _cfg(compile_model="auto").validate_block_audit()
    _cfg(sample_model_path="some/turbo-model").validate_block_audit()


def test_the_settings_survive_a_yaml_round_trip(tmp_path):
    cfg = _cfg(block_audit_groups=4, block_audit_seeds=2)
    path = tmp_path / "c.yaml"
    cfg.save(path)
    back = TrainConfig.load(path)
    assert (back.block_audit_every, back.block_audit_groups, back.block_audit_seeds) == (2, 4, 2)


# ---------------------------------------------------------------- switching a group off
CASES = [("lora", False), ("lora", True), ("lokr", False), ("lokr", True), ("oft", False)]


def _layer(kind, dora, seed=0):
    torch.manual_seed(seed)
    m = L.LoRALinear(nn.Linear(32, 32), 4, 4.0, kind=kind, factor=4, block=8, dora=dora)
    g = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for p in m.adapter_params().values():
            p.add_(torch.randn(p.shape, generator=g) * (0.05 if p.dim() == 3 else 0.2))
    return m


@pytest.mark.parametrize("kind,dora", CASES)
def test_a_switched_off_layer_is_the_frozen_layer_and_comes_back_exactly(kind, dora):
    m, other = _layer(kind, dora), _layer(kind, dora, seed=5)
    other.multiplier = 0.5  # a layer of another group, and a multiplier that is not 1
    x = torch.randn(3, 32)
    with torch.no_grad():
        full, half, base = m(x), other(x), m.base(x)
        assert not torch.allclose(full, base, atol=1e-4)  # the adapter does something
        with L.switched_off([{"a": m, "b": other}], ["a"]):
            assert m.multiplier == 0.0 and other.multiplier == 0.5  # only the named layer
            assert torch.equal(m(x), base)
            assert torch.equal(other(x), half)
        assert m.multiplier == 1.0 and torch.equal(m(x), full)
        with L.switched_off([{"a": m, "b": other}], ["a", "b"]):
            assert torch.equal(other(x), other.base(x))
        assert other.multiplier == 0.5 and torch.equal(other(x), half)  # its own earlier value, not 1


def test_every_copy_is_switched_off_and_all_come_back_after_a_failure():
    copies = [{"a": _layer("lora", False, seed=s)} for s in (1, 2, 3)]
    with pytest.raises(RuntimeError):
        with L.switched_off(copies, ["a", "missing"]):  # a name a copy lacks is skipped
            assert [c["a"].multiplier for c in copies] == [0.0, 0.0, 0.0]
            raise RuntimeError("render failed")
    assert [c["a"].multiplier for c in copies] == [1.0, 1.0, 1.0]


def test_the_extras_stay_on_when_the_adapter_is_off():
    m = _layer("lora", False)
    down, up = torch.randn(2, 32), torch.randn(32, 2)
    m.extras, m.extras_enabled = [("lora", down, up, 1.0)], True
    x = torch.randn(3, 32)
    with torch.no_grad(), L.switched_off([{"a": m}], ["a"]):
        with_extra = m(x)
        m.extras_enabled = False
        assert not torch.allclose(with_extra, m(x), atol=1e-5) and torch.equal(m(x), m.base(x))


# ---------------------------------------------------------------- the images
def test_audit_images_are_not_probe_images(tmp_path):
    tag = BA.image_tag(2, 600)
    assert tag == "audit-g2-s000600" and "_step" not in tag and not tag.startswith("step")
    for name in (f"{tag}_face_0.png", f"{tag}_body_1.png"):
        assert RR.PROBE_RE.match(name) is None
        (tmp_path / name).write_bytes(b"x")
    (tmp_path / "step000600_face_0.png").write_bytes(b"x")
    (tmp_path / "relay_step000600_body_0.png").write_bytes(b"x")  # a race configuration's own probe
    found = RR.probe_images(tmp_path)
    assert sorted(found) == [("", 600, "face"), ("relay", 600, "body")]
    # and no probe record holds them: the audit never calls the controller or append_record
    import inspect

    from rulztrainer.trainer import Trainer

    audit_src = inspect.getsource(Trainer._run_block_audit)
    assert "controller.update" not in audit_src and "append_record" not in audit_src


# ---------------------------------------------------------------- the run report
def test_the_report_reads_the_audit_and_draws_its_table(tmp_path):
    pytest.importorskip("matplotlib")
    rows = [_row(100), _row(200)]
    rows[1]["groups"][0]["credit"]["face"]["mean"] = 6.0
    groups, steps, grid = RR.audit_credits(rows, "face")
    assert groups == [("g1", "0-9"), ("other", "no block")] and steps == [100, 200]
    assert grid == [[4.1, 6.0], [-0.4, -0.4]]
    assert RR.audit_credits(rows, "gold")[2] == [[None, None], [None, None]]
    plt = RR._mpl()
    fig, ax = plt.subplots()
    assert RR.audit_heat(ax, rows, "face") is True and RR.audit_heat(ax, rows, "gold") is False
    plt.close(fig)
    assert "needs two or more audits" in RR.audit_note(rows)  # two groups: too few to rank
    three = [{"step": st, "groups": [{"name": f"g{i}", "credit": {"face": {"mean": v}}}
                                     for i, v in enumerate(vals)]} for st, vals in ((100, [5, 3, 1]), (200, [6, 4, 2]))]
    assert "Spearman" in RR.audit_note(three) and "mean +1.00" in RR.audit_note(three)
    assert "needs two or more" in RR.audit_note(three[:1])
    (tmp_path / "block_audit.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\nnot json\n",
                                                encoding="utf-8")
    assert [r["step"] for r in RR.load(tmp_path).audits] == [100, 200]


# ---------------------------------------------------------------- the trainer's audit, with the rendering stubbed
def test_the_trainer_audits_each_group_off_in_turn_and_changes_nothing_else(tmp_path):
    import contextlib
    from types import SimpleNamespace

    from rulztrainer.trainer import Trainer

    model = Tiny(4)
    names = [f"layers.{i}.{k}" for i in range(4) for k in ("qkv", "proj")]
    injected = L.inject_lora(model, 4, 4.0, [], names=names)
    groups = BA.build_groups(BA.block_map(list(model.layers), injected), 2)
    cfg = TrainConfig(dataset_dir=".", probe_every=100, face_refs=["a.png"], probe_face_prompt="tok, a headshot",
                      probe_seeds=2)
    t = object.__new__(Trainer)
    t.cfg, t.block_audit = cfg, BA.BlockAudit(tmp_path, every=1)
    t.block_audit.set_groups(groups)
    t.controller = SimpleNamespace(normalize=True, check_ref=None, norm=lambda kind, v: v / 10.0,
                                   update=lambda *a: pytest.fail("the audit must not feed the controller"))
    t.last_probe = (60.0, None, [], {"face": [60.0, 62.0], "body": [None, None]})
    t.status = SimpleNamespace(data={}, update=lambda **kw: t.status.data.update(kw))
    t._free_for_rendering = lambda: None
    seen = []

    @contextlib.contextmanager
    def slots(model_, step):
        yield [SimpleNamespace(model=model)]

    def probe_with(slots_, step, tag="", n_seeds=None, check=None):
        off = sorted(n for n, m in injected.items() if m.multiplier == 0.0)
        seen.append((tag, n_seeds, check, off))
        drop = 10.0 if off == groups[0]["modules"] else 2.0
        return 0.0, None, ["x.png"] * 2, {"face": [60.0 - drop, 62.0 - drop], "body": [None, None]}

    t._sampling_slots, t._probe_with = slots, probe_with
    assert t.block_audit.tick(300)
    t._run_block_audit(model, 300)
    assert [s[:3] for s in seen] == [("audit-g1-s000300", 2, False), ("audit-g2-s000300", 2, False)]
    assert [s[3] for s in seen] == [g["modules"] for g in groups]  # one group off at a time
    assert all(m.multiplier == 1.0 for m in injected.values())  # and everything back
    row = BA.read_log(tmp_path)[0]
    assert row["step"] == 300 and row["renders"] == 4 and row["seeds"] == 2
    assert [g["credit"]["face"]["mean"] for g in row["groups"]] == [10.0, 2.0]
    assert row["groups"][0]["credit"]["face"]["norm"] == 1.0 and row["normal"]["face"] == 61.0
    assert t.status.data["block_audit"]["line"].startswith("g1 blocks 0-1 face +10.0")
    # a failure is never fatal: one warning and the audits stop
    t._probe_with = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("render failed"))
    t._block_audit_after_probe(model, 400)
    assert t.block_audit.failed and all(m.multiplier == 1.0 for m in injected.values())
