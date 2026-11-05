"""Hull's region gains (docs/hull-optimizer.md sections 5.1-5.3): the parts tensor, the LoRA backward and the optimizer's
excess. The design's M1b tests that need no model."""
import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from rulztrainer import lora as L
from rulztrainer import regions as R
from rulztrainer.config import TrainConfig
from rulztrainer.optimizers.hull import Hull


def layer(o=6, i=8, r=3, seed=0):
    torch.manual_seed(seed)
    m = L.LoRALinear(nn.Linear(i, o, bias=False), rank=r, alpha=r)
    m.lora_up.weight.data.normal_(0, 0.1)
    return m


def sink():
    got = []
    return got, lambda pa, pb, x_a, x_b, f_a, f_b: got.append((x_a, x_b, f_a, f_b))


def layout(n_img, text=0):
    return lambda: {"sample": 0, "img": (text, n_img), "txt": (0, text), "mask": None}


def run(m, x, target):
    out = m._added(x)
    ((out - target) ** 2).sum().backward()
    return out


def test_the_backward_gives_autograds_gradients_and_the_excess():
    m, x, target = layer(), torch.randn(2, 5, 8, requires_grad=True), torch.randn(2, 5, 6)
    plain = run(m, x, target).detach()
    grads = [m.lora_down.weight.grad.clone(), m.lora_up.weight.grad.clone(), x.grad.clone()]
    for p in (m.lora_down.weight, m.lora_up.weight, x):
        p.grad = None
    extra = [torch.tensor([0.5, 0.5, 0.0, 0.0, 0.0]), torch.tensor([0.0, 0.0, 0.0, 0.25, 0.25])]
    got, fn = sink()
    m.hull_arm = L.HullArm(extra, [None, None], layout(5), fn)
    armed = run(m, x, target)
    assert torch.equal(armed.detach(), plain)
    for p, want in zip((m.lora_down.weight, m.lora_up.weight, x), grads):
        assert torch.allclose(p.grad, want, atol=1e-6)
    # the excess by hand: one more backward with the output's gradient scaled by w - 1 per token
    m.hull_arm = None
    e = torch.stack(extra)[:, :, None]
    m.zero_grad()
    m._added(x).backward(2 * (plain - target) * e)
    assert torch.allclose(got[0][0], m.lora_down.weight.grad, atol=1e-6)
    assert torch.allclose(got[0][1], m.lora_up.weight.grad, atol=1e-6)


def test_text_tokens_take_no_gain_and_a_text_only_input_stays_plain():
    arm = L.HullArm([torch.tensor([1.0, 1.0, 1.0])], [torch.tensor([True, False, False])], layout(3, text=2), None)
    extra, face = arm.tokens(torch.zeros(1, 5, 8))
    assert extra.tolist() == [[0.0, 0.0, 1.0, 1.0, 1.0]]
    assert face.tolist() == [[0.0, 0.0, 1.0, 0.0, 0.0]]
    assert arm.tokens(torch.zeros(1, 2, 8)) is None
    assert arm.tokens(torch.zeros(1, 3, 8))[0].tolist() == [[1.0, 1.0, 1.0]]
    assert L.HullArm([None], [None], layout(3), None).tokens(torch.zeros(1, 3, 8)) is None


def test_a_checkpointed_layer_reports_once_with_the_same_excess():
    m, x, target = layer(), torch.randn(1, 4, 8, requires_grad=True), torch.randn(1, 4, 6)
    got, fn = sink()
    m.hull_arm = L.HullArm([torch.tensor([0.5, 0.0, 0.5, 0.0])], [None], layout(4), fn)
    run(m, x, target)
    m.zero_grad()
    assert len(got) == 1
    ((checkpoint(m._added, x, use_reentrant=False) - target) ** 2).sum().backward()
    assert len(got) == 2
    assert torch.allclose(got[0][1], got[1][1], atol=1e-6)


def pair(o=12, i=16, r=4, seed=0):
    torch.manual_seed(seed)
    a = nn.Parameter(torch.randn(r, i, dtype=torch.float32) * 0.1)
    b = nn.Parameter(torch.randn(o, r, dtype=torch.float32) * 0.1)
    return a, b


def hull(a, b, **kw):
    return Hull([a, b], pairs=[(a, b)], ref_norms=[1.0], scales=[1.0], lr=1e-3, **kw)


def twins(steps, excess, **kw):
    """The same gradients through a plain Hull and one armed with excess(ga, gb) -> (x_a, x_b, f_a, f_b)."""
    a0, b0 = pair()
    a1, b1 = pair()
    plain, armed = hull(a0, b0, **kw), hull(a1, b1, **kw)
    torch.manual_seed(1)
    for _ in range(steps):
        ga, gb = torch.randn_like(a0), torch.randn_like(b0)
        plain.zero_grad()
        armed.zero_grad()
        a0.grad, b0.grad, a1.grad, b1.grad = ga, gb, ga.clone(), gb.clone()
        armed.add_excess(a1, b1, *excess(ga, gb))
        plain.step()
        armed.step()
    return (a0, b0, plain), (a1, b1, armed)


def test_a_zero_excess_is_h0_exactly():
    (a0, b0, _), (a1, b1, armed) = twins(12, lambda ga, gb: (torch.zeros_like(ga), torch.zeros_like(gb), None, None))
    assert torch.equal(a0, a1) and torch.equal(b0, b1)
    assert armed.state[b1]["armed"].item() == 12


def test_gains_reach_the_momentum_but_never_v_or_u():
    (a0, b0, plain), (a1, b1, armed) = twins(1, lambda ga, gb: (0.5 * ga, 0.5 * gb, None, None))
    assert not torch.equal(plain.state[a0]["m"], armed.state[a1]["m"])
    assert torch.equal(plain.state[a0]["v"], armed.state[a1]["v"])
    assert torch.equal(plain.state[b0]["v"], armed.state[b1]["v"])
    assert torch.equal(plain.state[b0]["u"], armed.state[b1]["u"])


def test_a_uniform_gain_scales_the_momentum():
    (_, b0, plain), (_, b1, armed) = twins(1, lambda ga, gb: (ga.clone(), gb.clone(), None, None), clip=1e6)
    assert torch.allclose(armed.state[b1]["m"], 2 * plain.state[b0]["m"], rtol=1e-5)


def gained(opt, x_a, x_b, f_a, f_b, ga, gb):
    """_gained at the identity frame and unit normaliser."""
    t = torch.eye(ga.shape[0])[None]
    region = (x_a[None], x_b[None], f_a[None], f_b[None], torch.tensor([True]))
    return opt._gained({"region": region}, ga[None], gb[None], gb[None], t, t, torch.ones(1, ga.shape[1]),
                       torch.ones(1, gb.shape[0]), torch.ones(1), torch.zeros(1), torch.tensor([False]))


def test_the_excess_cannot_undo_the_face():
    a, b = pair()
    opt = hull(a, b)
    ga, gb, f_a, f_b = torch.randn_like(a), torch.randn_like(b), torch.randn_like(a), torch.randn_like(b)
    full_a, full_b, _, _ = gained(opt, -3 * f_a, -3 * f_b, f_a, f_b, ga, gb)
    assert torch.allclose(full_a[0], ga, atol=1e-5)  # an excess against the face goes
    assert torch.allclose(full_b[0], gb, atol=1e-5)
    full_a, _, _, _ = gained(opt, 3 * f_a, 3 * f_b, f_a, f_b, ga, gb)
    assert torch.allclose(full_a[0], ga + 3 * f_a, atol=1e-5)  # an excess along the face stays
    x = torch.randn_like(b)
    other = (x - 0.9 * f_b)
    _, full_b, _, _ = gained(opt, torch.zeros_like(a), other, torch.zeros_like(a), f_b, ga, gb)
    assert ((full_b[0] - gb) * f_b).sum() >= -1e-4


def test_the_retreat_halves_a_tensor_that_opposes_its_gradient_and_lam_never_rises():
    a, b = pair()
    opt = hull(a, b, face_safe=False)
    for _ in range(3):
        opt.zero_grad()
        a.grad, b.grad = torch.randn_like(a), torch.randn_like(b)
        opt.add_excess(a, b, torch.zeros_like(a), -3 * b.grad, None, None)
        opt.step()
    assert opt.state[b]["cos"].item() < 0
    opt.on_probe(0.0, 0.0)
    assert opt.state[b]["lam"].item() == 0.5
    opt.on_probe(0.0, 0.0)  # no armed step since: nothing to read
    assert opt.state[b]["lam"].item() == 0.5
    for _ in range(40):  # the smoothed cosine turns positive: lam holds, and nothing raises it
        opt.zero_grad()
        a.grad, b.grad = torch.randn_like(a), torch.randn_like(b)
        opt.add_excess(a, b, torch.zeros_like(a), 3 * b.grad, None, None)
        opt.step()
    assert opt.state[b]["cos"].item() > 0
    before = opt.state[b]["lam"].item()
    opt.on_probe(0.0, 0.0)
    assert opt.state[b]["lam"].item() == before <= 0.5


def test_zero_grad_clears_the_excess_and_a_state_from_before_the_gains_loads():
    a, b = pair()
    opt = hull(a, b)
    a.grad, b.grad = torch.randn_like(a), torch.randn_like(b)
    opt.step()
    old = opt.state_dict()
    for st in old["state"].values():
        for key in ("cos", "cos_top", "armed", "interval", "lam"):
            st.pop(key, None)
    fresh = hull(a, b)
    fresh.load_state_dict(old)
    assert fresh.state[b]["lam"].item() == 1.0
    fresh.add_excess(a, b, torch.zeros_like(a), torch.zeros_like(b), None, None)
    fresh.zero_grad()
    assert not fresh._excess


def classes(h, w):
    """A 16 px grid: background, a body-skin column, a clothes column, with the head's cells as face skin."""
    c = np.zeros((h * 16, w * 16), dtype=np.uint8)
    c[:, 16:32] = 2
    c[:, 32:48] = 4
    c[:16, 16:48] = 3
    return c


def test_the_parts_tensor_and_its_gains():
    parts = R.build_parts(classes(6, 6), 96, 96, (1, 0, 3, 1), 1)
    assert parts.shape == (7, 6, 6) and parts.dtype == torch.uint8
    assert parts[:6].sum(0).min() >= 254
    assert parts[6, 0, 1:3].sum() == 0  # the ring stays out of the face box
    assert parts[6].sum() > 0
    gains, face = R.token_gains(parts, (1, 0, 3, 1), 1.5, 0.8)
    assert gains[0, 1:3].eq(1.0).all() and face[0, 1:3].all() and face.sum() == 2
    assert gains[3, 1].item() == 1.5  # body skin
    assert gains[3, 2].item() == 1.5  # clothing's edge lies on the ring
    assert abs(gains[3, 5].item() - 0.8) < 1e-6  # far background
    inner = R.build_parts(np.full((96, 96), 4, np.uint8), 96, 96, None, 0)
    assert R.token_gains(inner, None, 1.5, 0.8)[0].eq(1.0).all()  # clothing and hair never go above 1


def test_a_map_keeps_its_parts_and_a_mismatch_rebuilds(tmp_path):
    cfg = TrainConfig()
    rm = R.uniform_region_map(96, 96)
    rm.parts, rm.parts_print = R.build_parts(classes(6, 6), 96, 96, None, 1), R.parts_print(cfg, 96, 96)
    path = tmp_path / "x_regions.safetensors"
    R.save_region_map(path, rm)
    back = R.load_region_map(path)
    assert torch.equal(back.parts, rm.parts) and back.parts_print == rm.parts_print
    assert not R._parts_stale(path, R.parts_print(cfg, 96, 96))
    cfg.hull_ring_cells = 2
    assert R._parts_stale(path, R.parts_print(cfg, 96, 96))


@pytest.mark.parametrize("change", [dict(flip_augment=True), dict(region_weights=True), dict(multi_gpu="data_parallel"),
                                    dict(hull_gain_body=3.0), dict(hull_gain_background=1.2), dict(hull_retreat=1.0)])
def test_the_config_refuses_what_breaks_the_gains(change):
    cfg = TrainConfig(optimizer="hull", hull_rho=2e-4, hull_regions="parts", **change)
    with pytest.raises(ValueError, match="hull_"):
        cfg.validate_hull()


def _race_trainer(tmp_path):
    from rulztrainer.trainer import Trainer

    rows = [{"name": "plain", "hull_regions": "off"}, {"name": "parts", "hull_regions": "parts"}]
    cfg = TrainConfig(optimizer="hull", face_gold="facenet", copy_check="on", race=True, race_configs=rows,
                      race_drop_lowest=False, race_parallel=False, cache_dir=str(tmp_path))
    return Trainer.__new__(Trainer), cfg


def test_a_race_row_with_parts_gets_maps_when_the_run_itself_has_none(tmp_path):
    t, cfg = _race_trainer(tmp_path)
    t.race_region_dirs = {}
    assert not cfg.wants_regions()
    prep = t._race_prep_config(cfg)
    assert prep.wants_regions() and not prep.wants_parts()  # the dataset is prepared with maps; no row is changed
    assert [c.wants_regions() for _, c, _ in t.race_rows] == [False, True]


def test_a_row_with_parts_never_shares_another_rows_maps():
    from rulztrainer import race

    a = TrainConfig(optimizer="hull", face_gold="facenet", copy_check="on", region_weights=True)
    b = TrainConfig(optimizer="hull", face_gold="facenet", copy_check="on", hull_regions="parts")
    assert race.region_key(a) != race.region_key(b)
