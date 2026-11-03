"""Hull's base rule H0 (docs/hull-optimizer.md section 11, M1a): the virtual balanced frame against explicit rebalancing,
gauge invariance, the frame's numerics and fallback, the first step, the step in dW units across ranks, the caps, the
skip on a non-finite gradient, no host sync, the state's round trip and race moves, and the wiring."""

import copy
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import lora as L  # noqa: E402
from rulztrainer import race as R  # noqa: E402
from rulztrainer.config import OPTIMIZERS, AUTOMAGIC, TrainConfig  # noqa: E402
from rulztrainer.optimizers.hull import Hull, balance_frame, newton_schulz  # noqa: E402
from rulztrainer import trainer as T  # noqa: E402

W0 = 10.0


def factors(o=96, i=128, r=8, dtype=torch.float64, seed=0):
    g = torch.Generator().manual_seed(seed)
    a = (torch.rand(r, i, dtype=dtype, generator=g) * 2 - 1) * (3.0 / i) ** 0.5
    return torch.nn.Parameter(a), torch.nn.Parameter(torch.zeros(o, r, dtype=dtype))


def make(a, b, rho=1e-3, s=1.0, **kw):
    kw.setdefault("state_dtype", "fp32")
    return Hull([a, b], [(a, b)], [W0], [s], lr=rho, **kw)


def target_of(o, i, dtype, rank=2, seed=7):
    g = torch.Generator().manual_seed(seed)
    t = torch.randn(o, rank, dtype=dtype, generator=g) @ torch.randn(rank, i, dtype=dtype, generator=g)
    return t / t.norm() * W0 * 0.5


def push(a, b, s, g):
    """dL/dW = g through dW = s B A, as autograd's factor gradients."""
    a.grad = s * b.detach().mT @ g
    b.grad = s * g @ a.detach().mT


def gradient(a, b, s, target, t, noise=0.02):
    with torch.no_grad():
        dw = s * b @ a
    g = dw - target
    g = g / g.norm()
    rng = torch.Generator().manual_seed(1000 + t)
    return g + noise * torch.randn(g.shape, dtype=g.dtype, generator=rng) / g.numel() ** 0.5


def run(a, b, opt, steps, s=1.0, target=None, after=None, noise=0.02, start=0):
    target = target_of(b.shape[0], a.shape[1], a.dtype) if target is None else target
    moved = []
    for t in range(start, start + steps):
        with torch.no_grad():
            before = s * b @ a
        push(a, b, s, gradient(a, b, s, target, t, noise))
        opt.step()
        with torch.no_grad():
            moved.append(float((s * b @ a - before).norm()))
        if after is not None:
            after(t)
    return moved


def balancing(b, a):
    """The exact SPD S with (B S)t(B S) = (S^-1 A)(S^-1 A)t, by eigendecompositions (the rule's test, not the rule)."""
    m, n = b.mT @ b, a @ a.mT

    def power(x, e):
        w, q = torch.linalg.eigh(x)
        return (q * w.clamp_min(1e-300).pow(e)) @ q.mT

    t = power(m, -0.5) @ power(power(m, 0.5) @ n @ power(m, 0.5), 0.5) @ power(m, -0.5)
    return power(t, 0.5), power(t, -0.5)


def rebalance(a, b, opt):
    with torch.no_grad():
        s, si = balancing(b.double(), a.double())
        s, si = s.to(a.dtype), si.to(a.dtype)
        b.copy_(b @ s)
        a.copy_(si @ a)
        opt.state[b]["m"].copy_(opt.state[b]["m"] @ si)
        opt.state[a]["m"].copy_(s @ opt.state[a]["m"])


@pytest.mark.parametrize("dtype,damping,iters,tol", [(torch.float64, 1e-12, 60, 1e-10), (torch.float64, 1e-6, 30, 1e-5),
                                                     (torch.float32, 1e-6, 30, 2e-3)])
def test_virtual_frame_equals_explicit_rebalancing_with_transported_momentum(dtype, damping, iters, tol):
    a1, b1 = factors(dtype=dtype)
    a2, b2 = factors(dtype=dtype)
    o1 = make(a1, b1, damping=damping, ns_iters=iters)
    o2 = make(a2, b2, damping=damping, ns_iters=iters)
    run(a1, b1, o1, 200)
    run(a2, b2, o2, 200, after=lambda t: rebalance(a2, b2, o2))
    with torch.no_grad():
        d1, d2 = b1 @ a1, b2 @ a2
    assert (d1 - d2).norm() / d1.norm() < tol


@pytest.mark.parametrize("cond,damping,iters,tol", [(5, 1e-12, 60, 1e-5), (100, 1e-12, 60, 1e-5), (100, 1e-6, 30, 2e-3)])
def test_a_gauged_start_gives_the_same_dw(cond, damping, iters, tol):
    a1, b1 = factors()
    a2, b2 = factors()
    g = torch.Generator().manual_seed(3)
    q = torch.linalg.qr(torch.randn(8, 8, dtype=torch.float64, generator=g))[0]
    s0 = (q * torch.logspace(0, torch.log10(torch.tensor(float(cond))).item(), 8, dtype=torch.float64)) @ q.mT
    with torch.no_grad():
        a2.copy_(torch.linalg.solve(s0, a2))
    o1 = make(a1, b1, damping=damping, ns_iters=iters)
    o2 = make(a2, b2, damping=damping, ns_iters=iters)
    run(a1, b1, o1, 60)
    run(a2, b2, o2, 60)
    with torch.no_grad():
        d1, d2 = b1 @ a1, b2 @ a2
    assert (d1 - d2).norm() / d1.norm() < tol


def test_a_step_from_a_gauged_pair_gives_the_same_dw():
    a1, b1 = factors()
    run(a1, b1, make(a1, b1), 5)
    g = torch.Generator().manual_seed(4)
    q = torch.linalg.qr(torch.randn(8, 8, dtype=torch.float64, generator=g))[0]
    s0 = (q * torch.logspace(0, 1.5, 8, dtype=torch.float64)) @ q.mT
    a2 = torch.nn.Parameter(torch.linalg.solve(s0, a1.detach()))
    b2 = torch.nn.Parameter(b1.detach() @ s0)
    o1, o2 = make(a1, b1, damping=1e-12, ns_iters=60), make(a2, b2, damping=1e-12, ns_iters=60)
    run(a1, b1, o1, 1, start=5)
    run(a2, b2, o2, 1, start=5)
    with torch.no_grad():
        assert (b1 @ a1 - b2 @ a2).norm() / (b1 @ a1).norm() < 1e-5


def test_newton_schulz_roots():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(4, 32, 64, dtype=torch.float64, generator=g)
    a = x @ x.mT
    a = a / a.diagonal(dim1=-2, dim2=-1).sum(-1)[:, None, None]
    y, z = newton_schulz(a, 30)
    eye = torch.eye(32, dtype=torch.float64)
    assert (y @ y - a).norm() / a.norm() < 1e-10 and (y @ z - eye).norm() < 1e-9


@pytest.mark.parametrize("held", [1, 4, 16, 32])
def test_the_frame_holds_for_a_low_rank_update_at_rank_32(held):
    """Why the roots are fp64: at this damping fp32 rounding leaves negative Gram eigenvalues and the iteration diverges
    for every rank below 32."""
    g = torch.Generator().manual_seed(0)
    a = torch.randn(32, 128, generator=g) / 128 ** 0.5
    b = 1e-2 * torch.randn(256, held, generator=g) @ torch.randn(held, 32, generator=g)
    ad, bd = a.double(), b.double()  # the Grams are fp64 too: in fp32 this b's Gram has negative eigenvalues
    m, n = bd.mT @ bd, ad @ ad.mT
    eye = torch.eye(32, dtype=torch.float64)
    m = (m + 1e-6 * m.trace() / 32 * eye)[None]
    n = (n + 1e-6 * n.trace() / 32 * eye)[None]
    t, ti, good = balance_frame(m, n, 30)
    assert good.all()
    assert (t @ m @ t - n).norm() / n.norm() < 1e-4 and (t @ ti - eye).norm() / 32 ** 0.5 < 1e-4


def test_the_frame_holds_through_a_run_at_rank_32():
    a, b = factors(o=96, i=128, r=32, dtype=torch.float32)
    opt = make(a, b, state_dtype="bf16")
    run(a, b, opt, 120)
    assert opt.report()["frame_fallbacks"] == 0


def test_a_failed_root_keeps_the_previous_frame():
    a, b = factors()
    opt = make(a, b)
    run(a, b, opt, 4)
    t_old = opt.state[b]["T"].clone()
    opt.ns_iters = 1  # one iteration cannot reach the root
    w0 = (a.detach().clone(), b.detach().clone())
    run(a, b, opt, 1, start=4)
    assert torch.equal(opt.state[b]["T"], t_old) and opt.report()["frame_fallbacks"] == 1
    assert not torch.equal(b.detach(), w0[1])  # the step still ran, in the old frame


@pytest.mark.parametrize("r,alpha", [(4, 4), (8, 16), (32, 32)])
def test_step_one_moves_only_b_along_the_row_normalised_gradient(r, alpha):
    a, b = factors(r=r)
    s = alpha / r
    opt = make(a, b, s=s, damping=1e-12)
    a0 = a.detach().clone()
    g = gradient(a, b, s, target_of(96, 128, a.dtype), 0)
    push(a, b, s, g)
    opt.step()
    assert torch.equal(a.detach(), a0)
    with torch.no_grad():
        dw = s * b @ a
    assert abs(float(dw.norm()) / (1e-3 * W0) - 1.0) < 1e-6
    proj = a0.mT @ torch.linalg.solve(a0 @ a0.mT, a0)
    rows = g @ proj
    rows = rows / rows.norm(dim=1, keepdim=True)
    assert (dw / dw.norm() + rows / rows.norm()).norm() < 1e-6
    assert float(opt.state[b]["k"]) == 0 and float(opt.state[b]["u"]) == 0  # the moments start at step 2


def test_a_pair_with_no_gradient_at_step_one_starts_later():
    a, b = factors()
    opt = make(a, b)
    a.grad, b.grad = torch.zeros_like(a), torch.zeros_like(b)
    opt.step()
    assert not b.detach().any()
    run(a, b, opt, 2)
    assert b.detach().any() and float(opt.state[b]["k"]) == 1


def test_the_realised_step_is_one_multiple_of_rho_across_ranks():
    multiples = []
    for r in (4, 8, 16, 32, 64):
        for alpha in (r, 16):
            a, b = factors(r=r, dtype=torch.float32)
            s = alpha / r
            opt = make(a, b, s=s, state_dtype="bf16")
            moved = run(a, b, opt, 20, s=s)
            multiples.append(sum(moved[5:]) / 15 / (1e-3 * W0))
    mid = sum(multiples) / len(multiples)
    assert all(abs(x / mid - 1) < 0.10 for x in multiples), multiples
    a, b = factors(r=16, dtype=torch.float32)
    opt = make(a, b, s=1.0)
    noise = torch.Generator().manual_seed(5)

    moved = []
    for t in range(200):
        with torch.no_grad():
            before = b @ a
        push(a, b, 1.0, torch.randn(96, 128, generator=noise))
        opt.step()
        with torch.no_grad():
            moved.append(float((b @ a - before).norm()))
    ratio = sum(moved[50:]) / 150 / (1e-3 * W0) / mid
    assert 0.1 < ratio < 0.3, ratio


def test_v_and_the_clip_ignore_a_rotation_of_the_rank_axis():
    a1, b1 = factors()
    run(a1, b1, make(a1, b1), 3)
    g = torch.Generator().manual_seed(4)
    q = torch.linalg.qr(torch.randn(8, 8, dtype=torch.float64, generator=g))[0]
    a2 = torch.nn.Parameter(q.mT @ a1.detach())
    b2 = torch.nn.Parameter(b1.detach() @ q)
    o1, o2 = make(a1, b1, clip=0.5), make(a2, b2, clip=0.5)
    run(a1, b1, o1, 2, start=3)
    run(a2, b2, o2, 2, start=3)
    assert torch.allclose(o1.state[b1]["v"], o2.state[b2]["v"], rtol=1e-8)
    assert torch.allclose(o1.state[a1]["v"], o2.state[a2]["v"], rtol=1e-8)
    with torch.no_grad():
        assert (b1 @ a1 - b2 @ a2).norm() / (b1 @ a1).norm() < 1e-7


def test_the_realised_step_never_exceeds_the_safety_cap():
    a, b = factors(r=8)
    opt = make(a, b, safety_cap=1.1)
    moved = run(a, b, opt, 60, noise=0.0)
    assert max(moved) <= 1.1 * 1e-3 * W0 * (1 + 1e-9)
    assert opt.report()["safety_rate"] > 0  # it did bind: a consistent layer reaches about 1.25 rho


def test_a_big_step_hits_the_factor_cap_early_on():
    a, b = factors(r=8)
    opt = make(a, b, rho=0.05)
    run(a, b, opt, 8)
    assert opt.report()["factor_rate"] > 0


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_a_bad_gradient_leaves_weights_and_state_bitwise(bad):
    a, b = factors(dtype=torch.float32)
    opt = make(a, b, state_dtype="bf16")
    run(a, b, opt, 5)
    keep = (a.detach().clone(), b.detach().clone(), copy.deepcopy(opt.state[a]), copy.deepcopy(opt.state[b]))
    push(a, b, 1.0, torch.ones(96, 128))
    b.grad[3, 2] = bad
    opt.step()
    assert torch.equal(a.detach(), keep[0]) and torch.equal(b.detach(), keep[1])
    for st, old in ((opt.state[a], keep[2]), (opt.state[b], keep[3])):
        for key, value in old.items():
            if torch.is_tensor(value):
                assert torch.equal(st[key], value), key


def test_one_bad_pair_does_not_stop_the_others():
    a1, b1 = factors(dtype=torch.float32)
    a2, b2 = factors(dtype=torch.float32, seed=1)
    opt = Hull([a1, b1, a2, b2], [(a1, b1), (a2, b2)], [W0, W0], [1.0, 1.0], lr=1e-3)
    for t in range(3):
        push(a1, b1, 1.0, torch.randn(96, 128))
        push(a2, b2, 1.0, torch.randn(96, 128))
        if t == 2:
            b2.grad[0, 0] = float("nan")
        keep = b2.detach().clone(), b1.detach().clone()
        opt.step()
    assert torch.equal(b2.detach(), keep[0]) and not torch.equal(b1.detach(), keep[1])


def test_the_step_scales_with_the_group_lr_and_ignores_the_gradient_scale():
    def moved(factor, lr_scale):
        a, b = factors()
        opt = make(a, b)
        target = target_of(96, 128, a.dtype)
        for t in range(30):  # past the first steps, where the factor cap binds
            push(a, b, 1.0, factor * gradient(a, b, 1.0, target, t))
            opt.step()
        opt.param_groups[0]["lr"] *= lr_scale
        with torch.no_grad():
            before = b @ a
        push(a, b, 1.0, factor * gradient(a, b, 1.0, target, 30))
        opt.step()
        with torch.no_grad():
            return float((b @ a - before).norm())

    assert abs(moved(1.0, 0.5) / moved(1.0, 1.0) - 0.5) < 0.01
    assert abs(moved(1000.0, 1.0) / moved(1.0, 1.0) - 1.0) < 1e-6


def test_stochastic_rounding_is_unbiased():
    gen = torch.Generator().manual_seed(0)
    x = torch.full((200_000,), 1.5 + 1.0 / 512, dtype=torch.float32)  # a quarter of a bf16 ulp above 1.5
    out = Hull._round_bf16(x, gen)
    assert out.dtype == torch.bfloat16 and set(out.unique().tolist()) == {1.5, 1.5078125}
    assert abs(out.float().mean().item() - x[0].item()) < 1e-4


def test_the_dither_leaves_the_runs_rng_alone():
    a, b = factors(dtype=torch.float32)
    opt = make(a, b, state_dtype="bf16")
    push(a, b, 1.0, torch.ones(96, 128))
    opt.step()
    torch.manual_seed(1)
    expected = torch.rand(3)
    torch.manual_seed(1)
    push(a, b, 1.0, torch.ones(96, 128))
    opt.step()
    assert torch.equal(torch.rand(3), expected)


def twins(steps=6):
    out = []
    for _ in range(2):
        a, b = factors(dtype=torch.float32)
        opt = make(a, b, state_dtype="bf16", seed=3)
        run(a, b, opt, steps)
        out.append((a, b, opt))
    return out


def test_the_state_round_trips_exactly_and_the_constructor_wins():
    (a1, b1, o1), (a2, b2, o2) = twins()
    saved = copy.deepcopy(o1.state_dict())
    a3 = torch.nn.Parameter(a1.detach().clone())
    b3 = torch.nn.Parameter(b1.detach().clone())
    o3 = make(a3, b3, state_dtype="bf16", seed=3, rho=9.0, clip=2.5)
    o3.load_state_dict(saved)
    assert o3.param_groups[0]["lr"] == 1e-3 and o3.clip == 2.5  # the scheduler's lr comes back, the options do not
    o3.clip = o1.clip
    assert o3.state[b3]["m"].dtype == torch.bfloat16 and o3.state[a3]["m"].dtype == torch.bfloat16
    for key, value in o1.state[b1].items():
        assert torch.equal(o3.state[b3][key], value) if torch.is_tensor(value) else o3.state[b3][key] == value
    run(a1, b1, o1, 3, start=6)
    run(a3, b3, o3, 3, start=6)
    assert torch.equal(b1.detach(), b3.detach()) and torch.equal(a1.detach(), a3.detach())


def test_next_steps_are_bitwise_after_a_race_move_and_a_relay():
    (a1, b1, o1), (a2, b2, o2) = twins()
    for st in o2.state.values():  # what move_optimizer does
        for k, t in list(st.items()):
            if torch.is_tensor(t):
                st[k] = t.clone()
    run(a1, b1, o1, 2, start=6)
    run(a2, b2, o2, 2, start=6)
    assert torch.equal(b1.detach(), b2.detach())
    a3, b3 = factors(dtype=torch.float32)
    o3 = make(a3, b3, state_dtype="bf16", seed=3)
    run(a3, b3, o3, 2)
    o3.state.clear()  # what follow_leader does, on the shared wrappers' tensors
    for p1, p3 in ((a1, a3), (b1, b3)):
        o3.state[p3] = {k: t.detach().clone() if torch.is_tensor(t) else copy.deepcopy(t) for k, t in o1.state[p1].items()}
        p3.data.copy_(p1.data)
    run(a1, b1, o1, 2, start=8)
    run(a3, b3, o3, 2, start=8)
    assert torch.equal(b1.detach(), b3.detach()) and torch.equal(a1.detach(), a3.detach())


def test_buckets_mix_shapes_and_ranks_and_match_one_pair_at_a_time():
    shapes = [(96, 128, 8), (96, 128, 8), (64, 32, 8), (64, 64, 4)]
    def pairs_of():
        return [factors(o, i, r, dtype=torch.float32, seed=n) for n, (o, i, r) in enumerate(shapes)]

    together = pairs_of()
    flat = [p for pair in together for p in pair]
    opt = Hull(flat, together, [W0] * 4, [1.0] * 4, lr=1e-3, state_dtype="fp32")
    apart = pairs_of()
    opts = [make(a, b) for a, b in apart]
    for t in range(4):
        grads = [torch.randn(o, i, generator=torch.Generator().manual_seed(t * 10 + n)) for n, (o, i, _) in enumerate(shapes)]
        for (a, b), g in zip(together, grads):
            push(a, b, 1.0, g)
        opt.step()
        for (a, b), g, o in zip(apart, grads, opts):
            push(a, b, 1.0, g)
            o.step()
    for (a1, b1), (a2, b2) in zip(together, apart):
        assert torch.equal(a1.detach(), a2.detach()) and torch.equal(b1.detach(), b2.detach())


def test_a_pair_without_gradients_is_left_alone():
    a1, b1 = factors()
    a2, b2 = factors(seed=1)
    opt = Hull([a1, b1, a2, b2], [(a1, b1), (a2, b2)], [W0, W0], [1.0, 1.0], lr=1e-3)
    push(a1, b1, 1.0, torch.ones(96, 128, dtype=torch.float64))
    opt.step()
    assert not b2.detach().any() and "m" not in opt.state[b2]


def test_no_host_sync_in_a_step():
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    device = torch.device("cuda", torch.cuda.device_count() - 1)
    a, b = factors(o=256, i=256, r=8, dtype=torch.float32)
    a = torch.nn.Parameter(a.detach().to(device))
    b = torch.nn.Parameter(b.detach().to(device))
    opt = make(a, b, state_dtype="bf16")  # every path: the first-step branch, the momentum's dither, the caps
    g = torch.randn(256, 256, device=device)
    for _ in range(3):
        push(a, b, 1.0, g)
        torch.cuda.set_sync_debug_mode("error")
        try:
            opt.step()
        finally:
            torch.cuda.set_sync_debug_mode("default")
    assert b.detach().any()


def _cfg(**kw):
    return TrainConfig(**{**dict(optimizer="hull", hull_rho=1e-3, steps=20), **kw})


def test_config_refuses_what_could_raise_the_step_or_train_on_a_judge():
    assert "hull" in OPTIMIZERS and "hull" not in AUTOMAGIC
    _cfg().validate()
    assert _cfg(max_grad_norm=1.0).clip_norm == 0 and TrainConfig(max_grad_norm=1.0).clip_norm == 1.0
    for bad, match in (
            (dict(hull_rho=0.0), "hull_rho"), (dict(network_type="lokr"), "LoRA pairs"), (dict(dora=True), "LoRA pairs"),
            (dict(likeness_lr_control=True), "likeness_lr_control"), (dict(likeness_curriculum=True), "likeness_lr_control"),
            (dict(loss_watch_healthy_mult=1.1), "loss_watch_healthy_mult"), (dict(reward_steps=10), "reward_steps"),
            (dict(selftrain=True), "reward_steps"), (dict(hull_state_dtype="fp16"), "hull_state_dtype"),
            (dict(id_loss_weight=0.1, id_loss_backend="facenet"), "facenet"), (dict(hull_betas=[0.9]), "hull_betas")):
        with pytest.raises(ValueError, match=match):
            _cfg(**bad).validate()


def test_per_step_multipliers_stay_at_or_below_one():
    assert T.limit_step_mult(_cfg(), 1.1) == 1.0 and T.limit_step_mult(_cfg(), 0.5) == 0.5
    assert T.limit_step_mult(TrainConfig(), 1.1) == 1.1


def test_hull_settings_can_race():
    assert {"hull_rho", "hull_betas", "hull_clip", "hull_damping"} <= R.ALLOWED


def test_the_trainer_builds_hull_from_real_lora_layers():
    linear = torch.nn.Linear(64, 32).float()
    model = torch.nn.ModuleDict({"one": L.LoRALinear(linear, rank=4, alpha=2.0),
                                 "two": L.LoRALinear(torch.nn.Linear(64, 32).float(), rank=4, alpha=4.0)})
    injected = dict(model.items())
    fake = type("F", (), {"race": None})()
    inputs = T.Trainer._hull_inputs(fake, injected)
    assert inputs["scales"] == [0.5, 1.0]
    assert inputs["ref_norms"][0] == pytest.approx(linear.weight.detach().norm().item())
    groups = L.lora_param_groups(injected, 1e-3, 4.0)  # LoRA+ groups collapse
    opt = T.build_optimizer(_cfg(), groups, pairs=L.lora_pairs(injected), **inputs)
    assert isinstance(opt, Hull) and opt.param_groups[0]["lr"] == 1e-3
    sched = T.build_scheduler(_cfg(lr_scheduler="cosine", warmup_steps=0), opt)
    x = torch.randn(3, 64)
    for _ in range(3):
        opt.zero_grad()
        (model["one"](x).square().sum() + model["two"](x).square().sum()).backward()
        opt.step()
        sched.step()
    assert model["one"].lora_up.weight.detach().any() and opt.param_groups[0]["lr"] < 1e-3


def test_only_lora_pairs_are_accepted():
    a, b = factors()
    other = torch.nn.Parameter(torch.zeros(4))
    with pytest.raises(ValueError, match="LoRA pairs"):
        Hull([a, b, other], [(a, b)], [W0], [1.0], lr=1e-3)
    with pytest.raises(ValueError, match="pairs"):
        Hull([a, b], [(a, b)], [W0, W0], [1.0], lr=1e-3)
    empty = torch.nn.Parameter(torch.zeros(0))  # a kind's unused tensor
    Hull([a, b, empty], [(a, b)], [W0], [1.0], lr=1e-3)
