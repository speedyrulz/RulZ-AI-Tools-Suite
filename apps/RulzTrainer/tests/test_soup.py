"""Seed soup: the average of the files' weight changes (never of their factors), exact or at a rank, and the greedy
soup the probes pick."""
import contextlib
import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

from rulztrainer import soup as S
from rulztrainer.lora import save_lora

OUT, IN = 24, 20
NAMES = ("layers.0.attn.q", "layers.0.attn.k")


def stub(kind, parts, alpha=1.0, rank=1):
    return SimpleNamespace(kind=kind, alpha=float(alpha), rank=rank, factor=4, trainable=True,
                           adapter_params=lambda parts=parts: parts)


def lora_file(path, deltas: dict, rank=4, alpha=2.0, fmt="comfy", seed=0):
    """A LoRA whose layers change the weights by exactly up @ down x alpha / rank; returns those changes."""
    g = torch.Generator().manual_seed(seed)
    stubs, out = {}, {}
    for name in deltas:
        down, up = torch.randn(rank, IN, generator=g), torch.randn(OUT, rank, generator=g)
        stubs[name] = stub("lora", {"lora_down": down, "lora_up": up}, alpha, rank)
    save_lora(stubs, path, dtype=torch.float32, fmt=fmt)
    for name, st in stubs.items():
        p = st.adapter_params()
        out[name] = p["lora_up"] @ p["lora_down"] * (alpha / rank)
    return out


def soup_delta(path) -> dict:
    return {n: S.delta_of(e) for n, e in S.read_ingredient(path).items()}


def test_an_exact_soup_is_the_average_of_the_changes(tmp_path):
    a = lora_file(tmp_path / "a.safetensors", dict.fromkeys(NAMES), rank=4, alpha=2.0, seed=1)
    b = lora_file(tmp_path / "b.safetensors", dict.fromkeys(NAMES[:1]), rank=8, alpha=8.0, seed=2, fmt="peft")
    rep = S.make_soup([tmp_path / "a.safetensors", tmp_path / "b.safetensors"], tmp_path / "s.safetensors", rank=0,
                      weights=[3, 1], strength=1.2)
    got = soup_delta(tmp_path / "s.safetensors")  # saved in bf16: equal to its precision
    assert torch.allclose(got[NAMES[0]], 1.2 * (0.75 * a[NAMES[0]] + 0.25 * b[NAMES[0]]), rtol=0.02, atol=0.02)
    # b does not change k: it counts 0; k has a lower rank than q, so the PEFT file keeps its alpha per module
    assert torch.allclose(got[NAMES[1]], 1.2 * 0.75 * a[NAMES[1]], rtol=0.02, atol=0.02)
    assert rep["energy"] == pytest.approx(1.0) and rep["weights"] == [0.75, 0.25]
    assert any(k.endswith(".lora_A.weight") for k in load_file(rep["out"]))  # an input was PEFT: so is the soup
    # averaging the factors instead would be wrong: the product of averages is not the average of products
    fa, fb = S.read_ingredient(tmp_path / "a.safetensors")[NAMES[0]], S.read_ingredient(tmp_path / "b.safetensors")[NAMES[0]]
    assert fa[1].shape != fb[1].shape  # (different ranks: not even possible here)


def test_a_rank_keeps_what_the_files_share(tmp_path):
    torch.manual_seed(0)
    shared = torch.randn(OUT, 2) @ torch.randn(2, IN)
    for i in range(3):  # three "seeds": the same rank-2 change plus each one's own noise
        noise = torch.randn(OUT, 2) @ torch.randn(2, IN) * 0.3
        d = shared + noise
        u, s, vh = torch.linalg.svd(d)
        parts = {"lora_down": (s[:4, None] * vh[:4]).contiguous(), "lora_up": u[:, :4].contiguous()}
        save_lora({NAMES[0]: stub("lora", parts, 4, 4)}, tmp_path / f"{i}.safetensors", dtype=torch.float32)
    rep = S.make_soup([tmp_path / f"{i}.safetensors" for i in range(3)], tmp_path / "s.safetensors", rank=2)
    got = soup_delta(tmp_path / "s.safetensors")[NAMES[0]]
    each = soup_delta(tmp_path / "0.safetensors")[NAMES[0]]
    assert float((got - shared).norm()) < float((each - shared).norm())  # closer to the shared change than one run
    assert 0.5 < rep["agreement"] < 1.0 and rep["norm_ratio"] < 1.0 and 0.9 < rep["energy"] <= 1.0
    assert "3 ingredients" in S.describe(rep)


def test_peft_files_keep_their_alpha_per_module(tmp_path):
    """A PEFT file whose layers have different alphas keeps them in alpha_pattern; the sampling-LoRA reader used
    the file-wide lora_alpha for every layer before."""
    from rulztrainer.lora import parse_comfy_lora, read_peft_metadata

    g = torch.Generator().manual_seed(0)
    stubs = {NAMES[0]: stub("lora", {"lora_down": torch.randn(8, IN, generator=g),
                                     "lora_up": torch.randn(OUT, 8, generator=g)}, alpha=8, rank=8),
             NAMES[1]: stub("lora", {"lora_down": torch.randn(4, IN, generator=g),
                                     "lora_up": torch.randn(OUT, 4, generator=g)}, alpha=1, rank=4)}
    save_lora(stubs, tmp_path / "p.safetensors", dtype=torch.float32, fmt="peft")
    meta = read_peft_metadata(tmp_path / "p.safetensors")
    got = parse_comfy_lora(load_file(str(tmp_path / "p.safetensors")), meta.get("lora_alpha"),
                           meta.get("alpha_pattern"))
    assert got[NAMES[0]][3] == pytest.approx(1.0) and got[NAMES[1]][3] == pytest.approx(0.25)


def test_lokr_needs_a_rank_and_oft_is_refused(tmp_path):
    g = torch.Generator().manual_seed(0)
    lokr = stub("lokr", {"lokr_w1": torch.randn(4, 4, generator=g), "lokr_w2": torch.randn(6, 5, generator=g)})
    save_lora({NAMES[0]: lokr}, tmp_path / "k.safetensors", dtype=torch.float32)
    lora_file(tmp_path / "a.safetensors", dict.fromkeys(NAMES[:1]))
    files = [tmp_path / "k.safetensors", tmp_path / "a.safetensors"]
    with pytest.raises(ValueError, match="rank"):
        S.make_soup(files, tmp_path / "s.safetensors", rank=0)
    rep = S.make_soup(files, tmp_path / "s.safetensors", rank=8)
    assert rep["layers"] == 1
    k = torch.kron(lokr.adapter_params()["lokr_w1"], lokr.adapter_params()["lokr_w2"])
    assert S.delta_of(S.read_ingredient(tmp_path / "k.safetensors")[NAMES[0]]).allclose(k)
    save_file({"diffusion_model.x.oft_blocks": torch.zeros(2, 2, 2)}, str(tmp_path / "o.safetensors"))
    with pytest.raises(ValueError, match="OFT"):
        S.read_ingredient(tmp_path / "o.safetensors")
    with pytest.raises(ValueError, match="2 or more"):
        S.make_soup(files[:1], tmp_path / "s.safetensors")
    with pytest.raises(ValueError, match="weights"):
        S.normalize_weights(2, [1, -1])


def test_the_greedy_soup_keeps_what_helps(tmp_path):
    """Two runs near a target and one far off: the probes (here: closeness to the target) keep the first two."""
    from rulztrainer import archs
    from rulztrainer import trainer as T
    from rulztrainer.config import TrainConfig

    torch.manual_seed(1)
    target = torch.randn(OUT, 3) @ torch.randn(3, IN)
    files = []
    for i, d in enumerate((target + 0.4 * torch.randn(OUT, IN), target + 0.4 * torch.randn(OUT, IN),
                           -target)):
        u, s, vh = torch.linalg.svd(d)
        parts = {"lora_down": (s[:, None] * vh[:len(s)]).contiguous(), "lora_up": u[:, :len(s)].contiguous()}
        files.append(str(tmp_path / f"run{i}.safetensors"))
        save_lora({NAMES[0]: stub("lora", parts, len(s), len(s))}, files[-1], dtype=torch.float32)
    cfg = TrainConfig(job_name="job", soup_files=files, soup_rank=0, soup_strengths="1.0, 1.3", face_weight=1.0,
                      body_weight=0.0, probe_seeds=1)
    tr = T.Trainer.__new__(T.Trainer)
    tr.cfg, tr.arch, tr.job_dir, tr.devices = cfg, archs.arch_by_name("moss_image"), tmp_path, ["cpu"]
    tr.status = SimpleNamespace(update=lambda **k: None, data={"saved": []})
    tr.controller = SimpleNamespace(normalize=False, norm=lambda kind, v: v)
    tr._free_for_rendering = lambda: None
    tr._sampling_slots = lambda model, step: contextlib.nullcontext([])
    tr._probe_with = lambda slots, step, tag="", n_seeds=None: (0.0, None, [], {"face": [0.0], "body": []})

    def probe(model, path, tag):  # the "likeness": how close the soup's change is to the target
        d = soup_delta(path)[NAMES[0]]
        return 100.0 - 10.0 * float((d - target).norm() / target.norm()), None, {}

    tr._soup_probe = probe
    tr._soup_stage(None)
    rep = json.loads((tmp_path / "soup" / "soup.json").read_text())
    assert rep["soup"][:2] == ["run0.safetensors", "run1.safetensors"] or set(rep["soup"]) == {"run0.safetensors",
                                                                                              "run1.safetensors"}
    assert "run2.safetensors" not in rep["soup"] and rep["strength"] == 1.0
    assert rep["score"] > max(rep["alone"]["run0.safetensors"], rep["alone"]["run1.safetensors"])
    assert rep["uniform"] < rep["score"]  # the plain average of all three, with the bad run, is worse
    final = soup_delta(tmp_path / "job_soup.safetensors")[NAMES[0]]
    assert float((final - target).norm()) < 0.4 * float(target.norm())
    assert not (tmp_path / "soup" / "candidate.safetensors").exists()
