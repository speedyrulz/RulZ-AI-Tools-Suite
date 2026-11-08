"""Changes after OneTrainer's last year: hidden dataset folders skipped, `push` leaving the cache behind, the hf_xet
download cap, the PyTorch build picked from the driver, the Dockerfile's root, Scaled OFT's learning rates, peak RAM
and memory snapshots, activation offloading and torch.compile with block swap (the last two experimental; their
CUDA paths are only tested where a GPU is visible)."""
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "vast"))

import rulz_remote as remote  # noqa: E402
import torch_index  # noqa: E402
from rulztrainer import lora as L  # noqa: E402
from rulztrainer import profiling  # noqa: E402
from rulztrainer.actoffload import ActivationOffload  # noqa: E402
from rulztrainer.checkpointing import checkpointed  # noqa: E402
from rulztrainer.config import TrainConfig  # noqa: E402
from rulztrainer.dataset import skipped  # noqa: E402

CUDA = torch.cuda.is_available()


# ------------------------------------------------------------------ small fixes
def test_hidden_files_and_folders_never_train(tmp_path):
    root = tmp_path / ".hidden_parent" / "set"  # the dataset's own parents may be hidden
    for rel, skip in [("a.png", False), ("sub/b.png", False), (".thumbnails/a.png", True), ("sub/.@__thumb/b.png", True),
                      ("._a.png", True), (".rulz_cache/x.png", True), ("excluded_by_look/c.png", True)]:
        assert skipped(root / rel, root) is skip, rel
    assert not skipped(root / "a.png")  # without a root only the fixed folders count
    assert skipped(root / "excluded_by_look" / "c.png")


def test_push_leaves_hidden_folders_behind(tmp_path):
    ds = tmp_path / "set"
    (ds / ".rulz_cache" / "latents").mkdir(parents=True)
    (ds / ".rulz_cache" / "latents" / "a.safetensors").write_bytes(b"x" * 10)
    (ds / "a.png").write_bytes(b"png")
    (ds / ".DS_Store").write_bytes(b"x")
    names = {}
    for hidden in (False, True):
        out = tmp_path / f"t{hidden}.tar"
        with tarfile.open(out, "w") as tar:
            tar.add(ds, arcname="set", filter=remote.tar_info_for_instance if hidden else remote.without_hidden)
        with tarfile.open(out) as tar:
            names[hidden] = sorted(tar.getnames())
    assert names[False] == ["set", "set/a.png"]
    assert "set/.rulz_cache/latents/a.safetensors" in names[True] and "set/.DS_Store" in names[True]
    args = remote.parser().parse_args(["push", "datasets/set", "datasets", "--with-cache"])
    assert args.with_cache and not remote.parser().parse_args(["push", "a", "b"]).with_cache


def test_hf_xet_download_cap_on_import_and_a_set_value_wins():
    def after(env):
        code = "import os, rulztrainer; print(os.environ['HF_XET_FIXED_DOWNLOAD_CONCURRENCY'])"
        e = {k: v for k, v in os.environ.items() if k != "HF_XET_FIXED_DOWNLOAD_CONCURRENCY"}
        out = subprocess.run([sys.executable, "-c", code], cwd=REPO, env={**e, **env}, capture_output=True, text=True,
                             check=True)
        return out.stdout.strip()

    assert after({}) == "8"
    assert after({"HF_XET_FIXED_DOWNLOAD_CONCURRENCY": "32"}) == "32"


@pytest.mark.parametrize("smi, default, build, warns", [
    ("| NVIDIA-SMI 581.57   Driver Version: 581.57   CUDA Version: 13.0 |", "cu130", "cu130", False),
    ("CUDA Version: 13.1", "cu128", "cu128", False),  # never newer than the default
    ("CUDA Version: 12.9", "cu130", "cu128", True),
    ("CUDA Version: 12.6", "cu130", "cu126", True),
    ("CUDA Version: 12.4", "cu130", "cu126", True),  # older than any build: the oldest, with a stronger warning
    ("no header here", "cu130", "cu130", False),
    ("", "cu128", "cu128", False),
])
def test_torch_index_follows_the_driver(smi, default, build, warns):
    got, warning = torch_index.pick(torch_index.driver_cuda(smi), default)
    assert got == build and bool(warning) is warns
    assert torch_index.driver_cuda("CUDA Version: 12.8") == (12, 8)


def test_installers_use_the_index_script_and_docker_builds_as_root():
    bat = (REPO / "install.bat").read_text(encoding="utf-8")
    assert "scripts\\torch_index.py" in bat and 'if "%TORCH_INDEX%"=="" set "TORCH_INDEX=' in bat
    assert "scripts/torch_index.py --default cu128" in (REPO / "install.sh").read_text(encoding="utf-8")
    docker = (REPO / "Dockerfile").read_text(encoding="utf-8")
    assert docker.index("USER root") < docker.index("RUN ")


# ------------------------------------------------------------------ Scaled OFT
class _Tiny(torch.nn.Module):
    def __init__(self, outs=(64, 48)):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.ModuleDict({"q": torch.nn.Linear(64, outs[0]),
                                                                "o": torch.nn.Linear(64, outs[1])})])


def test_scaled_oft_lr_per_block_size():
    assert L.oft_lr_scale(32) == 1.0
    assert L.oft_lr_scale(16) == pytest.approx((31 / 15) ** 0.5) and L.oft_lr_scale(64) == pytest.approx((31 / 63) ** 0.5)
    inj = L.inject_lora(_Tiny(), 4, 4, ["q", "o"], kind="oft", block=32)
    groups = L.lora_param_groups(inj, 1e-4, oft_scaled=True)
    by = {g["name"]: g for g in groups}
    assert set(by) == {"oft_b32", "oft_b24"}  # 48 outputs: the largest divisor <= 32 is 24
    assert by["oft_b32"]["lr"] == 1e-4 and by["oft_b24"]["lr"] == pytest.approx(1e-4 * (31 / 23) ** 0.5)
    assert [g["name"] for g in L.lora_param_groups(inj, 1e-4)] == ["lora"]  # off: one group, as before
    lora = L.inject_lora(_Tiny(), 4, 4, ["q", "o"])  # a LoRA has no OFT blocks to scale
    assert [g["name"] for g in L.lora_param_groups(lora, 1e-4, 4.0, oft_scaled=True)] == ["lora_down", "lora_up"]
    TrainConfig(network_type="oft", oft_scaled_lr=True).validate()
    from rulztrainer import race as R

    assert "oft_scaled_lr" in R.RACE_KEYS["LoRA rank & layers"]


def _rotation_move(block: int, scaled: bool) -> float:
    """RMS per output channel of how far one AdamW step moves a layer's OFT rotation from the identity."""
    torch.manual_seed(0)
    inj = L.inject_lora(torch.nn.ModuleDict({"layers": torch.nn.ModuleList([torch.nn.ModuleDict(
        {"q": torch.nn.Linear(8, 256)})])}), 4, 4, ["q"], kind="oft", block=block)
    (m,) = inj.values()
    opt = torch.optim.AdamW(L.lora_param_groups(inj, 1e-3, oft_scaled=scaled), weight_decay=0.0)
    m.oft_blocks.grad = torch.randn_like(m.oft_blocks)
    opt.step()
    r = L.oft_rotation(m.oft_blocks.detach())
    eye = torch.eye(block).expand_as(r)
    return (r - eye).pow(2).sum(-1).sqrt().mean().item()  # each output channel's change


def test_scaled_oft_moves_every_block_size_about_as_far():
    plain = _rotation_move(64, False) / _rotation_move(16, False)
    scaled = _rotation_move(64, True) / _rotation_move(16, True)
    assert 1.7 < plain < 2.3  # sqrt(63 / 15) = 2.05: bigger blocks step further at one lr
    assert 0.9 < scaled < 1.1


# ------------------------------------------------------------------ peak RAM and memory snapshots
def test_peak_ram_and_snapshot_without_cuda(tmp_path, monkeypatch):
    assert (profiling.private_ram_bytes() or 0) > 0
    ram = profiling.PeakRam(every=0.01).start()
    held = bytearray(50_000_000)
    ram.sample()
    ram.stop()
    assert ram.peak >= 50_000_000 and ram.peak_gb() > 0
    del held
    monkeypatch.setenv("RULZ_MEMORY_SNAPSHOT_STEP", "7")
    assert profiling.requested_step("RULZ_MEMORY_SNAPSHOT_STEP") == 7
    if not CUDA:
        snap = profiling.MemorySnapshot(tmp_path, 7)
        snap.start()
        assert snap.stop() is None and not list(tmp_path.iterdir())  # nothing to record without CUDA


# ------------------------------------------------------------------ activation offloading
class _Block(torch.nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.a, self.b = torch.nn.Linear(dim, dim), torch.nn.Linear(dim, dim)

    def forward(self, x, rope):
        return x + self.b(torch.nn.functional.gelu(self.a(x * rope)))


class _Stack(torch.nn.Module):
    def __init__(self, n=4):
        super().__init__()
        self.layers = torch.nn.ModuleList([_Block() for _ in range(n)])

    def forward(self, x, rope):
        for blk in self.layers:
            x = checkpointed(self, blk, x, rope)
        return x


def _grads(model, x, rope):
    model.zero_grad(set_to_none=True)
    model(x, rope).square().mean().backward()
    return [p.grad.clone() for p in model.parameters()]


def test_checkpointed_without_offload_is_plain_checkpointing():
    torch.manual_seed(0)
    model = _Stack()
    x = torch.randn(2, 64, 256, requires_grad=True)
    rope = torch.randn(64, 256)
    with_ckpt = _grads(model, x, rope)
    model.activation_offload = ActivationOffload()  # enabled only with CUDA: on the CPU it must change nothing
    again = _grads(model, x, rope)
    for a, b in zip(with_ckpt, again):
        torch.testing.assert_close(a, b)
    off = ActivationOffload()
    t = torch.randn(1024, 1024, requires_grad=True)
    assert off._pack(t) is t and off._unpack(t) is t  # CPU tensors stay where they are
    with pytest.raises(ValueError, match="gradient_checkpointing"):
        TrainConfig(activation_offload=True, gradient_checkpointing=False).validate()


@pytest.mark.skipif(not CUDA, reason="activation offloading moves CUDA tensors")
def test_activation_offload_gives_the_same_gradients_on_cuda():
    torch.manual_seed(0)
    model = _Stack(6).cuda()
    x = torch.randn(2, 512, 256, device="cuda", requires_grad=True)
    rope = torch.randn(512, 256, device="cuda")  # needs no gradient: stays on the GPU
    plain = _grads(model, x, rope)
    off = ActivationOffload(min_bytes=1 << 16)
    model.activation_offload = off
    moved = _grads(model, x, rope)
    torch.cuda.synchronize()
    for a, b in zip(plain, moved):
        torch.testing.assert_close(a, b)
    assert off.take_bytes() == 6 * 2 * 512 * 256 * 4  # every block's input (each needs a gradient); rope stays


# ------------------------------------------------------------------ compile with block swap
def test_swapped_blocks_compile_their_forward_only(monkeypatch):
    pytest.importorskip("triton")
    from rulztrainer import compiling
    from rulztrainer.trainer import Trainer

    monkeypatch.setattr(compiling, "prepare", lambda *a, **k: None)  # no global dynamo settings in the test process
    model = _Stack(3)
    model.layers[0]._rz_swap_index = 0  # what BlockSwap marks its blocks with
    forward0 = model.layers[0].forward
    fake = SimpleNamespace(arch=SimpleNamespace(compile_blocks=lambda m: list(m.layers)))
    assert Trainer.compile(fake, model)
    swapped, plain = model.layers[0], model.layers[1]
    assert swapped.forward is not forward0 and getattr(swapped, "_compiled_call_impl", None) is None
    assert getattr(plain, "_compiled_call_impl", None) is not None and "forward" not in vars(plain)
    TrainConfig(block_swap=4, block_swap_compile=True).validate()
