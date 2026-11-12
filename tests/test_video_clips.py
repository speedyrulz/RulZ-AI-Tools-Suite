"""Video clip training (train_frames) for PAN 2 and MiniMax H3, and MiniMax H3's clip samples and probes, without
model weights: H3's multi-frame layout and forward against diffusers' MiniMaxH3Transformer3DModel on the pipeline's
own t2va packed sequence (build_packed_sequence with several latent frames, audio_latent_num_frames); H3's clip
lengths against the pipeline's helpers; the clip VAE paths on a tiny random VAE; the sampler's clip shapes; reading
frames from a real video file; the dataset's clip items, latents, batches and their cache; the config checks; a
training step's loss and gradients on a clip."""
import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import archs  # noqa: E402
from rulztrainer.config import TrainConfig  # noqa: E402
from rulztrainer.h3 import convert as C  # noqa: E402
from rulztrainer.h3 import model as HM  # noqa: E402
from rulztrainer.h3.model import AUDIO_TAG, TEXT_TAG, VIDEO_TAG, MiniMaxH3Transformer, audio_sigma  # noqa: E402

diffusers = pytest.importorskip("diffusers")

TINY_DIFFUSERS = dict(num_attention_heads=2, attention_head_dim=32, hidden_size=48, num_layers=2, num_refiner_layers=1,
                      ffn_dim=64, in_channels=24, audio_in_channels=32, patch_size=(1, 2, 2), text_dim=40,
                      freq_dim=256, time_embed_hidden_dim=56, time_embed_dim=24, rope_freq_dim=4, rope_theta=10000.0)
TINY = C.diffusers_config_to_port(TINY_DIFFUSERS)
H3 = archs.arch_by_name("minimax_h3")
PAN2 = archs.arch_by_name("pan2")


def pair(seed=0):
    torch.manual_seed(seed)
    ref = diffusers.MiniMaxH3Transformer3DModel(**TINY_DIFFUSERS).float().eval()
    with torch.no_grad():
        for p in ref.parameters():
            p.normal_(0, 0.1)
    ours = MiniMaxH3Transformer(**TINY).float().eval()
    missing, unexpected = ours.load_state_dict(C.diffusers_to_port(dict(ref.state_dict())), strict=True)
    assert not missing and not unexpected
    return ours, ref


def ref_clip(ref, x, sigma, enc, audio_rows):
    """diffusers' forward on the pipeline's t2va layout for a clip x [1, 24, T, h, w]: [text | 2 x A audio rows |
    T latent frames, frame-major], A = audio_latent_num_frames of the clip's frame count. Returns the data-ward
    video velocity [1, 24, T, h, w] and the audio rows' [1, 2A, 32]."""
    from diffusers.modular_pipelines.minimax_h3.before_denoise import (MiniMaxH3PrepareLayoutStep,
                                                                       MiniMaxH3SetTimestepsStep,
                                                                       patchify_video_latents)

    _, c, t, h, w = x.shape
    n_text = enc.shape[1]
    n_audio = audio_rows.shape[1] // 2
    pos, tags, vid_idx, aud_idx, txt_idx, n_cv, n_ca = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
        torch.full((n_text,), TEXT_TAG), num_latent_frames=t, latent_height=h, latent_width=w,
        num_audio_latents=n_audio, patch_size=(1, 2, 2), audio_channels=2, audio_tag=AUDIO_TAG, video_tag=VIDEO_TAG)
    t_v = 1.0 - float(sigma)
    t_a = 1.0 - float(audio_sigma(torch.tensor(float(sigma))))
    ts, idx = MiniMaxH3SetTimestepsStep.build_row_timesteps(vid_idx, aud_idx, n_cv, n_ca, n_text, t_v, t_a,
                                                            max(t_v, 0.999), 1.0)
    rows = patchify_video_latents(x, (1, 2, 2)).unsqueeze(0)
    out = ref(hidden_states=rows, audio_hidden_states=audio_rows, encoder_hidden_states=enc, timestep=ts,
              timestep_indices=idx, token_tags=tags, position_ids=pos.float(), video_indices=vid_idx,
              audio_indices=aud_idx, text_indices=txt_idx, return_dict=False)
    frames = HM.unpatchify(out[0].reshape(t, (h // 2) * (w // 2), c * 4), c, h, w)  # [T, C, h, w]
    return frames.transpose(0, 1).unsqueeze(0), out[1]


# ---------------------------------------------------------------- MiniMax H3: lengths, layout, forward


def test_h3_clip_lengths_match_the_pipeline():
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import (align_num_frames, audio_latent_num_frames,
                                                                         video_latent_num_frames)

    for f in (2, 5, 21, 22, 23, 49, 121, 360):
        snapped = HM.snap_frames(f)
        assert snapped == align_num_frames(f, 17, 5)
        assert HM.video_latent_frames(snapped) == video_latent_num_frames(snapped, 17, 5)
        assert HM.pixel_frames(HM.video_latent_frames(snapped)) == snapped
        assert HM.audio_latent_frames(snapped) == audio_latent_num_frames(snapped)
    assert HM.snap_frames(1) == 1 and HM.video_latent_frames(1) == 1 and HM.audio_latent_frames(1) == 2
    assert H3.supports_video and H3.snap_frames(49) == 56 and H3.latent_frames(56) == 17
    # the image-only relatives of H3 do not take clips
    assert not archs.arch_by_name("minimax_h3_image").supports_video
    assert not archs.arch_by_name("p_h3_image").supports_video
    with pytest.raises(ValueError, match="17n"):
        HM.video_latent_frames(20)


def test_h3_clip_positions_are_the_pipelines():
    from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3PrepareLayoutStep

    ours = MiniMaxH3Transformer(**TINY)
    for t, h, w, n_text in ((2, 4, 6, 7), (7, 4, 4, 3), (12, 6, 4, 5)):
        frames = HM.pixel_frames(t)
        n_audio = HM.audio_latent_frames(frames)
        pos = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
            torch.ones(n_text, dtype=torch.long), t, h, w, n_audio, (1, 2, 2), 2, AUDIO_TAG, VIDEO_TAG)[0]
        assert ours.audio_frames_for(t) == n_audio
        got = ours.positions(torch.tensor([n_text]), n_text, h, w, t)[0]
        assert torch.equal(got, pos)  # bit for bit, the non-uniform rotary time included
    # one latent frame is the one-frame layout as before
    still = ours.positions(torch.tensor([4]), 4, 4, 4)
    assert torch.equal(still, ours.positions(torch.tensor([4]), 4, 4, 4, 1))


def test_h3_clip_forward_matches_diffusers_per_sample_with_padding():
    ours, ref = pair(3)
    t = 7  # 22 frames: 37 audio latents per channel
    x = torch.randn(2, 24, t, 4, 6)
    sigma = torch.tensor([0.9, 0.35])
    lens = (8, 5)
    enc = torch.randn(2, max(lens), 40)
    mask = torch.zeros(2, max(lens), dtype=torch.bool)
    for i, n in enumerate(lens):
        mask[i, :n] = True
    n_audio = HM.audio_latent_frames(HM.pixel_frames(t))
    audio = torch.randn(2, 2 * n_audio, 32)
    with torch.no_grad():
        got, got_audio = ours(x, sigma * 1000.0, enc, mask, audio_states=audio, return_audio=True)
        assert got.shape == x.shape and got_audio.shape == audio.shape
        for i, n in enumerate(lens):
            want, want_audio = ref_clip(ref, x[i:i + 1], sigma[i], enc[i:i + 1, :n], audio[i:i + 1])
            assert torch.allclose(got[i:i + 1], -want, atol=3e-5, rtol=1e-4)
            assert torch.allclose(got_audio[i:i + 1], -want_audio, atol=3e-5, rtol=1e-4)
        # a one-latent-frame clip is the still
        x1 = x[:, :, :1]
        a1 = torch.randn(2, 4, 32)
        assert torch.allclose(ours(x1, sigma * 1000.0, enc, mask, audio_states=a1)[:, :, 0],
                              ours(x1[:, :, 0], sigma * 1000.0, enc, mask, audio_states=a1), atol=1e-6)


def test_h3_clip_audio_placeholder_slices_the_long_silence():
    ours, _ = pair(4)
    ours.audio_silence = torch.randn(4, 32)
    long = torch.randn(2, HM.audio_latent_frames(56) + 5, 32)
    ours.audio_silence_long = long
    n = HM.audio_latent_frames(HM.pixel_frames(17))
    rows = ours._clip_silence(n)
    assert rows.shape == (2 * n, 32)
    assert torch.equal(rows[:n], long[0, :n]) and torch.equal(rows[n:], long[1, :n])  # channel-major
    ours.audio_silence_long = None  # without it: each channel's last still frame, repeated
    rows = ours._clip_silence(n)
    still = ours.audio_silence.view(2, 2, 32)
    assert torch.equal(rows[:n], still[0, 1:].expand(n, 32)) and torch.equal(rows[n:], still[1, 1:].expand(n, 32))
    # the clip forward draws its placeholder at the clip's length (no audio_states)
    x, enc = torch.randn(1, 24, 2, 4, 4), torch.randn(1, 5, 40)
    with torch.no_grad():
        assert ours(x, 500.0, enc, None).shape == x.shape


def test_h3_clip_trains_with_checkpointing():
    """A training forward on a clip: bf16, gradient checkpointing, the loss reaches the LoRA-trained layers."""
    ours, _ = pair(5)
    ours = ours.to(torch.bfloat16).train()
    ours.gradient_checkpointing = True
    ours.audio_silence = torch.zeros(4, 32)
    for p in ours.parameters():
        p.requires_grad_(False)
    w = ours.blocks[1].attn.qkv_proj.weight
    w.requires_grad_(True)
    lat, noise = torch.randn(1, 24, 7, 4, 4), torch.randn(1, 24, 7, 4, 4)
    s = torch.tensor([0.6])
    noisy = ((1 - s.view(-1, 1, 1, 1, 1)) * lat + s.view(-1, 1, 1, 1, 1) * noise).to(torch.bfloat16)
    pred = ours(noisy, s * 1000.0, torch.randn(1, 5, 40, dtype=torch.bfloat16), None).float()
    loss = (pred - (noise - lat)).pow(2).mean(dim=(1, 2, 3, 4)).mean()
    loss.backward()
    assert torch.isfinite(loss) and w.grad is not None and w.grad.abs().sum() > 0


def test_h3_clip_sampler_shapes():
    from rulztrainer.h3 import sampler as HS

    ours, _ = pair(6)
    ours.audio_silence = torch.zeros(4, 32)
    enc, mask = torch.randn(1, 5, 40), torch.ones(1, 5, dtype=torch.bool)
    lat = HS.sample(ours.to(torch.bfloat16), enc.to(torch.bfloat16), mask, None, None, width=96, height=64, steps=2,
                    device="cpu", frames=20)
    assert lat.shape == (1, 24, 7, 4, 6)  # 20 -> 22 frames -> 7 latent frames
    still = HS.sample(ours, enc.to(torch.bfloat16), mask, None, None, width=96, height=64, steps=2, device="cpu")
    assert still.shape == (1, 24, 4, 6)


TINY_VAE = dict(block_out_channels=(8, 8, 8, 8, 8, 16), layers_per_block=1, norm_num_groups=4, decoder_num_layers=1,
                decoder_num_attention_heads=2, decoder_attention_head_dim=16)


def test_h3_clip_vae_encode_and_decode():
    from rulztrainer.h3 import loading as HL
    from rulztrainer.h3 import single_file as HSF

    torch.manual_seed(7)
    vae = HSF.vae_class()(**{**HSF.VAE_CONFIG, **TINY_VAE}).float().eval()
    with torch.no_grad():
        for p in vae.parameters():
            p.normal_(0, 0.05)
    clip = torch.rand(1, 3, 22, 64, 96) * 2 - 1
    lat = H3.encode_video(vae, clip)
    assert lat.shape == (1, 24, 7, 4, 6) and torch.isfinite(lat).all()
    # the VAE's own chunked encode of the ImageNet-normalized clip, normalized per channel
    with torch.no_grad():
        pm = torch.tensor(HL.IMAGENET_MEAN).view(1, 3, 1, 1, 1)
        ps = torch.tensor(HL.IMAGENET_STD).view(1, 3, 1, 1, 1)
        z = vae.encode(((clip + 1) / 2 - pm) / ps).latent_dist.mode()
        mean = torch.tensor(HSF.VAE_CONFIG["latents_mean"]).view(1, -1, 1, 1, 1)
        std = torch.tensor(HSF.VAE_CONFIG["latents_std"]).view(1, -1, 1, 1, 1)
    assert torch.allclose(lat, (z - mean) / std, atol=1e-5)
    frames = H3.decode_video(vae, lat)
    assert frames.shape == (22, 3, 64, 96) and frames.min() >= -1 and frames.max() <= 1


# ---------------------------------------------------------------- PAN 2


def test_pan2_clip_encode_normalizes_every_latent_frame():
    class Vae:  # 4k + 1 frames -> k + 1 latent frames of 5.0, as a stand-in for the Wan 2.2 VAE
        device = torch.device("cpu")

        class config:
            latents_mean = [1.0] * 48
            latents_std = [2.0] * 48

        def parameters(self):
            return iter([torch.zeros(1)])

        def encode(self, x):
            t = (x.shape[2] - 1) // 4 + 1
            z = torch.full((x.shape[0], 48, t, x.shape[3] // 16, x.shape[4] // 16), 5.0)

            class Dist:
                def mode(self):
                    return z

            class Out:
                latent_dist = Dist()

            return Out()

    lat = PAN2.encode_video(Vae(), torch.zeros(1, 3, 9, 32, 64))
    assert lat.shape == (1, 48, 3, 2, 4) and torch.allclose(lat, torch.full_like(lat, 2.0))  # (5 - 1) / 2
    assert PAN2.latent_frames(49) == 13 and PAN2.snap_frames(50) == 49


# ---------------------------------------------------------------- reading frames, the dataset, the config


def _write_video(path: Path, n: int, fps: int = 30, size=(64, 48)):
    av = pytest.importorskip("av")
    import numpy as np

    with av.open(str(path), "w") as c:
        st = c.add_stream("mpeg4", rate=fps)
        st.width, st.height, st.pix_fmt = size[0], size[1], "yuv420p"
        for k in range(n):
            img = np.full((size[1], size[0], 3), min(255, 8 * k), dtype=np.uint8)
            for pkt in st.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                c.mux(pkt)
        for pkt in st.encode():
            c.mux(pkt)


def test_frame_picking_follows_ai_toolkit():
    from rulztrainer import clips

    # real time: 30 fps source onto 24 fps, from the start
    assert clips.pick_frames(100, 30.0, 5, 24, "fps") == [0, 1, 2, 4, 5]
    # too short for real time: spread over the whole video instead
    assert clips.pick_frames(4, 30.0, 9, 24, "fps") == [0, 0, 1, 1, 2, 2, 2, 3, 3]
    # spread: the whole video, evenly
    assert clips.pick_frames(101, 30.0, 5, 24, "spread") == [0, 25, 50, 75, 100]
    assert clips.pick_frames(10, 24.0, 1, 24) == [0]


def test_reading_a_video_file(tmp_path):
    from rulztrainer import clips

    path = tmp_path / "walk.mp4"
    _write_video(path, 20)
    w, h, n, fps = clips.video_info(path)
    assert (w, h) == (64, 48) and abs(fps - 30) < 1e-6 and 18 <= n <= 22
    pics = clips.read_frames(path, 9, 24, "fps")
    assert len(pics) == 9 and pics[0].size == (64, 48)
    means = [sum(p.convert("L").getdata()) / (64 * 48) for p in pics]
    want = [8 * k for k in clips.pick_frames(n, fps, 9, 24, "fps")]
    assert all(abs(m - v) < 6 for m, v in zip(means, want))  # each the frame asked for (mpeg4 is lossy)


def test_dataset_clip_items_latents_and_batches(tmp_path, monkeypatch):
    from PIL import Image
    from safetensors.torch import save_file

    from rulztrainer import dataset as D

    root = tmp_path / "set"
    root.mkdir()
    Image.new("RGB", (640, 480), (90, 90, 90)).save(root / "photo.png")
    (root / "photo.txt").write_text("a photo", encoding="utf-8")
    _write_video(root / "walk.mp4", 30)
    (root / "walk.txt").write_text("a woman walks", encoding="utf-8")

    # pictures only: the video is left out
    cfg = TrainConfig(model_arch="pan2", dataset_dir=str(root), resolution=512, min_bucket_side=128)
    assert [it.category for it in D.scan_dataset(cfg)] == ["image"]

    cfg = TrainConfig(model_arch="pan2", dataset_dir=str(root), resolution=512, min_bucket_side=128,
                      train_frames=10, video_resolution=256, flip_augment=True)
    cfg.validate()
    items = D.scan_dataset(cfg)
    clip = next(it for it in items if it.frames > 1)
    still = next(it for it in items if it.frames == 1)
    assert clip.category == "clip" and clip.frames == 9 and clip.latent_frames == 3  # 10 -> 4k + 1 = 9
    assert clip.width % 32 == 0 and abs(clip.width * clip.height - 256 * 256) < 256 * 64
    assert clip.region_path is None and clip.side == 256
    assert clip.tokens == (clip.width // 16) * (clip.height // 16) * 3
    wrapped = json.loads(clip.caption)  # PAN 2: the caption carries the clip's length (8 frames at 24 fps)
    assert wrapped["init_frame_visual_description"] == "a woman walks"
    assert wrapped["events"][0]["time_range_sec"] == f"0 to {8 / 24:g}"
    assert clip.text_path == D.text_cache_path(cfg, clip.caption) and still.caption == "a photo"
    assert D.items_for_side(items, 512) == items  # clips train at every resolution stage
    # another clip length or frame choice is another latent file
    other = D.scan_dataset(TrainConfig(**{**cfg.__dict__, "train_video_mode": "spread"}))
    assert next(it for it in other if it.frames > 1).latent_path != clip.latent_path

    # caching: the clip's frames through the family's clip encoder, mirrored for the flip copy
    seen = []

    def fake_video(vae, frames):
        seen.append(frames.shape)
        return torch.zeros(frames.shape[0], 48, (frames.shape[2] - 1) // 4 + 1, frames.shape[3] // 16,
                           frames.shape[4] // 16)

    monkeypatch.setattr(PAN2, "encode_video", fake_video)
    monkeypatch.setattr(PAN2, "encode_images", lambda vae, x: torch.zeros(x.shape[0], 48, x.shape[2] // 16,
                                                                          x.shape[3] // 16))

    class Vae:
        device = "cpu"

    D.cache_latents(items, cfg, Vae())
    assert seen == [(1, 3, 9, clip.height, clip.width)] * 2
    assert clip.latent_path.exists() and clip.latent_flip_path.exists()

    # batches: clips with clips, latents [B, C, T, h, w] and no region maps
    batcher = D.BucketBatcher(items, 1, 0)
    assert (clip.width, clip.height, 9) in batcher.buckets
    for it in items:
        save_file({"embeds": torch.zeros(3, 8)}, str(it.text_path))
    save_file({"embeds": torch.zeros(1, 8)}, str(D.dropout_text_path(cfg)))
    import random

    lat, emb, mask, regions = D.load_batch([clip], cfg, random.Random(0))
    assert lat.shape == (1, 48, 3, clip.height // 16, clip.width // 16) and regions is None


def test_clip_training_config_checks():
    TrainConfig(model_arch="minimax_h3", train_frames=39, sample_frames=22, probe_frames=22).validate()
    TrainConfig(model_arch="pan2", train_frames=49).validate()
    for arch in ("krea2", "minimax_h3_image", "p_h3_image"):
        with pytest.raises(ValueError, match="video model"):
            TrainConfig(model_arch=arch, train_frames=49).validate()
    with pytest.raises(ValueError, match="1 .a still. or more"):
        TrainConfig(model_arch="pan2", train_frames=0).validate()
    with pytest.raises(ValueError, match="fps or spread"):
        TrainConfig(model_arch="pan2", train_frames=9, train_video_mode="random").validate()
    with pytest.raises(ValueError, match="Isolation"):
        TrainConfig(model_arch="pan2", train_frames=9, isolation=True).validate()


def test_h3_render_hooks_and_the_trainer_frames():
    from rulztrainer.trainer import render_frames, render_negative, render_prompt

    class T:
        cfg = TrainConfig(model_arch="minimax_h3", sample_frames=40, probe_frames=1)
        arch = H3

        def negative_prompt(self, what="still"):
            return "neg"

    t = T()
    assert render_frames(t, "sample") == 56 and render_frames(t, "probe") == 1
    assert render_prompt(t, "a fox", "sample") == "a fox"  # H3 reads prompts verbatim, clip or still
    assert render_negative(t, "sample") == "neg"
