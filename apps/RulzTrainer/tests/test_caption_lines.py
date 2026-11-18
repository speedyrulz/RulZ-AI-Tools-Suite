"""Several captions per picture (caption_lines): every non-empty line of a .txt caption file is its own caption with the
trigger word, each cached on its own; one line is drawn each time the picture trains. Off: one caption, as ever."""
import dataclasses
import random
import sys
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import dataset as D  # noqa: E402
from rulztrainer.config import TrainConfig  # noqa: E402


def _pic(root: Path, text: str | None, name="a", ext=".txt") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64)).save(root / f"{name}.png")
    if text is not None:
        (root / f"{name}{ext}").write_text(text, encoding="utf-8")
    return root / f"{name}.png"


def _cfg(root, **kw):
    return TrainConfig(dataset_dir=str(root), min_bucket_side=64, trigger_word="ohwx", **kw)


def test_off_by_default_a_multi_line_file_is_one_caption(tmp_path):
    p = _pic(tmp_path, "one\ntwo\n")
    assert TrainConfig().caption_lines is False
    assert len(D.read_captions(p, _cfg(tmp_path))) == 1
    it = D.scan_dataset(_cfg(tmp_path))[0]
    assert it.alt_captions == [] and it.caption == D.read_caption(p, _cfg(tmp_path))


def test_each_line_is_a_caption_with_the_trigger(tmp_path):
    p = _pic(tmp_path, "a woman smiling\n\n  walking in a park  \na woman smiling\n")
    caps = D.read_captions(p, _cfg(tmp_path, caption_lines=True))
    assert len(caps) == 2 and all("ohwx" in c for c in caps)
    assert "smiling" in caps[0] and "park" in caps[1]  # blank lines and a repeated line left out
    it = D.scan_dataset(_cfg(tmp_path, caption_lines=True))[0]
    assert it.caption == caps[0] and it.alt_captions == caps[1:] and len(it.alt_text_paths) == 1
    assert it.alt_text_paths[0] != it.text_path
    assert len(it.all_text_paths()) == 2


def test_json_and_empty_and_missing_captions_stay_one(tmp_path):
    cfg = _cfg(tmp_path, caption_lines=True)
    assert len(D.read_captions(_pic(tmp_path, "", "empty"), cfg)) == 1
    assert len(D.read_captions(_pic(tmp_path, None, "bare"), cfg)) == 1
    cfg = _cfg(tmp_path, caption_lines=True, model_arch="ideogram4")
    p = _pic(tmp_path, '{"short_description": "x"}', "j", ".json")
    assert len(D.read_captions(p, cfg)) == 1


def test_every_line_is_a_text_job(tmp_path):
    _pic(tmp_path, "one\ntwo\nthree\n")
    cfg = _cfg(tmp_path, caption_lines=True)
    items = D.scan_dataset(cfg)
    jobs, total = D.text_jobs(items, cfg)
    assert set(jobs.values()) >= {items[0].caption, *items[0].alt_captions}
    assert total == 3 + 1  # three lines and the dropout prompt


def test_one_line_is_drawn_each_time_from_the_runs_generator(tmp_path):
    cfg = _cfg(tmp_path, caption_dropout=0.0)
    paths = {}
    for name, value in (("lat", None), ("l0", 1.0), ("l1", 2.0), ("l2", 3.0)):
        paths[name] = tmp_path / f"{name}.safetensors"
        save_file({"latent": torch.zeros(128, 4, 4)} if value is None else
                  {"embeds": torch.full((3, 4096), value).to(torch.bfloat16)}, str(paths[name]))
    it = D.Item(image=tmp_path / "x.png", caption="c", width=64, height=64, latent_path=paths["lat"],
                latent_flip_path=None, text_path=paths["l0"], alt_captions=["d", "e"],
                alt_text_paths=[paths["l1"], paths["l2"]])

    def draws(seed):
        rng = random.Random(seed)
        return [float(D.load_batch([it], cfg, rng)[1].float().mean()) for _ in range(40)]

    a = draws(5)
    assert a == draws(5) and set(a) == {1.0, 2.0, 3.0}  # reproducible, and every line comes up
    # a dropped caption is the dropout prompt, never a line
    D.dropout_text_path(cfg).parent.mkdir(parents=True, exist_ok=True)
    save_file({"embeds": torch.full((3, 4096), 9.0).to(torch.bfloat16)}, str(D.dropout_text_path(cfg)))
    drop = dataclasses.replace(cfg, caption_dropout=1.0)
    assert float(D.load_batch([it], drop, random.Random(0))[1].float().mean()) == 9.0


def test_an_item_without_lines_draws_nothing_extra_from_the_generator(tmp_path):
    class Rng:
        def random(self):
            raise AssertionError("no draw expected")

        def choice(self, seq):
            raise AssertionError("no draw expected")

    cfg = _cfg(tmp_path, caption_dropout=0.0)
    lat, txt = tmp_path / "l.safetensors", tmp_path / "t.safetensors"
    save_file({"latent": torch.zeros(128, 4, 4)}, str(lat))
    save_file({"embeds": torch.zeros(3, 4096).to(torch.bfloat16)}, str(txt))
    it = D.Item(image=tmp_path / "x.png", caption="c", width=64, height=64, latent_path=lat, latent_flip_path=None,
                text_path=txt)
    D.load_batch([it], cfg, Rng())
