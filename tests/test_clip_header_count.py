"""Reading a clip from a video whose header promises more frames than it holds."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


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


def test_a_header_that_says_more_frames_than_there_are_spreads_over_the_real_ones(tmp_path, monkeypatch):
    from rulztrainer import clips

    path = tmp_path / "walk.mp4"
    _write_video(path, 20)
    w, h, n, fps = clips.video_info(path)
    monkeypatch.setattr(clips, "video_info", lambda p: (w, h, n + 12, fps))  # the duration counted a longer track
    pics = clips.read_frames(path, 5, 24, "spread")
    means = [sum(p.convert("L").getdata()) / (64 * 48) for p in pics]
    want = [8 * k for k in clips.pick_frames(n, fps, 5, 24, "spread")]
    assert all(abs(m - v) < 6 for m, v in zip(means, want))  # spread over the frames there are: no frozen tail
    assert len(set(round(m) for m in means)) == 5
