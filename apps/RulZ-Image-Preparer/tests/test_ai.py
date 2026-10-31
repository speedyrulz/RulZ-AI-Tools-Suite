"""Tests for unloading the AI models and SAM's empty-mask filter, with stand-in models
(python -m unittest discover tests). Skipped without PyTorch."""
import importlib.util
import threading
import time
import unittest

import numpy as np

from rulz import ai_jpeg, ai_mask

HAVE_TORCH = importlib.util.find_spec("torch") is not None


class _BlockingNet:
    """Stands in for FBCNN: returns its input, after waiting for `go` the first time."""

    def __init__(self):
        self.started, self.go = threading.Event(), threading.Event()

    def __call__(self, t, qf):
        self.started.set()
        self.go.wait(10)
        return t


@unittest.skipUnless(HAVE_TORCH, "needs PyTorch")
class JpegReleaseTests(unittest.TestCase):
    def setUp(self):
        import torch

        self.torch = torch
        self.saved = ai_jpeg._model

    def tearDown(self):
        ai_jpeg._model, ai_jpeg._release_pending = self.saved, False

    def test_release_when_idle(self):
        ai_jpeg._model = (object(), self.torch.device("cpu"))
        ai_jpeg.release()
        self.assertIsNone(ai_jpeg._model)

    def test_release_during_a_run_waits_for_it(self):
        net = _BlockingNet()
        ai_jpeg._model = (net, self.torch.device("cpu"))
        rgb = np.random.default_rng(0).random((40, 56, 3)).astype(np.float32)
        out = {}
        worker = threading.Thread(target=lambda: out.setdefault("img", ai_jpeg.remove_artifacts(rgb, 50)))
        worker.start()
        self.assertTrue(net.started.wait(10))
        t0 = time.perf_counter()
        ai_jpeg.release()  # must not block on the running job
        self.assertLess(time.perf_counter() - t0, 0.5)
        self.assertIsNotNone(ai_jpeg._model)  # still in use
        net.go.set()
        worker.join(10)
        np.testing.assert_allclose(out["img"], rgb, atol=1e-6)  # the run finished normally
        self.assertIsNone(ai_jpeg._model)  # and dropped the model afterwards

    def test_out_of_memory_retries_with_smaller_tiles(self):
        torch = self.torch
        calls = []

        class Net:
            def __call__(self, t, qf):
                calls.append(tuple(t.shape[2:]))
                if len(calls) == 1:
                    raise torch.cuda.OutOfMemoryError("pretend")
                return t

        ai_jpeg._model = (Net(), torch.device("cpu"))
        rgb = np.random.default_rng(1).random((900, 500, 3)).astype(np.float32)
        out = ai_jpeg.remove_artifacts(rgb, 50)
        np.testing.assert_allclose(out, rgb, atol=1e-6)
        self.assertGreater(max(calls[0]), ai_jpeg.TILE // 2)  # first try: full-size tiles
        self.assertLessEqual(max(max(c) for c in calls[1:]), ai_jpeg.TILE // 2)

    def test_real_network_tiles_match_one_pass(self):
        """With random weights on the CPU: tiling only changes pixels near tile seams, and in place ReLU
        changes nothing."""
        torch = self.torch
        from torch import nn

        torch.manual_seed(0)
        _, FBCNN = ai_jpeg._build()
        net = FBCNN().eval()
        for m in net.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)
        rgb = np.random.default_rng(2).random((96, 128, 3)).astype(np.float32)
        ai_jpeg._model = (net, torch.device("cpu"))
        tiled = ai_jpeg.remove_artifacts(rgb, 60)
        with torch.inference_mode():
            t = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))[None]
            qf = torch.tensor([[0.9 * 0.6]])
            whole = net(t, qf)[0].clamp(0, 1).numpy().transpose(1, 2, 0)
            for m in net.modules():
                if isinstance(m, nn.ReLU):
                    m.inplace = False
            not_inplace = net(t, qf)[0].clamp(0, 1).numpy().transpose(1, 2, 0)
        np.testing.assert_array_equal(whole, not_inplace)
        np.testing.assert_allclose(tiled, whole, atol=1e-5)  # one tile covers this picture


class _FakeInputs(dict):
    def to(self, *_):
        return self


class _FakeProc:
    def __init__(self, scores, masks):
        self.scores, self.masks = scores, masks

    def __call__(self, **_):
        return _FakeInputs()

    def post_process_instance_segmentation(self, out, threshold, mask_threshold, target_sizes):
        import torch

        return [{"scores": torch.tensor(self.scores), "masks": torch.tensor(self.masks)}]


@unittest.skipUnless(HAVE_TORCH and ai_mask.libs_installed(), "needs PyTorch, Transformers and Pillow")
class MaskTests(unittest.TestCase):
    def setUp(self):
        self.saved = ai_mask._model

    def tearDown(self):
        ai_mask._model, ai_mask._release_pending = self.saved, False

    def _use(self, scores, masks):
        import torch

        ai_mask._model = (lambda **_: None, _FakeProc(scores, masks), torch.device("cpu"), torch.float32)

    def test_empty_masks_are_not_matches(self):
        m = np.zeros((3, 20, 30), np.int64)
        m[0, 2:5, 3:9] = 1  # match 0: a small area
        m[2, 10:12, 10:20] = 1  # match 1 is empty, match 2 a strip
        self._use([0.5, 0.875, 0.75], m)
        mask, scores = ai_mask.find(np.zeros((20, 30, 3), np.uint8), "x")
        self.assertEqual(mask.dtype, bool)
        self.assertEqual(scores, [0.5, 0.75])
        self.assertEqual(int(mask.sum()), 18 + 20)
        mask, scores = ai_mask.find(np.zeros((20, 30, 3), np.uint8), "x", best_only=True)
        self.assertEqual(scores, [0.75])  # the best non-empty one, not the empty 0.875
        self.assertEqual(int(mask.sum()), 20)

    def test_only_empty_masks(self):
        self._use([0.9], np.zeros((1, 20, 30), np.int64))
        mask, scores = ai_mask.find(np.zeros((20, 30, 3), np.uint8), "x")
        self.assertEqual(scores, [])
        self.assertFalse(mask.any())

    def test_pictures_are_masked_at_most_at_sam_size(self):
        seen = {}

        class Proc(_FakeProc):
            def __call__(self, images=None, **_):
                seen["size"] = images.size
                return _FakeInputs()

        import torch

        ai_mask._model = (lambda **_: None, Proc([], np.zeros((0, 1, 1), np.int64)), torch.device("cpu"),
                          torch.float32)
        mask, _ = ai_mask.find(np.zeros((3000, 2000, 3), np.uint8), "x")
        self.assertEqual(max(seen["size"]), ai_mask.MASK_MAX_SIDE)
        self.assertEqual(mask.shape, (1008, 672))

    def test_release_during_a_search(self):
        import torch

        started, go = threading.Event(), threading.Event()

        def net(**_):
            started.set()
            go.wait(10)

        ai_mask._model = (net, _FakeProc([], np.zeros((0, 8, 8), np.int64)), torch.device("cpu"), torch.float32)
        worker = threading.Thread(target=lambda: ai_mask.find(np.zeros((8, 8, 3), np.uint8), "x"))
        worker.start()
        self.assertTrue(started.wait(10))
        ai_mask.release()
        self.assertIsNotNone(ai_mask._model)
        go.set()
        worker.join(10)
        self.assertIsNone(ai_mask._model)


if __name__ == "__main__":
    unittest.main()
