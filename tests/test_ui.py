"""Smoke tests of the main window, the editor and Auto crop, offscreen with a stand-in SAM
(QT_QPA_PLATFORM=offscreen python -m unittest discover tests). Settings go to a temporary file."""
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from rulz import ai_mask, config

_tmp = tempfile.TemporaryDirectory()
_settings = QSettings(str(Path(_tmp.name) / "settings.ini"), QSettings.IniFormat)
config.settings = lambda: _settings  # never the real registry

from rulz.ui import auto_crop, theme  # noqa: E402
from rulz.ui.auto_crop import AutoCropDialog, Found  # noqa: E402
from rulz.ui.main_window import ITEM_ROLE, MainWindow  # noqa: E402

app = QApplication.instance() or QApplication([])
theme.apply(app)


def _wait(cond, timeout=20.0):
    end = time.time() + timeout
    while not cond():
        if time.time() > end:
            raise TimeoutError
        app.processEvents()
        time.sleep(0.01)


def _pictures(folder: Path, n: int) -> list[Path]:
    rng = np.random.default_rng(0)
    paths = []
    for i in range(n):
        img = cv2.resize(rng.integers(0, 256, (12, 16, 3), dtype=np.uint8), (800, 600), interpolation=cv2.INTER_CUBIC)
        p = folder / f"p{i}.jpg"
        cv2.imwrite(str(p), img)
        paths.append(p)
    return paths


class FoundTests(unittest.TestCase):
    def test_packed_mask_round_trips(self):
        m = np.random.default_rng(0).random((1008, 671)) > 0.7
        f = Found.of(m, [0.8], (3000, 2000))
        np.testing.assert_array_equal(f.mask, m)
        self.assertLess(f.bits.nbytes, m.size // 7)


class WindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        cls.paths = _pictures(Path(cls.dir.name), 6)

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def setUp(self):
        self.win = MainWindow()
        self.win.show()
        self.assertEqual(self.win.add_paths([str(p) for p in self.paths]), len(self.paths))
        _wait(lambda: all(it.preview is not None for it in self.win.items))

    def tearDown(self):
        self.win.close()
        self.win.deleteLater()
        app.processEvents()

    def test_grid_bookkeeping(self):
        w = self.win
        self.assertLessEqual(w._pool.maxThreadCount(), 4)
        self.assertEqual(set(w._cards), {id(it) for it in w.items})
        w._on_global("color", 40)
        _wait(lambda: not w._render_queue)
        self.assertTrue(all(it.preview_key == ("color", 40, None) for it in w.items))
        w.grid.item(1).setSelected(True)
        w.grid.item(3).setSelected(True)
        gone = {id(w.items[1]), id(w.items[3])}
        w._remove_selected()
        self.assertEqual(len(w.items), 4)
        self.assertEqual(set(w._cards), {id(it) for it in w.items})
        self.assertFalse(gone & set(w._cards))
        self.assertEqual([w.grid.item(r).data(ITEM_ROLE) for r in range(w.grid.count())], w.items)
        w._on_global("color", 0)

    def test_repeats_are_skipped_and_removed_pictures_can_come_back(self):
        w = self.win
        self.assertEqual(w.add_paths([str(self.paths[2]), str(self.paths[2].parent)]), 0)
        w.grid.item(2).setSelected(True)
        w._remove_selected()
        self.assertEqual(w.add_paths([str(self.paths[2]), str(self.paths[0])]), 1)
        self.assertEqual(len(w.items), len(self.paths))
        self.assertLessEqual(max(w.items[0].thumb.shape[:2]), config.THUMB_MAX_SIDE)

    def test_opening_a_picture_leaves_nothing_selected(self):
        """The click that opens a picture selects it; left selected, Auto crop would take only that one."""
        w = self.win
        w.grid.item(2).setSelected(True)  # what that click does
        w._open(2)
        w.editor.navigate.emit(1)  # arrow keys move the selection along
        w._show_library()
        self.assertEqual(w.grid.selectedItems(), [])
        for r in (1, 3):  # a selection made on purpose stays
            w.grid.item(r).setSelected(True)
        w._show_library()
        self.assertEqual(len(w.grid.selectedItems()), 2)

    def test_next_picture_loads_while_the_last_one_is_still_processing(self):
        from rulz.ui import workers

        saved = workers.process_cached
        gate = threading.Event()

        def slow(*args, **kw):
            gate.wait(20)
            return saved(*args, **kw)

        workers.process_cached = slow
        try:
            w = self.win
            ed = w.editor
            w._open(0)
            _wait(lambda: ed._full is not None and ed._running)  # picture 0's processing is now held up
            w._open(1)
            _wait(lambda: ed._full is not None, timeout=5)  # yet picture 1 is loaded
            self.assertFalse(gate.is_set())
            gate.set()
            _wait(lambda: not ed._running and ed._shown_key == ed._wanted())
            self.assertIs(ed.item, w.items[1])
        finally:
            gate.set()
            workers.process_cached = saved

    def test_editor_reuses_the_slow_stage(self):
        w = self.win
        w._open(0)
        ed = w.editor
        _wait(lambda: ed._full is not None and not ed._running)
        ed._on_slider("jpeg", 40)
        _wait(lambda: ed._shown_key == ed._wanted() and not ed._running)
        stage = ed._cache[1]
        ed._on_slider("detail", 60)
        _wait(lambda: ed._shown_key == ed._wanted() and not ed._running)
        self.assertIs(ed._cache[1], stage)
        w._show_library()
        self.assertIsNone(ed._cache)


class AutoCropTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        cls.paths = _pictures(Path(cls.dir.name), auto_crop.DISPLAY_KEPT + 3)

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def setUp(self):
        self.calls = {"release": 0}
        self.saved = ai_mask.find, ai_mask.release

        def find(img, prompt, *_args):
            m = np.zeros((300, 400), bool)
            m[100:200, 150:250] = True
            return m, [0.9]

        def release():
            self.calls["release"] += 1

        ai_mask.find, ai_mask.release = find, release
        from rulz.ui.workers import ImageItem

        self.items = [ImageItem(p) for p in self.paths]

    def tearDown(self):
        ai_mask.find, ai_mask.release = self.saved

    def test_preview_crop_box_memory_and_release(self):
        dlg = AutoCropDialog(self.items, {"color": 0, "jpeg": 0, "detail": 0}, "balanced")
        dlg.show()
        dlg.prompt.setText("thing")
        dlg._preview()
        _wait(lambda: dlg._finder is None)
        self.assertEqual(len(dlg._found), len(self.items))
        self.assertLessEqual(len(dlg._pix), auto_crop.DISPLAY_KEPT)
        dlg.strip.setCurrentRow(0)  # its preview was let go: loaded again
        _wait(lambda: id(self.items[0]) in dlg._pix)
        self.assertIsNotNone(dlg.preview._box)
        self.assertLessEqual(max(dlg.preview._overlay.width(), dlg.preview._overlay.height()),
                             auto_crop.OVERLAY_MAX_SIDE)
        # a match whose mask came back empty must not break the preview
        dlg._found[id(self.items[0])] = Found.of(np.zeros((300, 400), bool), [0.9], (600, 800))
        dlg._show_current()
        self.assertIsNone(dlg.preview._box)
        dlg.reject()
        self.assertEqual(self.calls["release"], 1)
        self.assertEqual(len(dlg._pix), 0)

    def test_starting_note_shows_until_the_first_result(self):
        dlg = AutoCropDialog(self.items[:2], {"color": 0, "jpeg": 0, "detail": 0}, "balanced")
        dlg.show()
        dlg.prompt.setText("thing")
        dlg._preview()
        self.assertIn("Starting SAM", dlg.info.text())  # SAM isn't loaded in these tests
        _wait(lambda: dlg._finder is None)
        self.assertTrue(dlg.info.text().startswith("Checked 2 of 2"), dlg.info.text())
        dlg.reject()

    def test_crop_all_writes_crops(self):
        out = Path(self.dir.name) / "out"
        out.mkdir()
        from rulz.crop import CropSettings
        from rulz.ui.auto_crop import CropAll, FindSettings

        jobs = [(it, out / it.path.name, {"color": 0, "jpeg": 50, "detail": 30}, None) for it in self.items[:2]]
        done = {}
        w = CropAll(jobs, FindSettings("thing"), CropSettings(aspect_limit="off", padding_percent=0), "balanced")
        w.finished_all.connect(lambda s, e, err: done.update(saved=s, empty=e, errors=err))
        w.start()
        _wait(lambda: "saved" in done)
        w.wait()
        self.assertEqual(done, {"saved": 2, "empty": [], "errors": []})
        img = cv2.imread(str(out / self.items[0].path.name))
        self.assertEqual(img.shape[:2], (200, 200))  # the mask's box, scaled from 400 x 300 to 800 x 600


class SaveTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.dir.name)

    def tearDown(self):
        self.dir.cleanup()

    def test_output_never_lands_on_any_source(self):
        from rulz.ui.workers import ImageItem, output_paths

        x, y = self.tmp / "X", self.tmp / "Y"
        x.mkdir()
        y.mkdir()
        srcs = [x / "a.jpg", x / "b.jpg", y / "a.jpg", y / "A_prepared.jpg"]
        for p in srcs:
            p.write_bytes(b"")
        dests = output_paths([ImageItem(p) for p in srcs], x)
        self.assertFalse({d.resolve() for d in dests} & {p.resolve() for p in srcs})
        self.assertEqual(len({d.name.lower() for d in dests}), len(dests))
        self.assertEqual([d.name for d in dests], ["a_prepared.jpg", "b_prepared.jpg", "a_prepared_2.jpg",
                                                   "A_prepared_prepared.jpg"])

    def test_unchanged_pictures_are_copied_not_re_encoded(self):
        from rulz.ui.workers import SaveAll

        src = _pictures(self.tmp, 1)[0]
        empty = self.tmp / "empty.jpg"
        empty.write_bytes(b"")
        out = self.tmp / "out"
        out.mkdir()
        zero = {"color": 0, "jpeg": 0, "detail": 0}
        empty2 = self.tmp / "empty2.jpg"
        empty2.write_bytes(b"")
        jobs = [(empty2, out / "empty2.jpg", zero, None), (src, out / "same.jpg", zero, None),
                (src, out / "cleaned.jpg", {**zero, "color": 40}, None), (empty, out / "empty.jpg", zero, None)]
        done = {}
        saver = SaveAll(jobs, "balanced")
        saver.finished_all.connect(lambda s, err: done.update(saved=s, errors=err))
        saver.run()  # on this thread
        self.assertEqual(done["saved"], 2)
        # reported, not copied, in list order although two pictures are saved at a time
        self.assertEqual([n for n, _ in done["errors"]], ["empty2.jpg", "empty.jpg"])
        self.assertEqual(SaveAll(jobs, "best").workers(), 1)
        self.assertFalse((out / "empty.jpg").exists())
        self.assertEqual((out / "same.jpg").read_bytes(), src.read_bytes())
        self.assertNotEqual((out / "cleaned.jpg").read_bytes(), src.read_bytes())


class SetupDialogTests(unittest.TestCase):
    def test_esc_and_close_button_stop_the_download(self):
        from rulz.ui.ai_setup import AISetupDialog

        class Download:
            cancelled = False

            def isRunning(self):
                return True

            def cancel(self):
                self.cancelled = True

            def wait(self):
                pass

        dlg = AISetupDialog()
        dlg._dl = dl = Download()
        dlg.show()
        dlg.reject()  # Esc and the window's close button land here, like Cancel
        self.assertTrue(dl.cancelled)
        dlg._next()  # a step finishing just then starts nothing
        self.assertIs(dlg._dl, dl)
        self.assertIsNone(dlg._proc)


if __name__ == "__main__":
    unittest.main()
