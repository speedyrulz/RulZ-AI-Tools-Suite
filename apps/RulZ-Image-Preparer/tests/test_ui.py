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


def tearDownModule():
    """Free every window while the QApplication is still there. A dialog's lambda slots keep it alive where
    Python's garbage collector can't see, so it was freed only as Python exited, after the application, and
    the process crashed then (exit code 0xC0000005 although every test had passed). The app itself deletes
    its dialogs after use (deleteLater)."""
    import shiboken6
    from PySide6.QtCore import QCoreApplication, QEvent

    for w in QApplication.topLevelWidgets():
        if shiboken6.isValid(w):
            shiboken6.delete(w)
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


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


class CompareViewTests(unittest.TestCase):
    def _event(self, kind, pos, buttons=None):
        from PySide6.QtCore import QEvent, QPointF, Qt
        from PySide6.QtGui import QMouseEvent

        kinds = {"press": QEvent.MouseButtonPress, "move": QEvent.MouseMove, "release": QEvent.MouseButtonRelease}
        held = Qt.NoButton if kind == "release" else Qt.LeftButton
        return QMouseEvent(kinds[kind], QPointF(pos), QPointF(pos), Qt.LeftButton, held, Qt.NoModifier)

    def test_a_stroke_keeps_its_place(self):
        """The full picture replacing the thumbnail mid-stroke put the first points in the top-left corner,
        and in Side by side a stroke dragged on into the other pane jumped across the picture."""
        from PySide6.QtCore import QPointF
        from PySide6.QtGui import QPixmap

        from rulz.ui.compare_view import PAINT, SIDE, CompareView

        v = CompareView()
        v.resize(820, 400)
        v.set_mode(SIDE)
        v.set_images(QPixmap(200, 150), None)  # the thumbnail
        v.set_tool(PAINT)
        strokes = []
        v.stroked.connect(strokes.append)
        left = v._panes()[0]
        t = v._target(left)

        def at(fx, fy):
            return QPointF(t.left() + fx * t.width(), t.top() + fy * t.height())

        v.mousePressEvent(self._event("press", at(0.5, 0.5)))
        v.mouseMoveEvent(self._event("move", at(0.6, 0.5)))
        v.set_images(QPixmap(4000, 3000), None)  # the full picture arrives: same shape, same place on screen
        v.mouseMoveEvent(self._event("move", at(0.7, 0.5)))
        past = QPointF(left.right() + 30, at(0.7, 0.5).y())  # on into the right-hand pane
        v.mouseMoveEvent(self._event("move", past))
        v.mouseReleaseEvent(self._event("release", past))
        pts = strokes[0].points
        self.assertAlmostEqual(pts[0][0], 0.5, delta=0.01)
        self.assertEqual([round(x, 2) for x, _ in pts], [0.5, 0.6, 0.7, 1.0])  # 1.0: clamped at the edge
        self.assertTrue(all(abs(y - 0.5) < 0.01 for _, y in pts))

    def test_overlay_is_redrawn_only_when_the_strokes_change(self):
        from dataclasses import replace

        from PySide6.QtGui import QPixmap

        from rulz.shadows import ShadowEdit, Stroke
        from rulz.ui.compare_view import CompareView

        v = CompareView()
        v.set_images(QPixmap(400, 300), None)
        edit = ShadowEdit((Stroke(((0.3, 0.3), (0.6, 0.6)), 0.05),))
        v.set_shadow(edit)
        first = v._overlay
        v.set_shadow(replace(edit, strength=30, blend=80))  # Shadow lift / Edge blend steps
        self.assertIs(v._overlay, first)
        v.set_shadow(replace(edit, strokes=edit.strokes + (Stroke(((0.1, 0.1),), 0.02),)))
        self.assertIsNot(v._overlay, first)
        v.set_images(QPixmap(200, 150), None)  # another picture size: redrawn for it
        self.assertEqual((v._overlay.width(), v._overlay.height()), (200, 150))


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

    def test_cards_on_screen_render_first_every_time(self):
        """While a global slider is dragged, cards already rendered once went to the back of the queue."""
        from PySide6.QtCore import QRect

        w = self.win
        w._render_timer.stop()
        shown = {id(w.items[4]), id(w.items[5])}
        inside, outside = QRect(0, 0, 10, 10), QRect(-500, -500, 10, 10)
        w.grid.visualItemRect = lambda li: inside if id(li.data(ITEM_ROLE)) in shown else outside
        try:
            w._queue_render(w.items)  # the first slider step
            self.assertEqual(list(w._render_queue)[:2], [w.items[4], w.items[5]])
            for _ in range(2):  # the two on screen are rendered...
                w._queued.discard(id(w._render_queue.popleft()))
            w._queue_render(w.items)  # ...and the next step puts them first again, not last
            self.assertEqual(list(w._render_queue), [w.items[i] for i in (4, 5, 0, 1, 2, 3)])
            self.assertEqual(w._queued, {id(it) for it in w.items})
            w._render_queue.clear()
            w._queued.clear()
            w._on_global("detail", 30)  # thumbnails don't show it: nothing to render
            self.assertFalse(w._render_queue)
        finally:
            del w.grid.visualItemRect
            w._on_global("detail", 0)

    def test_arrow_keys_nudge_a_slider_that_has_the_keyboard(self):
        w = self.win
        w._open(0)
        ed = w.editor
        moved = []
        ed.navigate.connect(moved.append)
        slider = ed.rows["color"].slider
        w.activateWindow()
        slider.setFocus()
        app.processEvents()
        if QApplication.focusWidget() is not slider:
            self.skipTest("this platform gives no keyboard focus offscreen")
        ed._arrow(1)
        self.assertEqual((slider.value(), moved), (1, []))
        self.assertEqual(ed.item.custom.get("color"), 1)
        ed.view.setFocus()
        app.processEvents()
        ed._arrow(1)
        self.assertEqual(moved, [1])
        w._show_library()

    def test_removed_pictures_are_not_loaded(self):
        from rulz.ui.workers import ImageItem, ThumbLoad

        got = []

        class Emitter:
            class done:
                @staticmethod
                def emit(*a):
                    got.append(a)

        it = ImageItem(self.paths[0])
        it.removed = True
        ThumbLoad(it, Emitter).run()
        self.assertEqual(got, [])

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

    def test_changing_the_words_stops_the_search(self):
        gate, calls = threading.Event(), []

        def slow(img, prompt, *_args):
            calls.append(prompt)
            gate.wait(10)
            m = np.zeros((300, 400), bool)
            m[100:200, 150:250] = True
            return m, [0.9]

        ai_mask.find = slow
        dlg = AutoCropDialog(self.items, {"color": 0, "jpeg": 0, "detail": 0}, "balanced")
        dlg.show()
        dlg.prompt.setText("thing")
        dlg._preview()
        _wait(lambda: calls)
        self.assertFalse(dlg.preview_btn.isEnabled())
        dlg.prompt.setText("other thing")  # the search for "thing" stops after the picture it's on
        gate.set()
        _wait(lambda: dlg._finder is None, timeout=5)
        self.assertEqual(calls, ["thing"])
        self.assertTrue(dlg.preview_btn.isEnabled())
        dlg.prompt.setText("thing")  # back to the old words: what was found for them is still there
        self.assertEqual(len(dlg._found), 1)
        dlg.reject()

    def test_closing_does_not_wait_for_a_search(self):
        """SAM can take half a minute to load; closing the screen meanwhile hung the window."""
        gate, calls = threading.Event(), []

        def slow(img, prompt, *_args):
            calls.append(prompt)
            gate.wait(10)
            return np.zeros((300, 400), bool), []

        ai_mask.find = slow
        dlg = AutoCropDialog(self.items, {"color": 0, "jpeg": 0, "detail": 0}, "balanced")
        dlg.show()
        dlg.prompt.setText("thing")
        dlg._preview()
        _wait(lambda: calls)
        finder = dlg._finder
        t0 = time.perf_counter()
        dlg.reject()
        self.assertLess(time.perf_counter() - t0, 2)
        self.assertTrue(finder.isRunning())
        self.assertEqual(self.calls["release"], 1)
        gate.set()
        auto_crop.finish_searches()  # what closing the app does
        self.assertFalse(finder.isRunning())
        self.assertEqual(calls, ["thing"])  # stopped after the picture it was on
        self.assertEqual(self.calls["release"], 2)  # and let SAM go once it ended
        self.assertEqual(dlg._found, {})

    def test_a_search_left_running_spares_the_next_screens_sam(self):
        """Close Auto crop mid-search and open it again: the old search ending mustn't drop the SAM the new
        screen has loaded."""
        gate, calls = threading.Event(), []

        def slow(img, prompt, *_args):
            calls.append(prompt)
            gate.wait(10)
            return np.zeros((300, 400), bool), []

        ai_mask.find = slow
        first = AutoCropDialog(self.items, {"color": 0, "jpeg": 0, "detail": 0}, "balanced")
        first.prompt.setText("thing")
        first._preview()
        _wait(lambda: calls)
        finder = first._finder
        first.reject()  # 1 release: the closing screen's own
        second = AutoCropDialog(self.items, {"color": 0, "jpeg": 0, "detail": 0}, "balanced")
        gate.set()
        _wait(lambda: not finder.isRunning())
        self.assertEqual(self.calls["release"], 1)  # not again while the second screen is open
        second.reject()
        self.assertEqual(self.calls["release"], 2)

    def test_folders_are_searched_with_subfolders(self):
        from rulz.ui.main_window import _images_in

        root = Path(self.dir.name) / "scan"
        (root / "sub" / "deeper").mkdir(parents=True)
        for rel in ("a.jpg", "sub/b.PNG", "sub/deeper/c.webp", "notes.txt", "sub/d.tif.bak"):
            (root / rel).write_bytes(b"")
        got = sorted(p.relative_to(root).as_posix() for p in _images_in(root))
        self.assertEqual(got, ["a.jpg", "sub/b.PNG", "sub/deeper/c.webp"])
        try:
            os.symlink(root / "gone.jpg", root / "broken.jpg")  # a link to nothing isn't a picture
        except OSError:
            return  # needs Developer Mode or admin rights on Windows
        self.assertNotIn("broken.jpg", [p.name for p in _images_in(root)])

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
        # Auto crop of only Y/a.jpg into X: X/a.jpg is in the list too, just not selected
        crop = output_paths([ImageItem(srcs[2])], x, srcs)
        self.assertEqual([d.name for d in crop], ["a_prepared.jpg"])
        self.assertEqual([d.name for d in output_paths([ImageItem(srcs[2])], x)], ["a.jpg"])  # nothing else listed

    def test_links_count_as_the_file_they_point_to(self):
        """Folders are resolved once, not each file; a file that is a link must still be taken for its target."""
        from rulz.ui.workers import ImageItem, output_paths

        x, y, z = self.tmp / "X", self.tmp / "Y", self.tmp / "Z"
        for d in (x, y, z):
            d.mkdir()
        target = y / "photo.jpg"
        target.write_bytes(b"")
        try:
            os.symlink(target, x / "photo.jpg")
        except OSError:  # Windows needs Developer Mode or admin rights for this (the CI runner has them)
            self.skipTest("this account can't make symbolic links")
        # the list holds a link to Y/photo.jpg: saving into Y must not write over what it points to
        self.assertEqual([d.name for d in output_paths([ImageItem(x / "photo.jpg")], y)], ["photo_prepared.jpg"])
        # an existing name in the output folder that links to a listed picture isn't written through either
        src = z / "b.jpg"
        src.write_bytes(b"")
        os.symlink(src, y / "b.jpg")
        self.assertEqual([d.name for d in output_paths([ImageItem(src)], y)], ["b_prepared.jpg"])

    def test_auto_crop_never_writes_over_an_unselected_picture(self):
        x, y = self.tmp / "X", self.tmp / "Y"
        x.mkdir()
        y.mkdir()
        listed = [x / "a.jpg", y / "a.jpg"]
        for p in listed:
            p.write_bytes(b"")
        from rulz.ui.workers import ImageItem

        items = [ImageItem(p) for p in listed]
        got = {}

        class Signal:
            def connect(self, _fn):
                pass

        class FakeCropAll:  # records what would be written instead of writing it
            progress = finished_all = Signal()

            def __init__(self, jobs, *_args):
                got["dests"] = [d for _, d, _, _ in jobs]

            def start(self):
                pass

            def cancel(self):
                pass

            def isRunning(self):
                return False

        saved = auto_crop.CropAll, auto_crop.QFileDialog.getExistingDirectory, auto_crop.QMessageBox.question
        auto_crop.CropAll = FakeCropAll
        auto_crop.QFileDialog.getExistingDirectory = staticmethod(lambda *a, **k: str(x))
        auto_crop.QMessageBox.question = staticmethod(lambda *a, **k: auto_crop.QMessageBox.No)
        try:
            dlg = AutoCropDialog(items[1:], {"color": 0, "jpeg": 0, "detail": 0}, "balanced", library=items)
            dlg.prompt.setText("thing")
            dlg._crop()  # only Y/a.jpg is selected; the crop goes into X, where X/a.jpg is
            self.assertEqual(got.get("dests"), [x / "a_prepared.jpg"])
            dlg.reject()
        finally:
            auto_crop.CropAll, auto_crop.QFileDialog.getExistingDirectory, auto_crop.QMessageBox.question = saved

    def test_unchanged_pictures_are_copied_not_re_encoded(self):
        from rulz.ui import workers
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
        # Best quality's GPU model takes one picture at a time, but only pictures with JPEG removal use it
        self.assertEqual(SaveAll(jobs, "best").workers(), workers.SAVE_WORKERS)
        self.assertEqual(SaveAll(jobs + [(src, out / "j.jpg", {**zero, "jpeg": 30}, None)], "best").workers(), 1)
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
