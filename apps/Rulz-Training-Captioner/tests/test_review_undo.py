"""Undo and redo of caption changes, the "Needs review" marks, and edits made to many captions."""
import os
import time

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QAbstractItemDelegate, QApplication

from captioner import edits, model as model_mod, settings_dialog
from captioner.backend import CUT_OFF
from captioner.bulk_dialog import BulkEditDialog
from captioner.config import DEFAULT_SETTINGS
from captioner.main_window import FILTERS
from captioner.model import ERROR, FilterProxy, ImageModel, ReviewRole
from test_window import _image

FINE = "A woman in a red coat stands on a wet street at night."
CUT = "A woman stands in a field. She is holding a"


def _model(tmp_path, captions):
    """A model of images 0.png, 1.png… with these captions (None: no caption file)."""
    paths = []
    for i, c in enumerate(captions):
        paths.append(_image(tmp_path / f"{i}.png"))
        if c is not None:
            (tmp_path / f"{i}.txt").write_text(c, encoding="utf-8")
    m = ImageModel()
    m.add_paths(paths)
    return m


def _file(tmp_path, i):
    p = tmp_path / f"{i}.txt"
    return p.read_text(encoding="utf-8") if p.exists() else None


# ------------------------------------------------------------------ undo / redo
def test_undo_and_redo_of_one_edit(tmp_path):
    m = _model(tmp_path, ["old"])
    m.set_caption(0, "new")
    assert m.undo() == ("edit", 1, 0)
    assert m.items[0]["caption"] == "old" and _file(tmp_path, 0) == "old"
    assert m.redo() == ("edit", 1, 0)
    assert m.items[0]["caption"] == "new" and _file(tmp_path, 0) == "new"
    assert m.redo() is None and m.undo() == ("edit", 1, 0) and m.undo() is None


def test_undo_removes_a_caption_file_that_wasnt_there_before(tmp_path):
    m = _model(tmp_path, [None])
    m.set_caption(0, "a cat")
    assert _file(tmp_path, 0) == "a cat" and m.captioned == 1
    m.undo()
    assert _file(tmp_path, 0) is None  # not an empty file, which a trainer may count as a caption
    assert m.items[0]["caption"] == "" and not m.items[0]["has_file"] and m.captioned == 0
    m.redo()
    assert _file(tmp_path, 0) == "a cat" and m.items[0]["has_file"] and m.captioned == 1


def test_changes_of_one_group_are_one_step(tmp_path):
    m = _model(tmp_path, ["a", "b", "c"])
    g = m.history.group("find and replace")
    for row in (0, 1):
        m.set_caption(row, "x", group=g)
    m.set_caption(2, "typed")  # its own step
    assert [s["label"] for s in m.history.undo] == ["find and replace", "edit"]
    assert m.undo() == ("edit", 1, 0)
    assert m.undo() == ("find and replace", 2, 0)
    assert [it["caption"] for it in m.items] == ["a", "b", "c"]
    assert m.redo() == ("find and replace", 2, 0) and [it["caption"] for it in m.items] == ["x", "x", "c"]
    # a job goes on writing after an edit made meanwhile: later changes are a step of their own
    job = m.history.group("captioning")
    m.set_caption(0, "one", group=job)
    m.set_caption(2, "by hand")
    m.set_caption(1, "two", group=job)
    assert [s["label"] for s in m.history.undo][-3:] == ["captioning", "edit", "captioning"]


def test_a_new_change_ends_redo(tmp_path):
    m = _model(tmp_path, ["a"])
    m.set_caption(0, "b")
    m.undo()
    m.set_caption(0, "c")
    assert m.redo() is None and m.undo() == ("edit", 1, 0) and m.items[0]["caption"] == "a"


def test_a_caption_saved_unchanged_is_not_a_step(tmp_path):
    m = _model(tmp_path, ["same", None])
    m.set_caption(0, "same", force=True)  # the model wrote the caption it already had
    m.set_caption(1, "")                  # nothing typed into an empty box
    assert m.history.undo == []


def test_undo_leaves_captions_changed_since_alone(tmp_path):
    m = _model(tmp_path, ["a", "b", "c"])
    g = m.history.group("find and replace")
    for row in range(3):
        m.set_caption(row, f"new {row}", group=g)
    (tmp_path / "1.txt").write_text("written by another program", encoding="utf-8")
    m.remove_rows([2])  # off the list
    assert m.undo() == ("find and replace", 1, 2)
    assert _file(tmp_path, 0) == "a" and _file(tmp_path, 1) == "written by another program"
    assert _file(tmp_path, 2) == "new 2"
    # redo only does again what undo took back
    assert m.redo() == ("find and replace", 1, 0) and _file(tmp_path, 0) == "new 0"


def test_undo_with_nothing_it_can_put_back_is_dropped(tmp_path):
    m = _model(tmp_path, ["a"])
    m.set_caption(0, "b")
    (tmp_path / "0.txt").write_text("elsewhere", encoding="utf-8")
    assert m.undo() == ("edit", 0, 1)
    assert m.history.undo == [] and m.history.redo == []


def test_undo_matches_captions_however_their_lines_end(tmp_path):
    m = _model(tmp_path, ["old\r\n"])  # as loaded: with the file's line break
    m.set_caption(0, "new")
    assert m.undo() == ("edit", 1, 0) and _file(tmp_path, 0) == "old"
    assert m.redo() == ("edit", 1, 0) and m.undo() == ("edit", 1, 0)  # "old" now, "old\r\n" in the step


def test_only_the_newest_steps_are_kept_and_clear_forgets_them(tmp_path):
    m = _model(tmp_path, ["0"])
    for i in range(1, model_mod.UNDO_STEPS + 6):
        m.set_caption(0, str(i))
    assert len(m.history.undo) == model_mod.UNDO_STEPS
    m.clear()
    assert m.history.undo == [] and m.undo() is None


def test_history_changed_is_not_sent_for_every_caption_of_a_step(tmp_path):
    m = _model(tmp_path, ["a", "b", "c", "d"])
    sent = []
    m.history_changed.connect(lambda: sent.append(1))
    g = m.history.group("find and replace")
    for row in range(4):
        m.set_caption(row, "x", group=g)
    assert len(sent) == 1  # 5,000 captions would repaint the Edit menu 5,000 times


# ------------------------------------------------------------------ caption check
def _issues(m, row):
    return [c for c, _ in m.items[row].get("issues", ())]


def test_recheck_marks_captions_and_counts_them(tmp_path):
    m = _model(tmp_path, [FINE, CUT, None, "Sure, here is a caption: a cat."])
    m.recheck()
    assert [_issues(m, r) for r in range(4)] == [[], ["cut"], [], ["preamble"]]
    assert m.review == 2
    assert [m.data(m.index(r), ReviewRole) for r in (0, 1, 3)] == ["", "cut off", "chat text"]
    tip = m.data(m.index(1), Qt.ToolTipRole)
    assert tip.startswith(m.items[1]["path"]) and "Needs review" in tip and "middle of a sentence" in tip
    assert m.data(m.index(0), Qt.ToolTipRole) == m.items[0]["path"]
    m.set_status(m.items[0]["path"], ERROR)  # a failed image is one to review too
    m.recheck()
    assert m.review == 3


def test_fixing_a_caption_clears_its_mark(tmp_path):
    m = _model(tmp_path, [CUT])
    m.recheck()
    assert m.review == 1
    m.set_caption(0, CUT + " basket.")
    m.recheck()
    assert _issues(m, 0) == [] and m.review == 0


def test_marked_as_checked_until_the_caption_changes(tmp_path):
    m = _model(tmp_path, [CUT, CUT + " book"])
    m.recheck()
    m.mark_checked([0])
    assert [_issues(m, r) for r in range(2)] == [[], ["cut"]] and m.review == 1
    m.set_caption(0, CUT + " hat")
    m.recheck()
    assert _issues(m, 0) == ["cut"]


def test_a_caption_the_model_cut_off_is_marked_until_it_is_edited(tmp_path):
    m = _model(tmp_path, ["1girl, solo, long hair, red"])  # nothing about the text gives it away
    m.recheck()
    assert _issues(m, 0) == []
    m.mark_cut_off(m.items[0]["path"])
    m.recheck()
    assert _issues(m, 0) == ["cut"] and "token limit" in m.data(m.index(0), Qt.ToolTipRole)
    m.set_caption(0, "1girl, solo, long hair, red dress")
    m.recheck()
    assert _issues(m, 0) == []


def test_review_options(tmp_path):
    m = _model(tmp_path, ["ohwx person, a man on a beach", CUT, "A man stands. He waves., film grain"])
    m.set_review_options(words=["ohwx"])
    m.recheck()  # the options change looks at every caption again in slices; this does it all now
    assert [_issues(m, r) for r in range(3)] == [["word"], ["cut"], ["cut"]]
    m.set_review_options(words=["ohwx"], skip=["word"], suffix=", film grain")
    m.recheck()
    assert [_issues(m, r) for r in range(3)] == [[], ["cut"], []]
    m.set_review_options(skip=["cut"])
    m.mark_cut_off(m.items[1]["path"])
    m.recheck()
    assert m.review == 0  # the switched-off check stays off for captions the model cut off too


def test_the_check_after_changes_runs_in_slices(tmp_path, monkeypatch):
    m = _model(tmp_path, [CUT] * 5)
    monkeypatch.setattr(model_mod, "LOOK_SLICE", -1.0)  # every pass is out of time after one caption
    monkeypatch.setattr(model_mod, "LOOK_GAP_MS", 0)
    m.recheck(full=False)
    assert m.review == 0 and m._review_timer.isActive()  # one caption looked at, nothing marked yet
    deadline = time.time() + 10
    while m._review_timer.isActive() and time.time() < deadline:
        QApplication.processEvents()
    assert m.review == 5 and all(_issues(m, r) == ["cut", "duplicate"] for r in range(5))


def test_adding_images_schedules_a_check(tmp_path):
    m = _model(tmp_path, [CUT])
    assert m._review_timer.isActive()
    deadline = time.time() + 10
    while m._review_timer.isActive() and time.time() < deadline:
        QApplication.processEvents()
    assert m.review == 1


def test_review_and_failed_filters_follow_the_marks(tmp_path):
    m = _model(tmp_path, [FINE, CUT, FINE + " Again."])
    proxy = FilterProxy()
    proxy.setSourceModel(m)
    m.recheck()
    proxy.set_filter(mode="review")
    assert proxy.rowCount() == 1 and proxy.index(0, 0).data(Qt.DisplayRole) == "1.png"
    m.set_caption(1, CUT + " basket.")  # fixed: leaves the list without the filter being set again
    m.recheck()
    assert proxy.rowCount() == 0
    m.set_caption(0, CUT)
    m.recheck()
    assert proxy.rowCount() == 1
    proxy.set_filter(mode="failed")
    assert proxy.rowCount() == 0
    m.set_statuses([2], ERROR)
    assert proxy.rowCount() == 1 and proxy.index(0, 0).data(Qt.DisplayRole) == "2.png"
    proxy.set_filter(mode="review")
    assert proxy.rowCount() == 2  # the cut-off one and the failed one


# ------------------------------------------------------------------ the window
def _add(win, tmp_path, captions):
    paths = []
    for i, c in enumerate(captions):
        paths.append(_image(tmp_path / f"{i}.png"))
        if c is not None:
            (tmp_path / f"{i}.txt").write_text(c, encoding="utf-8")
    win.model.add_paths(paths)
    return paths


def test_a_jobs_captions_are_undone_together(win, tmp_path):
    paths = _add(win, tmp_path, ["old", None])
    win.job_group, win.job_saved = win.model.history.group("captioning"), set()
    win.snapshots = {p: win.model.items[i]["caption"] for i, p in enumerate(paths)}
    for p in paths:
        win._on_captioned(p, "from the model")
    win._on_finished()
    assert win.a_undo.isEnabled() and win.a_undo.text() == "Undo captioning (2)"
    win.undo()
    assert _file(tmp_path, 0) == "old" and _file(tmp_path, 1) is None
    assert "Undid captioning: 2 caption(s)" in win.statusBar().currentMessage()
    assert not win.a_undo.isEnabled() and win.a_redo.text() == "Redo captioning (2)"
    win.redo()
    assert _file(tmp_path, 0) == _file(tmp_path, 1) == "from the model"
    win.redo()
    assert "Nothing to redo" in win.statusBar().currentMessage()


def test_cut_off_warning_marks_the_saved_caption_only(win, tmp_path):
    kept, saved = _add(win, tmp_path, ["mine", "old tags"])
    win.job_group, win.job_saved = win.model.history.group("captioning"), set()
    win.snapshots = {kept: "", saved: "old tags"}  # `kept` was edited after it was queued
    for p in (kept, saved):
        win._on_captioned(p, "1girl, solo, long")
        win.engine.warning.emit(p, CUT_OFF)
    win.model.recheck()
    assert _issues(win.model, 0) == [] and win.model.items[0]["caption"] == "mine"
    assert _issues(win.model, 1) == ["cut"]
    assert "1 to review" in win.stats.text()


def test_clear_captions_is_one_undo_step(win, boxes, tmp_path):
    _add(win, tmp_path, ["a", "b"])
    win.grid.selectAll()
    win._clear_selected_captions()
    assert _file(tmp_path, 0) == _file(tmp_path, 1) == ""
    assert win.a_undo.text() == "Undo clear captions (2)"
    win.undo()
    assert _file(tmp_path, 0) == "a" and _file(tmp_path, 1) == "b"


def test_needs_review_filter_and_mark_as_checked(win, tmp_path):
    _add(win, tmp_path, [FINE, CUT, CUT + " book"])
    win.filter_box.setCurrentIndex([mode for mode, _ in FILTERS].index("review"))
    assert win.proxy.rowCount() == 2 and "2 to review" in win.stats.text() and "showing 2" in win.stats.text()
    win.grid.setCurrentIndex(win.proxy.index(0, 0))
    win.mark_checked()
    assert win.proxy.rowCount() == 1 and "1 to review" in win.stats.text()
    assert "marked as checked" in win.statusBar().currentMessage()


def _dialog(win, action=edits.REPLACE):
    return BulkEditDialog(win._captions, win._apply_changes, len(win._selected_rows()),
                          win.proxy.rowCount(), action, win)


def test_find_and_replace_dialog_counts_then_changes(win, tmp_path):
    _add(win, tmp_path, ["a cat and a cat", "a dog", "category", None])
    d = _dialog(win)
    assert not d.r_selected.isEnabled() and d.r_shown.isChecked()
    d.count()
    assert not d.apply_btn.isEnabled() and d.note.text() == ""  # nothing typed yet: no complaint
    d.find_box.setText("cat")
    d.text_box.setText("fox")
    d.count()
    assert d.note.text() == "3 match(es) in 2 of 4 caption(s)." and d.apply_btn.isEnabled()
    d.words.setChecked(True)
    d.count()
    assert d.note.text() == "2 match(es) in 1 of 4 caption(s)."
    d.run()
    assert [_file(tmp_path, i) for i in range(4)] == ["a fox and a fox", "a dog", "category", None]
    assert d.note.text().startswith("Changed 1 caption(s).") and "0 match(es)" in d.note.text()
    assert not d.apply_btn.isEnabled()
    assert win.a_undo.text() == "Undo find and replace"
    win.undo()
    assert _file(tmp_path, 0) == "a cat and a cat"
    d.regex.setChecked(True)
    d.words.setChecked(False)
    d.find_box.setText("(")
    d.count()
    assert "regular expression" in d.note.text() and not d.apply_btn.isEnabled()
    d.deleteLater()


def test_add_text_dialog_applies_to_the_selected_images(win, tmp_path):
    _add(win, tmp_path, ["a man", "a woman", "ohwx, a dog", None])
    win.grid.selectAll()
    win.grid.selectionModel().select(win.proxy.index(1, 0), win.grid.selectionModel().SelectionFlag.Deselect)
    d = _dialog(win, edits.START)
    assert d.r_selected.isChecked() and not d.form.isRowVisible(d.find_box)
    d.text_box.setText("ohwx, ")
    d.count()
    assert d.note.text().startswith("1 of 3 caption(s) get the text.")  # not the empty one or the one that has it
    d.run()
    assert [_file(tmp_path, i) for i in range(4)] == ["ohwx, a man", "a woman", "ohwx, a dog", None]
    d.r_shown.setChecked(True)
    d.action.setCurrentIndex(d.action.findData(edits.END))
    d.text_box.setText(", film grain")
    d.run()
    assert _file(tmp_path, 1) == "a woman, film grain" and _file(tmp_path, 3) is None
    assert win.a_undo.text() == "Undo add to the end (3)"
    d.deleteLater()


def test_edit_dialog_skips_captions_that_couldnt_be_read(win, boxes, tmp_path):
    _add(win, tmp_path, ["a cat"])
    locked = _image(tmp_path / "locked.png")
    os.mkdir(tmp_path / "locked.txt")  # a folder where the caption file would be: can't be read
    win.model.add_paths([locked])
    assert win.model.items[1].get("unreadable")
    assert win._captions("shown") == [(0, "a cat")]


def test_caption_check_settings_round_trip(monkeypatch):
    monkeypatch.setattr(settings_dialog, "detect_gpus", lambda: [])
    d = settings_dialog.SettingsDialog(dict(DEFAULT_SETTINGS, flag_words="ohwx", review_skip="length,duplicate"))
    off = [code for code, c in d.review_checks if not c.isChecked()]
    assert off == ["length", "duplicate"] and d.flag_words.text() == "ohwx"
    dict(d.review_checks)["length"].setChecked(True)
    d.flag_words.setText(" ohwx, blonde hair ")
    out = d.result_settings()
    assert out["review_skip"] == "duplicate" and out["flag_words"] == "ohwx, blonde hair"
    d.deleteLater()


def test_settings_reach_the_check(win, tmp_path):
    _add(win, tmp_path, ["ohwx person, a man on a beach"])
    win.s["flag_words"] = "ohwx"
    win._apply_review_options()
    assert _issues(win.model, 0) == ["word"]
    win.s["review_skip"] = "word"
    win._apply_review_options()
    assert _issues(win.model, 0) == []


def test_undo_doesnt_read_back_the_captions_the_app_wrote(tmp_path, monkeypatch):
    # opening a file that was just written cost 47 ms each (a virus scanner's first look): minutes for
    # a job of thousands. Its size and time say as much, without opening it.
    m = _model(tmp_path, [None, "old"])
    g = m.history.group("captioning")
    for row in range(2):
        m.set_caption(row, f"caption {row}", group=g)
    with monkeypatch.context() as mp:
        mp.setattr(model_mod, "read_caption", lambda p: pytest.fail(f"read {p}"))
        assert m.undo() == ("captioning", 2, 0)
    assert _file(tmp_path, 0) is None and _file(tmp_path, 1) == "old"
    assert m.redo() == ("captioning", 2, 0) and _file(tmp_path, 0) == "caption 0"


def test_a_file_touched_but_not_changed_is_still_undone(tmp_path):
    m = _model(tmp_path, ["old"])
    m.set_caption(0, "new")
    os.utime(tmp_path / "0.txt", (1_600_000_000, 1_600_000_000))  # a sync tool set its time: same text
    assert m.undo() == ("edit", 1, 0) and _file(tmp_path, 0) == "old"
    m.redo()
    (tmp_path / "0.txt").write_text("NEW", encoding="utf-8")  # same size, other text
    assert m.undo() == ("edit", 0, 1) and _file(tmp_path, 0) == "NEW"


def test_an_options_change_is_checked_in_slices(tmp_path, monkeypatch):
    m = _model(tmp_path, [CUT] * 4)
    m.recheck()
    calls = []
    monkeypatch.setattr(m, "recheck", lambda full=True: calls.append(full))
    m.set_review_options(words=["ohwx"])
    assert calls == [False]


def test_the_prefix_reaches_the_check(tmp_path):
    m = _model(tmp_path, ["ohwx person, I'm sorry, but I can't help with that."])
    m.recheck()
    assert _issues(m, 0) == []
    m.set_review_options(prefix="ohwx person, ")
    m.recheck()
    assert _issues(m, 0) == ["refusal"]


def test_undo_keeps_a_bounded_number_of_captions(tmp_path, monkeypatch):
    monkeypatch.setattr(model_mod, "UNDO_CHANGES", 4)
    m = _model(tmp_path, ["a", "b", "c"])
    for step in range(3):
        g = m.history.group(f"step {step}")
        for row in range(3):
            m.set_caption(row, f"{step} {row}", group=g)
    assert [s["label"] for s in m.history.undo] == ["step 2"]  # 9 captions > 4: the oldest went
    m.set_caption(0, "typed")
    assert [s["label"] for s in m.history.undo] == ["step 2", "edit"]


def test_a_card_being_typed_in_stays_while_the_filter_would_hide_it(tmp_path):
    m = _model(tmp_path, [FINE, CUT])
    proxy = FilterProxy()
    proxy.setSourceModel(m)
    m.recheck()
    proxy.set_filter(mode="review")
    path = proxy.hold(proxy.index(0, 0))
    m.set_caption(1, CUT + " basket.")  # another change makes it fine: the filter would drop it
    m.recheck()
    assert proxy.rowCount() == 1
    proxy.release(path)
    QApplication.processEvents()
    assert proxy.rowCount() == 0 and not proxy.held


def test_the_grid_holds_the_card_whose_box_is_open(win, tmp_path):
    _add(win, tmp_path, [FINE, CUT])
    win.model.recheck()
    win.proxy.set_filter(mode="review")
    win.grid.edit(win.proxy.index(0, 0))
    assert win.proxy.held == {win.model.items[1]["path"]}
    win.model.set_caption(1, CUT + " basket.")
    win.model.recheck()
    editor = win.grid.indexWidget(win.proxy.index(0, 0))
    assert win.proxy.rowCount() == 1 and editor is not None  # the box and what is in it are still there
    win.grid.card.closeEditor.emit(editor, QAbstractItemDelegate.NoHint)
    QApplication.processEvents()
    assert not win.proxy.held and win.proxy.rowCount() == 0  # closed: the filter has its way


def test_no_undo_while_captioning(win, tmp_path):
    paths = _add(win, tmp_path, ["old"])
    win.model.set_caption(0, "typed")
    assert win.a_undo.isEnabled()
    win.a_stop.setEnabled(True)  # a job is running
    win._update_undo()
    assert not win.a_undo.isEnabled() and win.a_undo.toolTip() == "Not while captioning"
    win.undo()
    assert _file(tmp_path, 0) == "typed"
    win.a_stop.setEnabled(False)
    win._update_undo()
    win.undo()
    assert _file(tmp_path, 0) == "old" and paths
