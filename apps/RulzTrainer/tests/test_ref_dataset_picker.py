"""add_ref_path: the text the "Add a training dataset" button puts in the face/body reference fields."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def add(text, path):
    import webui

    return webui.add_ref_path(text, path)


def test_empty_text_gives_just_the_path():
    assert add("", "datasets/foo") == "datasets/foo"
    assert add(None, "datasets/foo") == "datasets/foo"
    assert add("\n  \n", "datasets/foo") == "datasets/foo"


def test_appends_on_its_own_line_and_keeps_typed_lines():
    out = add("C:\\photos\\me.jpg\n  extra/one  ", "datasets/foo")
    assert out == "C:\\photos\\me.jpg\n  extra/one  \ndatasets/foo"


def test_trailing_blank_lines_are_dropped_before_appending():
    assert add("a/b\n\n\n", "datasets/foo") == "a/b\ndatasets/foo"
    assert not add("a/b\n", "datasets/foo").endswith("\n")


def test_duplicate_is_skipped():
    text = "a/b\ndatasets/foo\n"
    assert add(text, "datasets/foo") == text
    assert add("  datasets/foo  ", "datasets/foo") == "  datasets/foo  "


def test_duplicate_with_slash_variants_is_skipped():
    text = "datasets/foo"
    assert add(text, "datasets/foo/") == text
    if os.name == "nt":
        assert add(text, "datasets\\foo") == text
        assert add(text, "Datasets\\FOO\\") == text
        assert add("DATASETS\\Foo", "datasets/foo") == "DATASETS\\Foo"


def test_different_paths_are_added():
    assert add("datasets/foo", "datasets/foobar") == "datasets/foo\ndatasets/foobar"
