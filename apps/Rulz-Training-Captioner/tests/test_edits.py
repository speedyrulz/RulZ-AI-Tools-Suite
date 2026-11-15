"""Edits made to many captions at once (find and replace, adding text)."""
import pytest

from captioner import edits
from captioner.edits import END, REPLACE, START, EditError, make_edit, plan


def test_replace_ignores_case_unless_asked():
    assert make_edit(REPLACE, "Cat", "dog")("A cat and a CAT") == ("A dog and a dog", 2)
    assert make_edit(REPLACE, "Cat", "dog", case=True)("A cat and a Cat") == ("A cat and a dog", 1)


def test_whole_words_only():
    edit = make_edit(REPLACE, "cat", "dog", words=True)
    assert edit("a cat, a category, cats, CAT.") == ("a dog, a category, cats, dog.", 2)
    # also for text that starts or ends with punctuation, where \b would not match
    assert make_edit(REPLACE, "c++", "rust", words=True)("c++ and c++, not c++x") == ("rust and rust, not c++x", 2)


def test_plain_text_is_not_a_pattern_and_the_replacement_is_as_typed():
    assert make_edit(REPLACE, "a.b", "x")("a.b aXb") == ("x aXb", 1)
    assert make_edit(REPLACE, "(1)", r"\1 $& \g<0>")("see (1)") == (r"see \1 $& \g<0>", 1)


def test_replacing_with_nothing_removes_the_text():
    assert make_edit(REPLACE, "ohwx person, ", "")("ohwx person, a man") == ("a man", 1)


def test_regular_expressions_and_groups():
    edit = make_edit(REPLACE, r"(\w+) hair", r"\1 fur", regex=True)
    assert edit("red hair, long hair") == ("red fur, long fur", 2)
    assert make_edit(REPLACE, r"^", "ohwx, ", regex=True)("a man") == ("ohwx, a man", 1)


@pytest.mark.parametrize("kw, part", [
    (dict(action=REPLACE, find=""), "find"),
    (dict(action=REPLACE, find="(", regex=True), "regular expression"),
    (dict(action=REPLACE, find="(a)", text=r"\2", regex=True), "replacement"),
    (dict(action=START, text=""), "add"),
    (dict(action=END, text=""), "add"),
    (dict(action="sideways", text="x"), "Unknown"),
])
def test_edits_that_cant_be_made_say_why(kw, part):
    with pytest.raises(EditError, match=part):
        make_edit(**kw)


def test_a_broken_pattern_is_fine_as_plain_text():
    assert make_edit(REPLACE, "(", "[")("a (b") == ("a [b", 1)


def test_adding_text_skips_empty_captions_and_ones_that_have_it():
    start = make_edit(START, text="ohwx, ")
    assert start("a man") == ("ohwx, a man", 1)
    assert start("ohwx, a man") == ("ohwx, a man", 0)
    assert start("") == ("", 0) and start("  \n") == ("  \n", 0)
    end = make_edit(END, text=", film grain")
    assert end("a man") == ("a man, film grain", 1)
    assert end("a man, film grain") == ("a man, film grain", 0)
    assert end("a man \n") == ("a man, film grain", 1)  # after the caption, not after its line break
    assert end("") == ("", 0)
    # case and spaces don't make it new
    assert start("OHWX,a man") == ("OHWX,a man", 0) and start(" ohwx, a man") == (" ohwx, a man", 0)
    assert end("a man, Film Grain") == ("a man, Film Grain", 0)


def test_plan_lists_only_what_changes():
    captions = [(0, "a cat"), (1, "a dog"), (2, "cat cat"), (3, "")]
    changes, places = plan(captions, make_edit(REPLACE, "cat", "fox"))
    assert changes == [(0, "a fox"), (2, "fox fox")] and places == 3
    # replacing text with itself finds it, but changes nothing
    assert plan(captions, make_edit(REPLACE, "cat", "cat", case=True)) == ([], 0)
    assert [a for a, _ in edits.ACTIONS] == [REPLACE, START, END]
