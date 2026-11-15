"""The caption check: what it flags, and the ordinary captions it must leave alone."""
import time

import pytest

from captioner import check


def codes(text, **kw):
    return [c for c, _ in check.inspect(text, **kw)[0]]


GOOD = [
    "A woman in a red coat stands on a wet street at night, neon signs reflecting in the puddles.",
    "A woman sits on a bench. She is reading a book, and a dog sleeps at her feet.",
    "1girl, solo, long hair, red dress, standing, outdoors, night, city lights, looking at viewer",
    "ohwx person, a man leaning on a railing with the sea behind him",
    "A sign on the wall reads \"Sorry, we are closed\" above an empty counter.",
    "A photo of a cat on a windowsill",  # "a photo of" is a caption style, not talk about the image
    "A man says \"I can't believe it!\" while the others laugh.",
    "the the cat",  # a typo, not a loop
    "A tall tree, a short tree and a tall fence stand by a tall house in the tall grass.",
    "",
    "   ",
]


@pytest.mark.parametrize("text", GOOD)
def test_ordinary_captions_are_not_flagged(text):
    assert codes(text) == []


@pytest.mark.parametrize("text, code", [
    ("A woman stands in a field. She is holding a", "cut"),
    ("A man walks his dog. The street is lined with trees and", "cut"),
    ("I'm sorry, but I can't help with describing this image.", "refusal"),
    ("I cannot provide a caption for this picture.", "refusal"),
    ("I’m unable to describe people in images.", "refusal"),
    ("Sure, here is a caption: a cat on a sofa.", "preamble"),
    ("Here's a detailed caption for the image: a cat.", "preamble"),
    ("Caption: a cat on a sofa.", "preamble"),
    ("A cat sits on a sofa. Let me know if you would like a shorter version.", "preamble"),
    ("This image shows a cat on a sofa.", "image"),
    ("The photo depicts a busy market.", "image"),
    ("In this image, a cat sleeps.", "image"),
    ("**Subject:** a cat on a sofa.", "markup"),
    ("A cat.\n- on a sofa\n- by a window", "markup"),
    ("<think>the user wants a caption</think> A cat on a sofa.", "markup"),
    ("A cat on a sofa.<|im_end|>", "markup"),
    ("a cat, a cat on a sofa, a cat on a sofa, a cat on a sofa, a cat on a sofa", "repeat"),
    ("The room is very very very very dark.", "repeat"),
    ("She wears a hat. The sky is blue. She wears a hat. A dog runs. She wears a hat. It is warm.", "repeat"),
    ("1girl, solo, red dress, solo, smile, solo, outdoors", "repeat"),
])
def test_what_gets_flagged(text, code):
    assert code in codes(text)


def test_a_fixed_suffix_is_not_a_cut_off_caption():
    text = "A woman stands in a field. She holds a basket., film grain"
    assert "cut" in codes(text)
    assert "cut" not in codes(text, suffix=", film grain")
    # and with the suffix taken off, a caption that really stops short still shows
    assert "cut" in codes("A woman stands in a field. She holds a, film grain", suffix=", film grain")


def test_flagged_words_are_whole_words_in_any_case():
    find = check.word_finder(check.flag_words("ohwx, Blonde Hair,, \n watermark"))
    assert codes("a woman with blonde hair", find_word=find) == ["word"]
    assert codes("OHWX stands by a car", find_word=find) == ["word"]
    assert codes("a blonde-haired woman, watermarked", find_word=find) == []  # parts of longer words
    assert "\"blonde hair\"" in check.inspect("long blonde hair", find)[0][0][1]
    assert check.flag_words("") == () and check.word_finder(())("anything") is None


def test_regex_characters_in_a_flagged_word_are_plain_text():
    find = check.word_finder(["c++", "(nsfw)"])
    assert find("written in c++ on a board") == "c++" and find("a tag (nsfw) here") == "(nsfw)"
    assert find("abc") is None


def test_length_outliers_need_a_set_to_compare_with():
    usual = "a woman in a red coat stands on a wet street at night under neon signs"  # 15 words
    infos = [check.inspect(usual + f" number {i}") for i in range(9)]
    short, long = check.inspect("a woman"), check.inspect(" ".join(["word%d" % i for i in range(80)]))
    out = check.review(infos + [short, long])
    assert [c for c, _ in out[-2]] == ["length"] and "shorter" in out[-2][0][1]
    assert [c for c, _ in out[-1]] == ["length"] and "longer" in out[-1][0][1]
    assert all(o == () for o in out[:9])
    assert check.review([short, long, infos[0]]) == [(), (), ()]  # too few to say what is usual


def test_duplicates_are_flagged_on_every_copy_but_not_for_short_captions():
    a = check.inspect("A woman in a red coat stands on a street.")
    b = check.inspect("a woman  in a red coat stands on a street.")  # spacing and case don't matter
    other = check.inspect("A man in a blue coat sits on a bench.")
    tiny = check.inspect("ohwx person")
    out = check.review([a, b, other, tiny, tiny])
    assert [[c for c, _ in o] for o in out] == [["duplicate"], ["duplicate"], [], [], []]
    assert "1 other" in out[0][0][1]


def test_checks_switched_off_and_images_left_out():
    cut = check.inspect("A woman stands in a field. She is holding a")
    out = check.review([cut, None, cut], skip={"cut"})
    assert out == [(("duplicate", out[0][0][1]),), (), (("duplicate", out[0][0][1]),)]
    assert check.review([cut, None, cut], skip={"cut", "duplicate"}) == [(), (), ()]
    assert set(check.CODES) == {"cut", "refusal", "preamble", "image", "markup", "repeat", "length",
                                "duplicate", "word"}


def test_a_big_set_is_checked_quickly():
    words = "a woman in a red coat stands on a wet street at night while the rain falls on the old town".split()
    captions = [" ".join(words[i % 7:] + words[:i % 7]) + f" and picture {i}. It is the end." for i in range(20000)]
    best = 9.0
    for _ in range(2):
        t = time.perf_counter()
        infos = [check.inspect(c) for c in captions]
        check.review(infos)
        best = min(best, time.perf_counter() - t)
    assert best < 4.0, f"{best:.1f} s for 20,000 captions"


def test_a_trigger_word_in_front_doesnt_hide_the_start_checks():
    p = "ohwx person, "
    assert "refusal" in codes("ohwx person, I'm sorry, but I can't help with that.", prefix=p)
    assert "image" in codes("ohwx person, This image shows a man on a beach.", prefix=p)
    assert "preamble" in codes("ohwx person, Sure! Here is a caption: a man on a beach.", prefix=p)
    assert codes("ohwx person, a man on a beach", prefix=p) == []
    assert codes("ohwx person", prefix=p) == []  # only the trigger word: nothing to look at after it


@pytest.mark.parametrize("text", [
    "Photo of Mr. Smith in a park",
    "A portrait of J. Smith at his desk",
    "A man in a hat, e.g. in a park",
    "masterpiece, 1girl, standing. outdoors, sky",
])
def test_abbreviations_and_stray_periods_are_not_cut_off(text):
    assert "cut" not in codes(text)
