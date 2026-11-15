"""Caption check: finds captions worth a second look, without a model.

inspect() looks at one caption by itself; review() adds what only shows across the whole set (a caption
much shorter or longer than the others, the same caption on two images). Nothing here touches Qt or the
disk, so it is cheap enough to run again whenever captions change.
"""
import re
import statistics
from collections import Counter

# (code, what Settings calls it): the order they are listed in
CHECKS = [
    ("cut", "Cut off: stopped by the token limit, or ends in the middle of a sentence"),
    ("refusal", "A refusal instead of a caption (\"I'm sorry, I can't…\")"),
    ("preamble", "Chat text around the caption (\"Sure, here is…\", \"Let me know if…\")"),
    ("image", "Talks about the image (\"This image shows…\")"),
    ("markup", "Markdown or leftover model tokens"),
    ("repeat", "A phrase or tag repeated over and over"),
    ("length", "Much shorter or longer than the other captions"),
    ("duplicate", "The same caption as another image"),
    ("word", "Contains one of your flagged words"),
]
CODES = [c for c, _ in CHECKS]
# a word or two per check, for the card itself (the tooltip has the whole sentence)
TAGS = {"cut": "cut off", "refusal": "refusal", "preamble": "chat text", "image": "about the image",
        "markup": "markup", "repeat": "repeats", "length": "length", "duplicate": "duplicate",
        "word": "flagged word"}

MIN_SET = 8          # length outliers need this many captions to compare with
SHORT, LONG = 3, 3   # a third of the median length or less, three times it or more
DUP_WORDS = 5        # shorter captions ("a woman", a trigger word) repeat for good reasons

_AP = "['’]"  # straight or curly apostrophe
# The three "starts with" patterns are only tried on captions whose first word could begin one (_STARTS),
# and every other pattern has a plain-text test in front of it: 20,000 captions are looked at when a big
# folder is added, and running each pattern over each caption took several seconds.
_REFUSAL = re.compile(
    rf"\W*(i{_AP}m sorry|i am sorry|sorry,|i apologi[sz]e|i can{_AP}t\b|i cannot\b|i won{_AP}t\b|i will not\b|"
    rf"i{_AP}m (unable|not able)|i am (unable|not able)|as an ai\b|unfortunately,? i\b)")
_PREAMBLE = re.compile(
    rf"\W*((sure|certainly|of course|okay|ok|absolutely)\s*[,!.:]|here{_AP}s\b|here (is|are)\b|"
    r"(caption|description|tags|answer|output)\s*:)")
_IMAGE = re.compile(
    r"\W*((this|the) (image|picture|photo|photograph|illustration|artwork) "
    r"(shows|depicts|features|is|captures|displays|portrays|presents|contains)\b|"
    r"in (this|the) (image|picture|photo|photograph)\b|(this is|here we see) an? (image|picture|photo|photograph)\b)")
_STARTS = frozenset("i i'm i’m sorry as unfortunately sure certainly of okay ok absolutely here here's here’s "
                    "caption description tags answer output this the in".split())
_CLOSINGS = ("let me know if", "i hope this helps", "is there anything else", "feel free to ask",
             "would you like me to")
_MARKUP = re.compile(r"\*\*|```|^\s{0,3}#{1,6}\s|</?think>|<\|[^|>\n]{1,40}\|>|\[/?inst\]", re.M)
_MARKUP_CHARS = "*`#<["
_LIST_LINE = re.compile(r"^\s*([-*•]|\d{1,2}[.)])\s+\S", re.M)
# a sentence that ended: a stop, then a capital letter. "Mr. Smith", "J. Smith" and "standing. outdoors"
# (a tag list with a stray period) don't count.
_SENTENCE_END = re.compile(r"(?<!\bMr)(?<!\bMrs)(?<!\bMs)(?<!\bDr)(?<!\bSt)(?<!\bvs)(?<!\b[A-Z])"
                           r"[.!?][\"'”’)\]]?\s+[\"'“‘(]?[A-Z]")
_ENDS_WELL = re.compile(r"[.!?…][\"'”’)\]*]*$")
_WORD = re.compile(r"\w+(?:['’]\w+)*")
_SENTENCES = re.compile(r"[.!?]+\s+")


def flag_words(text):
    """The words and phrases of a "flag these" setting: comma or line separated, blanks dropped."""
    return tuple(w.strip() for w in re.split(r"[,\n]", text or "") if w.strip())


def word_finder(words):
    """A function giving the first of `words` found in a caption (whole words, any case), or None."""
    words = [w for w in words if w]
    if not words:
        return lambda _text: None
    # longest first, so "red hair" is named rather than "red" when both are listed
    alt = "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))
    rx = re.compile(rf"(?<!\w)(?:{alt})(?!\w)", re.I)

    def find(text):
        m = rx.search(text)
        return m.group(0) if m else None
    return find


def _repeats(words):
    """What repeats in a loop: the same word 4 times running, a group of 2 to 12 words 3 times running,
    or None. Models that lose the thread fill the token limit this way."""
    n = len(words)
    # a loop says at least 3 words again, and every pair of neighbours in it comes 3 times: most
    # captions stop at one of these two tests
    if n - len(set(words)) < 3 or max(Counter(zip(words, words[1:])).values()) < 3:
        return None
    run = 1
    for i in range(1, n):
        run = run + 1 if words[i] == words[i - 1] else 1
        if run >= 4:
            return words[i]
    for size in range(2, min(12, n // 3) + 1):
        run = 0
        for i in range(n - size):
            run = run + 1 if words[i] == words[i + size] else 0
            if run >= 2 * size:  # `size` words matched the next `size` twice over: three in a row
                return " ".join(words[i + 1 - size:i + 1])
    return None


def _repeated_part(low, words):
    loop = _repeats(words)
    if loop:
        return loop
    # the same sentence or tag three times, even with others in between (five parts or more)
    split = []
    if low.count(".") + low.count("!") + low.count("?") >= 4:
        split.append((_SENTENCES.split(low), 4))
    if low.count(",") >= 4:
        split.append((low.split(","), 1))
    for parts, least in split:
        parts = [" ".join(p.split()).strip(" .!?") for p in parts]
        if len(parts) < 5:
            continue
        for part, n in Counter(p for p in parts if len(p.split()) >= least).most_common(1):
            if n >= 3:
                return part
    return None


def inspect(text, find_word=None, suffix="", prefix=""):
    """One caption by itself: ([(code, what is wrong)], word count, key shared by identical captions).
    `prefix` and `suffix` are the fixed texts the app adds to every caption (a trigger word, say), left
    out when looking at how it starts and ends."""
    body = text.strip()
    if not body:
        return [], 0, ""
    issues = []
    low = body.lower()
    words = _WORD.findall(low)
    prefix = prefix.strip().lower()
    start, start_words = low, words  # how it starts, after the trigger word: "ohwx, I'm sorry…"
    if prefix and low.startswith(prefix) and len(low) > len(prefix):
        start = low[len(prefix):].lstrip(" ,;:.-")
        start_words = _WORD.findall(start[:40])
    core = body
    suffix = suffix.strip()
    if suffix and core.endswith(suffix):
        core = core[:-len(suffix)].rstrip()
    # prose that stops without closing its last sentence; tag lists have no sentences to stop in
    if not _ENDS_WELL.search(core[-6:]) and _SENTENCE_END.search(core):
        issues.append(("cut", "ends in the middle of a sentence (cut off?)"))
    preamble = False
    if start_words and start_words[0] in _STARTS:
        if _REFUSAL.match(start):
            issues.append(("refusal", "looks like a refusal, not a caption"))
        if _PREAMBLE.match(start):
            preamble = True
            issues.append(("preamble", "starts with chat text, not the caption"))
        if _IMAGE.match(start):
            issues.append(("image", "talks about the image instead of what is in it"))
    if not preamble:
        tail = low[-200:]
        if any(c in tail for c in _CLOSINGS):
            issues.append(("preamble", "ends with chat text"))
    if ((any(c in low for c in _MARKUP_CHARS) and _MARKUP.search(low))
            or ("\n" in low and len(_LIST_LINE.findall(low)) >= 2)):
        issues.append(("markup", "has markdown or leftover model tokens"))
    again = _repeated_part(low, words)
    if again:
        issues.append(("repeat", f"repeats \"{again[:40]}\""))
    hit = find_word(body) if find_word else None
    if hit:
        issues.append(("word", f"contains \"{hit}\""))
    key = " ".join(low.split()) if len(words) >= DUP_WORDS else ""
    return issues, len(words), key


def review(infos, skip=()):
    """Issues per caption for a whole set: `infos` holds inspect() results (or None for an image to
    leave out). Returns a list of tuples of (code, message), without the checks named in `skip`."""
    counts = [info[1] for info in infos if info and info[1]]
    median = statistics.median(counts) if len(counts) >= MIN_SET else 0
    same = Counter(info[2] for info in infos if info and info[2])
    out = []
    for info in infos:
        if not info:
            out.append(())
            continue
        own, n, key = info
        issues = list(own)
        if median and n:
            if n * SHORT <= median:
                issues.append(("length", f"much shorter than the others ({n} words, usually {median:g})"))
            elif n >= median * LONG:
                issues.append(("length", f"much longer than the others ({n} words, usually {median:g})"))
        if key and same[key] > 1:
            issues.append(("duplicate", f"the same caption as {same[key] - 1} other image(s)"))
        out.append(tuple(i for i in issues if i[0] not in skip))
    return out
