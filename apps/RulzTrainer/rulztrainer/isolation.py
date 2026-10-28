"""LoRA Isolation (after Fizgig's): the subject stays out of everyone else in a picture.

A character LoRA spreads its subject onto other people: strangers in the background, a friend beside them, a crowd.
Fizgig measured on Krea 2, two people in one LoRA, 50 split pictures of a described stranger next to the subject,
each half scored with ArcFace: strangers that read as the subject went from 48 of 50 (a normal LoRA) to 0 of 50,
and the subject's own likeness stayed (0.77 against 0.78). Its parts, all here:

1. DOP (Differential Output Preservation, Ostris's AI-Toolkit). Every picture whose caption names a subject also
   trains, every dop_every steps, with the subject swapped for its class phrase ("gweltman" -> "a man"), on a freshly
   noised copy, held to what the model predicts there with the LoRA at 0 (weight dop_weight). Without the trigger
   the LoRA learns to change nothing.
2. Several subjects, each from its own dataset with its own trigger: two faces learned side by side teach the LoRA
   that a face belongs to one trigger rather than to the class. Even for one person a second one helps.
3. The fused trigger: "gwelt, a man" trains as "gweltman" (the caption files are never changed). The plain " man"
   token is the one every other man shares, the bridge the subject's look crosses to reach them; joined onto the
   trigger it is a different token. Prompt with the joined word: "gweltman at a cafe".
4. Side-by-side pairs: per person isolation_pairs 2:1 pictures, a training photo (a close-up around the face) next to
   one of 32 fixed generated strangers, side at random, captioned stranger first ("an Indian man in his 30s on the
   left, and gweltman on the right, two people side by side, two close-up portraits": the text encoders are
   causal, so words before the trigger stay clean of it), at isolation_pair_weight x the loss.
5. Balance: smaller subject datasets repeat so each subject trains about as often as the largest.
6. What a subject is: a person gets the fused trigger and the pairs; an animal or thing the fused trigger only; a
   style keeps its trigger as written and drops it from the trigger-free caption, so the look only appears when
   called.

The strangers are ShootTheSound/fizgig-multi-person-strangers on Hugging Face (CC BY 4.0, generated people: 8
ethnicities, men and women, 20s to 50s), pinned to a revision so a run's pairs never change underneath it."""
from __future__ import annotations

import collections
import json
import logging
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("rulztrainer")

STRANGERS_REPO = "ShootTheSound/fizgig-multi-person-strangers"
STRANGERS_REVISION = "b690f7aaa2ac3deace92b8326d040d2ccfb1a8b8"
PANEL = 512  # each half of a 1024 x 512 pair
TAIL = "two people side by side, two close-up portraits"
CLASS_WORDS = ("woman", "man", "girl", "boy", "lady", "guy", "person", "child", "kid", "baby", "teenager", "dog", "cat",
               "puppy", "kitten", "horse")
PEOPLE_WORDS = ("woman", "man", "girl", "boy", "lady", "guy", "person", "child", "kid", "baby", "teenager")
KINDS = ("auto", "person", "thing", "style")
FACE_SCALE = 2.2  # a pair's subject half: a square this many face sizes around the face (else the centre square)


@dataclass
class Subject:
    dataset: str
    trigger: str
    phrase: str  # "a man" ("" for a style)
    fused: str  # "gweltman" (the trigger itself for a style)
    kind: str  # person | thing | style
    rules: list = field(default_factory=list)  # [(pattern, replacement)] read into every caption, in order

    @property
    def dop_pair(self) -> tuple[str, str]:
        """(the word in a training caption, what the trigger-free caption says instead)."""
        return self.fused, self.phrase


def _word(text: str) -> re.Pattern:
    return re.compile(r"(?<!\w)" + re.escape(text) + r"(?!\w)", re.I)


def fuse(text: str, rules) -> str:
    """A caption or prompt with a subject's trigger joined to its class word."""
    for a, b in rules or ():
        if a:
            text = _word(a).sub(lambda _m, t=b: t, text)
    return text


def dop_caption(caption: str, pairs) -> str | None:
    """The trigger-free caption: every subject's word swapped for its class phrase (dropped when that is empty), or
    None when the caption names no subject (nothing to keep apart)."""
    if not caption:
        return None
    out, hit = caption, False
    for word, phrase in pairs:
        pat = _word(word)
        if pat.search(out):
            hit = True
            out = pat.sub(lambda _m, p=phrase: p, out)
    if not hit:
        return None
    out = re.sub(r"\s*,\s*(?:,\s*)+", ", ", out)  # a dropped trigger leaves ", ," or a stray comma
    out = re.sub(r"^\s*,\s*|\s*,\s*$", "", out)
    return re.sub(r"\s{2,}", " ", out).strip()


def resolve_subject(dataset: str, trigger: str, captions: list[str], cls: str = "", kind: str = "auto") -> Subject:
    """A subject's fused trigger from its trigger, its class (cls: "man" or "a man"; empty: read from the captions'
    "<trigger>, a [young] man" openings, else the class word its trigger's captions use most) and what it is."""
    trigger = str(trigger or "").strip()
    if not trigger:
        raise ValueError(f"{dataset}: LoRA Isolation needs a trigger word for every subject dataset")
    kind = (kind or "auto").lower()
    if kind not in KINDS:
        raise ValueError(f"{dataset}: subject kind must be one of {KINDS}, got {kind!r}")
    if kind == "style":
        return Subject(dataset, trigger, "", trigger, "style", [])
    if trigger.lower().endswith(CLASS_WORDS) and not cls:  # already joined ("gweltman"): used as written
        noun = next(w for w in sorted(CLASS_WORDS, key=len, reverse=True) if trigger.lower().endswith(w))
        if len(trigger) > len(noun):
            phrase = ("an " if noun[0] in "aeiou" else "a ") + noun
            return Subject(dataset, trigger, phrase, trigger, _kind(kind, noun), [])
    phrase = str(cls or "").strip()
    if not phrase:
        pat = re.compile(r"^\s*" + re.escape(trigger) + r"\s*,?\s*(an?)\s+(?:[A-Za-z'-]+\s+){0,3}?(" +
                         "|".join(CLASS_WORDS) + r")\b", re.I)
        found = collections.Counter(f"{m.group(1).lower()} {m.group(2).lower()}" for c in captions
                                    if (m := pat.match(c)))
        if not found:
            anywhere = re.compile(r"(?<!\w)(" + "|".join(CLASS_WORDS) + r")(?!\w)", re.I)
            trig = _word(trigger)
            words = collections.Counter(w.lower() for c in captions if trig.search(c) for w in anywhere.findall(c))
            if not words:  # the trigger is prepended later (trigger_word): any caption's class words
                words = collections.Counter(w.lower() for c in captions for w in anywhere.findall(c))
            found = collections.Counter({("an " if w[0] in "aeiou" else "a ") + w: n for w, n in words.items()})
        if not found:
            raise ValueError(f"{dataset}: its captions never say what {trigger} is (woman, man, dog...): set the "
                             "subject's class, or make it a style")
        phrase = found.most_common(1)[0][0]
    if not re.match(r"^an?\s", phrase, re.I):
        phrase = ("an " if phrase[:1].lower() in "aeiou" else "a ") + phrase
    noun = phrase.split()[-1]
    fused = re.sub(r"\s+", "", trigger) + noun.lower()
    rules = [(f"{trigger}, {phrase}", fused), (f"{trigger} {phrase}", fused), (trigger, fused)]
    return Subject(dataset, trigger, phrase, fused, _kind(kind, noun), rules)


def _kind(kind: str, noun: str) -> str:
    return kind if kind in ("person", "thing") else ("person" if noun.lower() in PEOPLE_WORDS else "thing")


def fusion_warning(tokenizer, subject: Subject) -> str | None:
    """A warning when the fused word does not end in the class word's own token(s): "fyxo" + "woman" may read as
    "f|yx|ow|oman", and then the class token is not whole and the decoupling is weaker."""
    if subject.kind == "style" or not subject.phrase:
        return None
    noun = subject.phrase.split()[-1]
    try:
        fused_ids = tokenizer(subject.fused, add_special_tokens=False)["input_ids"]
        noun_ids = tokenizer(noun, add_special_tokens=False)["input_ids"]
        pieces = tokenizer.convert_ids_to_tokens(fused_ids)
    except Exception:  # noqa: BLE001
        return None
    if len(noun_ids) and fused_ids[-len(noun_ids):] == noun_ids:
        return None
    return (f"'{subject.fused}' reads as {' | '.join(p.replace('Ġ', '') for p in pieces)}: the class word "
            f"'{noun}' is not a whole token in it, so it shares less with every other {noun} than it could. A "
            "trigger that ends cleanly (a consonant cluster, e.g. 'gwelt') joins better.")


# ---------------------------------------------------------------- strangers and pairs


def strangers(models_dir: str | Path = "models") -> list[tuple[Path, dict]]:
    """[(image, {"description", "sex", ...}), ...]: the 32 strangers, downloaded once (about 28 MB) at the pinned
    revision into <models_dir>/strangers__fizgig."""
    from huggingface_hub import snapshot_download

    d = Path(models_dir) / "strangers__fizgig"
    mark = d / "revision.txt"
    if not (mark.is_file() and mark.read_text(encoding="utf-8").strip() == STRANGERS_REVISION
            and (d / "strangers.json").is_file()):
        log.info("LoRA Isolation: downloading the strangers (%s, CC BY 4.0)", STRANGERS_REPO)
        snapshot_download(STRANGERS_REPO, repo_type="dataset", revision=STRANGERS_REVISION, local_dir=str(d),
                          allow_patterns=["stranger_*.png", "strangers.json", "README.md"])
        mark.write_text(STRANGERS_REVISION + "\n", encoding="utf-8")
    people = json.loads((d / "strangers.json").read_text(encoding="utf-8"))
    return [(d / p["file"], p) for p in people if (d / p["file"]).is_file()]


def _square(im, box=None, size: int = PANEL):
    from PIL import Image

    if box is None:
        w, h = im.size
        m = min(w, h)
        box = ((w - m) // 2, (h - m) // 2, (w - m) // 2 + m, (h - m) // 2 + m)
    return im.crop(box).resize((size, size), Image.LANCZOS)


def face_box(rec: dict | None, scale: float = FACE_SCALE):
    """A square around a detected face (face analysis record), clamped into the photo, or None."""
    if not rec or not rec.get("bbox") or rec.get("det", 0.0) < 0.5:
        return None
    W, H = rec["W"], rec["H"]
    x0, y0, x1, y1 = rec["bbox"]
    side = min(scale * max(x1 - x0, y1 - y0), W, H)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    bx = min(max(0.0, cx - side / 2), W - side)
    by = min(max(0.0, cy - side / 2), H - side)
    return int(round(bx)), int(round(by)), int(round(bx + side)), int(round(by + side))


def build_pairs(subject: Subject, photos: list[Path], out_dir: Path, people, n: int, seed: int,
                analysis: dict | None = None) -> list[tuple[Path, str]]:
    """The subject's n side-by-side pictures: [(png, caption)]. Rebuilt only when what they come from changes."""
    from PIL import Image, ImageOps

    photos = sorted(photos)
    if not photos or not people or n <= 0:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    stamps = [[p.name, int(p.stat().st_mtime), p.stat().st_size] for p in photos]
    sig = {"photos": stamps, "fused": subject.fused, "n": n, "seed": seed, "revision": STRANGERS_REVISION,
           "faces": bool(analysis)}
    mark = out_dir / "pairs.json"
    try:
        old = json.loads(mark.read_text(encoding="utf-8"))
        if old.get("sig") == sig and all((out_dir / f).is_file() for f, _ in old["pairs"]):
            return [(out_dir / f, c) for f, c in old["pairs"]]
    except (OSError, ValueError, KeyError):
        pass
    for f in out_dir.glob("pair_*.png"):
        f.unlink()
    rng = random.Random(seed * 1000 + sum(map(ord, subject.fused)))
    order = list(range(len(people)))
    rng.shuffle(order)
    pairs = []
    for k in range(n):
        path, meta = people[order[k % len(order)]]
        mine = rng.choice(photos)
        left = rng.random() < 0.5
        with Image.open(mine) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            a = _square(im, face_box((analysis or {}).get(str(mine.resolve()))))
        with Image.open(path) as im:
            b = _square(im.convert("RGB"))
        sheet = Image.new("RGB", (2 * PANEL, PANEL))
        sheet.paste(a if left else b, (0, 0))
        sheet.paste(b if left else a, (PANEL, 0))
        name = f"pair_{k:02d}.png"
        sheet.save(out_dir / name)
        me, them = ("left", "right") if left else ("right", "left")
        pairs.append((name, f"{meta['description']} on the {them}, and {subject.fused} on the {me}, {TAIL}"))
    mark.write_text(json.dumps({"sig": sig, "pairs": pairs}, indent=1), encoding="utf-8")
    return [(out_dir / f, c) for f, c in pairs]


def pair_side(caption: str, fused: str) -> str:
    """The subject's panel in a side-by-side pair from its caption ("... and gweltman on the right, ..."): "left",
    "right", or "" when the caption does not say."""
    m = re.search(rf"{re.escape(fused)} on the (left|right)\b", caption, re.IGNORECASE)
    return m.group(1).lower() if m else ""


def balance_repeats(counts: dict[str, int], repeats: dict[str, int]) -> dict[str, int]:
    """Repeats per subject dataset so each comes up about as often per epoch as the largest (images x repeats)."""
    top = max((counts[d] * repeats.get(d, 1) for d in counts), default=0)
    out = {}
    for d, c in counts.items():
        r = repeats.get(d, 1)
        out[d] = max(r, int(round(top / max(1, c)))) if c else r
    return out
