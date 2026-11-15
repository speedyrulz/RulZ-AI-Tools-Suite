"""Text edits applied to many captions at once: find and replace, and adding text at either end.

make_edit() turns what the dialog holds into a function of one caption, so the dialog's count of what
would change and the change itself can't differ. No Qt and no disk here.
"""
import re

REPLACE, START, END = "replace", "start", "end"
ACTIONS = [(REPLACE, "Find and replace"), (START, "Add to the start"), (END, "Add to the end")]


class EditError(ValueError):
    """The edit can't be made as typed (nothing to find, a regular expression that doesn't parse)."""


def make_edit(action, find="", text="", case=False, words=False, regex=False):
    """A function caption -> (new caption, number of places changed). Raises EditError.

    Adding text leaves an empty caption empty (a caption is never just the added text, as with the
    prefix and suffix of generated captions) and skips a caption that already starts or ends with it."""
    if action == REPLACE:
        if not find:
            raise EditError("Type the text to find.")
        pattern = find if regex else re.escape(find)
        if words:
            pattern = rf"(?<!\w)(?:{pattern})(?!\w)"
        try:
            rx = re.compile(pattern, 0 if case else re.IGNORECASE)
        except re.error as e:
            raise EditError(f"That regular expression can't be used: {e}")
        if regex:
            try:
                rx.sub(text, "")  # the replacement is parsed even with nothing to replace: \2 with one
            except (re.error, IndexError) as e:  # group is refused here, not part way through a run
                raise EditError(f"The replacement can't be used: {e}")
            return lambda caption: rx.subn(text, caption)
        # as typed: a backslash or \1 in the replacement is not special
        return lambda caption: rx.subn(lambda _m: text, caption)
    if action not in (START, END):
        raise EditError(f"Unknown edit: {action}")
    if not text:
        raise EditError("Type the text to add.")

    def add(caption):
        if not caption.strip():
            return caption, 0
        # "OHWX,a man" already has "ohwx, ": case and spaces don't make it new
        if action == START:
            return (caption, 0) if caption.lstrip().lower().startswith(text.strip().lower()) else (text + caption, 1)
        body = caption.rstrip()
        return (caption, 0) if body.lower().endswith(text.strip().lower()) else (body + text, 1)
    return add


def plan(captions, edit):
    """What `edit` would do to `captions`, a list of (key, caption): ([(key, new caption)] for the ones
    that change, places changed in all)."""
    changes, places = [], 0
    for key, caption in captions:
        new, n = edit(caption)
        if n and new != caption:
            changes.append((key, new))
            places += n
    return changes, places
