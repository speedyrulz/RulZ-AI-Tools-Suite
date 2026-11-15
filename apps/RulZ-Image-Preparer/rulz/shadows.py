"""Shadow lift for painted areas.

A picture is lighting x surface. The lighting is estimated with an edge-aware smoothing of log-luminance
that removes texture but keeps shadow edges sharp; what's left over is the surface detail. Inside
the painted area the lighting is raised toward the light level just around it (filled in smoothly from
the unpainted pixels). The shadow's depth is measured once for the painted area and undone by a
multiplication in linear light, the way a real light would brighten the spot, so texture and local
contrast keep their ratios instead of being flattened, and parts of the painted area that are already as
bright as their surroundings aren't touched. The tint shadows usually
carry (blue from the sky) is matched to the surroundings too, and the painted area's edge is feathered
so the change blends in.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .filters import _guided_coeffs

DEFAULT_STRENGTH, DEFAULT_BLEND = 70, 40
MASK_MAX_SIDE = 2048  # the painted area is drawn at this size or smaller, then scaled up
FIELD_MAX_SIDE = 256  # the surrounding light level is filled in at this size
CLOSE_RADIUS = 16  # at that size: dips in a shadow's share up to about 12% of the picture across are filled
# Telling lit ground from a bright thing, at that size, by how far a spot's light is from the light of the plain
# lit ground around the painting, in shadow depths (0 = as lit, 1 = a full depth darker): within LIT_LEVEL it's at
# the lit level. A spot brighter than the lit ground by BRIGHTER is part of a bright thing if some spot of that
# patch is brighter by BRIGHT_PEAK; a patch that is only a little brighter is the ground's texture. Each is
# (shadow depths, natural log), the larger of the two counting, as in a shallow shadow texture alone can be a few
# tenths of a depth. The plain lit ground's light is an average of the light outside the painting, over a near
# and a wide reach (LIT_REF_BLUR, Gaussian sigmas), leaving out the cells further than BRIGHTER's depths from either
# (bright things, a shadow's unpainted fringe): the wide one as wide as the closing, so that no bright thing the
# closing could fill makes up most of it.
LIT_LEVEL = (-0.15, 0.4)
BRIGHTER = (0.25, 0.15)
BRIGHT_PEAK = (0.5, 0.2)
LIT_REF_BLUR = (8.0, 2.0 * CLOSE_RADIUS)
FLOOD_STEP = 0.05  # how finely lit ground and bright things are grown, level by level
# With Keep lit spots, a cell is a lit spot, or a soft edge sloping down to one, only if it's lighter than the plain
# shadow around it by at least SPOT_MARGIN of the shadow's own spread (its light varies with the ground's pattern):
# on ground whose pattern is about as strong as a faint shadow, its lighter patches reach the lit level too, and
# they'd be left unlifted with all that slopes down to them. With 2, sun flecks on coarse ground (grass, leaf
# litter) stay lit and a deep shadow there is lifted as with the switch off (a faint one about 0.045 too dark, its
# pattern flatter); 3 hardly changed the faint one and left more of its flecks too bright. The plain shadow's
# middle is followed across the painting, a Gaussian SPOT_REACH of the field's longer side wide (sigma), and its
# spread is how far its cells are from that middle: one middle and spread for all of it was widened by light that
# changes across the shadow, or a dark thing in a quarter of it, until it took the switch's effect away
# everywhere. 1/16 and wider brought part of that back (a dark thing in a third of the shadow: fleck edges 0.2 too
# bright instead of 0.1); narrower takes more of a coarse pattern for changing light.
SPOT_MARGIN = 2.0
SPOT_REACH = 1 / 32
SPOT_TRIM = 4.5  # median distances (about 3 spreads) over the plain shadow's median: darker is a dark thing
MAX_LIFT = 2.5  # natural log: at most ~12x brighter
GAMMA = 2.2
LUMA = np.array([0.0722, 0.7152, 0.2126], np.float32)  # B, G, R weights of luminance


@dataclass(frozen=True)
class Stroke:
    """One brush stroke. Points are fractions of the picture's width and height; the radius is a
    fraction of its longer side, so a stroke fits the thumbnail, the preview and the full picture alike."""

    points: tuple[tuple[float, float], ...]
    radius: float
    erase: bool = False


@dataclass(frozen=True)
class ShadowEdit:
    strokes: tuple[Stroke, ...] = field(default_factory=tuple)
    strength: int = DEFAULT_STRENGTH  # 0..100
    blend: int = DEFAULT_BLEND  # 0..100, how soft the painted area's edge is
    # Keep lit spots: every lit spot in the painting is lit ground, left lit (sun through leaves), and so is a
    # thing in the shadow about as bright as lit ground. Off, a lit spot that shadow encloses is lifted like a
    # bright thing in the shadow: nothing in the picture tells the two apart (see _lit_ground).
    spots: bool = False

    @property
    def active(self) -> bool:
        return self.strength > 0 and any(not s.erase for s in self.strokes)

    def to_dict(self) -> dict:
        return {"strength": self.strength, "blend": self.blend, "spots": self.spots,
                "strokes": [{"p": [list(p) for p in s.points], "r": s.radius, "e": s.erase} for s in self.strokes]}

    @classmethod
    def from_dict(cls, d: dict) -> "ShadowEdit":
        strokes = tuple(Stroke(tuple((float(x), float(y)) for x, y in s["p"]), float(s["r"]), bool(s.get("e")))
                        for s in d.get("strokes", []) if s.get("p"))
        return cls(strokes, int(d.get("strength", DEFAULT_STRENGTH)), int(d.get("blend", DEFAULT_BLEND)),
                   bool(d.get("spots", False)))


def _draw(mask: np.ndarray, strokes) -> None:
    h, w = mask.shape
    side = max(h, w)
    for s in strokes:
        r = max(1, round(s.radius * side))
        pts = np.array([[x * w, y * h] for x, y in s.points], np.float32)
        val = 0 if s.erase else 255
        shift = 4  # sub-pixel precision for smooth strokes at small sizes
        ipts = np.round(pts * (1 << shift)).astype(np.int32)
        if len(ipts) == 1:
            cv2.circle(mask, tuple(ipts[0]), r << shift, val, -1, cv2.LINE_AA, shift)
        else:
            cv2.polylines(mask, [ipts], False, val, 2 * r, cv2.LINE_AA, shift)


def render_mask(edit: ShadowEdit, h: int, w: int, feather: bool = True) -> np.ndarray:
    """The painted area as a float32 0..1 mask of h x w, with its edge feathered by `blend`."""
    f = min(1.0, MASK_MAX_SIDE / max(h, w))
    sh, sw = max(1, round(h * f)), max(1, round(w * f))
    small = np.zeros((sh, sw), np.uint8)
    _draw(small, edit.strokes)
    m = small.astype(np.float32) / 255.0
    if feather and edit.blend > 0:
        sigma = max(sh, sw) * 0.025 * (edit.blend / 100.0) ** 1.5
        if sigma >= 0.3:
            m = cv2.GaussianBlur(m, (0, 0), sigma)
    if (sh, sw) != (h, w):
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(m, 0.0, 1.0, out=m)  # m is always a new array here: no full-size copy


def _fill(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Smoothly fill where weights are 0 from where they're 1 (push-pull), keeping known values."""
    levels = [(values * weights, weights)]
    while min(levels[-1][1].shape) > 2:
        vw, wt = levels[-1]
        size = (max(1, wt.shape[1] // 2), max(1, wt.shape[0] // 2))
        levels.append((cv2.resize(vw, size, interpolation=cv2.INTER_AREA),
                       cv2.resize(wt, size, interpolation=cv2.INTER_AREA)))
    vw, wt = levels[-1]
    total = float(wt.sum())
    est = np.full(wt.shape, float(vw.sum()) / total if total > 1e-6 else 0.0, np.float32)
    for vw, wt in reversed(levels[:-1]):
        up = cv2.resize(est, (wt.shape[1], wt.shape[0]), interpolation=cv2.INTER_LINEAR)
        known = np.divide(vw, wt, out=np.zeros_like(vw), where=wt > 1e-6)
        a = np.clip(wt * 4.0, 0.0, 1.0)  # trust the level's own value where it has enough support
        est = a * known + (1.0 - a) * up
    return est


def _fill_dips(values: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    """values raised, at each pixel, to the lowest level a path from a seed has to climb to reach it: a dip that
    no seed is in comes up to the level of its rim, while a slope down to a seed stays as it is (a grayscale
    reconstruction by erosion). values: float32 0..1; seeds: bool, same shape. With no seeds, all of it is 1.
    Each step spreads the levels by one pixel; a winding dip that would need more steps than twice the
    picture's height plus width is left part filled (too high) rather than taking seconds."""
    out = np.where(seeds, values, np.float32(1.0)).astype(np.float32)
    k = np.ones((3, 3), np.uint8)
    for _ in range(0, 2 * sum(values.shape), 8):
        before = out
        for _ in range(8):  # a few steps between checks
            out = np.maximum(cv2.erode(out, k), values)
        if np.array_equal(out, before, equal_nan=True):
            break
    return out


def _lit_level(light_s: np.ndarray, outside: np.ndarray, depth: float) -> np.ndarray | None:
    """How far each cell's light is from the light of the plain lit ground around the painting, in shadow depths
    (+ darker, - brighter), or None with no plain lit ground outside the painting. light_s: the log lighting at
    the field size; outside: bool, the cells outside the painting.
    The light around the painting that the lift aims for takes on a bright thing's light where that thing crosses
    the painting's edge, so next to it the thing doesn't look brighter than around. Here the cells outside the
    painting that are much brighter or darker than the averages of those around them (see LIT_REF_BLUR) are left
    out, a few rounds, as much on either side, and the painting is filled in from the rest. A bright thing wider
    than the closing can stay in, but its shaded part is then too wide for the closing to fill anyway."""
    hgt, wid = light_s.shape
    quarter = (max(1, round(wid / 4)), max(1, round(hgt / 4)))
    keep = outside & np.isfinite(light_s)
    for _ in range(4):
        num, den = np.where(keep, light_s, 0.0).astype(np.float32), keep.astype(np.float32)
        new = outside.copy()
        for sigma in LIT_REF_BLUR:
            if sigma > 8:  # wide: blurred at a quarter of the size, which loses nothing at this width and is quick
                n, d = (cv2.resize(cv2.GaussianBlur(cv2.resize(a, quarter, interpolation=cv2.INTER_AREA), (0, 0),
                                                    sigma / 4), (wid, hgt), interpolation=cv2.INTER_LINEAR)
                        for a in (num, den))
            else:
                n, d = (cv2.GaussianBlur(a, (0, 0), sigma) for a in (num, den))
            ref = np.divide(n, d, out=light_s.copy(), where=d > 1e-3)
            new &= np.abs(ref - light_s) <= BRIGHTER[0] * depth
        if np.array_equal(new, keep):
            break
        keep = new
    if not keep.any():
        return None
    lit = _fill(np.where(keep, light_s, 0.0).astype(np.float32), keep.astype(np.float32))
    return (lit - light_s) / depth


def _bright_things(level: np.ndarray, depth: float) -> np.ndarray:
    """The cells of bright things (see BRIGHTER, BRIGHT_PEAK), over the whole field, as a patch's brightest spot
    can be far outside the painting. level: from _lit_level; depth: the shadow's depth (natural log)."""
    brighter = np.isfinite(level) & (level < -max(BRIGHTER[0], BRIGHTER[1] / depth))
    count, labels = cv2.connectedComponents(brighter.view(np.uint8), connectivity=8)
    patch = np.asarray(labels, np.int32)
    peaked = np.zeros(count, bool)
    peaked[patch[brighter & (level < -max(BRIGHT_PEAK[0], BRIGHT_PEAK[1] / depth))]] = True
    peaked[0] = False
    return peaked[patch]


def _plain_top(level: np.ndarray, free: np.ndarray) -> np.ndarray:
    """With Keep lit spots, the level each cell must be at or under to be a lit spot or a slope down to one:
    SPOT_MARGIN of the plain shadow's spread under the plain shadow's middle there (see SPOT_MARGIN, SPOT_REACH);
    inf everywhere with no plain shadow. level: from _lit_level; free: bool, the painting's cells that are numbers
    and not of a bright thing. The plain shadow is the shadier half of them (over half a depth: no lit spots in
    it), less the cells of things much darker than it (SPOT_TRIM), whose mean would raise the middle beside
    them (smooth ground between dark patches: fleck edges 0.1 to 0.2 too bright, now 0.03). Its spread is the
    median distance from the middle, which holds while less than half the cells are off for some other reason
    (a lit spot's soft edge, the cells beside a dark thing): with the upper quartile instead, dark patches in a
    quarter of the shadow took a sun fleck among them for plain shadow. A dark thing on ground whose pattern is
    nearly as strong as its own darkness can't be told from the pattern, and still loosens the bound beside it
    (a faint shadow on coarse ground: up to 0.14 too dark within a few pixels of it, against 0.07 elsewhere);
    so does one spread for a shadow over coarse and smooth ground, on the coarse part (as with one middle)."""
    shade = free & (level > 0.5)
    if not shade.any():
        return np.full(level.shape, np.inf, np.float32)
    v = level[shade]  # less a thing much darker than the shadow, which would raise the middle beside it
    shade &= level <= np.median(v) + SPOT_TRIM * np.median(np.abs(v - np.median(v)))
    sigma = SPOT_REACH * max(level.shape)
    num, den = (cv2.GaussianBlur(a.astype(np.float32), (0, 0), sigma, borderType=cv2.BORDER_CONSTANT)
                for a in (np.where(shade, level, 0.0), shade))
    mid = np.divide(num, den, out=np.full(level.shape, np.median(level[shade]), np.float32), where=den > 1e-3)
    off_mid = (level - mid)[shade]
    middle = np.median(off_mid)
    spread = 1.4826 * np.median(np.abs(off_mid - middle))  # 1.4826: a median distance in spreads
    return mid + np.float32(middle - SPOT_MARGIN * spread)


def _lit_ground(level: np.ndarray, outside: np.ndarray, depth: float, spots: bool = False) -> np.ndarray:
    """Which cells of the painting are lit ground joined to the lit ground outside it, or a slope down to that:
    a lit gap between two painted shadows, or one cutting into a shadow, and its soft edges. level: float32,
    from _lit_level; outside: bool, the cells outside the painting, which are taken as they are and never
    passed through; depth: the shadow's depth (natural log), for the floors of BRIGHTER and BRIGHT_PEAK; spots:
    lit spots that shadow encloses are lit ground too (ShadowEdit.spots).
    Such a spot (sun through leaves) and a thing in the shadow about as many times brighter than the ground as the
    shadow is darker both sit at the lit level with shadow all round, and differ in nothing at the field size (in
    colour neither: sunlight differs from a shadow's skylight just as a warm thing does from grey ground), so
    which they are is the user's call. With spots, each cell of the painting at the lit level or brighter (but not
    of a bright thing) joins the lit ground once the growing below has reached its level and no bright thing has
    taken it, so a bright thing still gets its own soft edge first. A part of one at the lit level that something
    darker cuts off from its bright part (a stripe) is a lit spot, as it would be on its own, while a bright
    thing in a lit spot (a pale stone in a sun fleck, or on strongly patterned ground the pattern's lighter bits)
    reaches all of the spot first and takes it, as it takes the near part of a lit gap. Such cells, and the slopes down to them, must be lighter than the plain shadow around
    them by SPOT_MARGIN of its spread too (_plain_top): on ground whose pattern is as strong as the shadow fewer
    cells are, and some of the pattern's lighter patches still pass. That never takes away lit ground found
    without spots (a lit gap's soft edges). The cells just outside a lit spot's soft edge, fully shaded but read
    as a little less by the smoothed lighting, get their own share as beside a lit gap (up to about a quarter of
    a depth too little, at single pixels).
    A bright thing reaching out of the shadow (a person at its edge) is left out, as its shaded part can be at
    the lit level too (when it's about as many times brighter than the ground as the shadow is darker), and the
    lit ground beside it then reaches its shaded part through the soft edge where the shadow crosses it, all the
    more so when the thing itself has a soft edge. So lit ground and bright things are grown together, side by
    side, from the lowest level up (a priority flood), each taking only cells the other hasn't: lit ground from
    the lit cells outside the painting, through cells at the lit level; bright things from their lit parts (see
    BRIGHT_PEAK), through any cell. Climbing from its lit part, a bright thing reaches its own shaded part at
    lower levels than the lit ground can, round through the shadow's soft edge, so it gets there first. The other
    side of this: a bright thing in a lit gap, or at its mouth, takes the part of the gap nearer to it than to
    where the lit ground comes in, and that part is lifted as before 0.5.2.
    The slopes are the cells whose steepest way down, side by side, passes through lit ground. Only the
    painting's box and the cells round it are worked on (nothing else changes; the box is found here, not with
    _extent). Growing spreads one cell a step, each level's capped at twice the box's height plus width steps,
    and a cell that isn't a number is never passed through."""
    lo, hi = LIT_LEVEL
    finite = np.isfinite(level)
    bright_all = _bright_things(level, depth)
    tops = np.arange(lo - FLOOD_STEP, hi + FLOOD_STEP / 2, FLOOD_STEP)
    result = outside & finite & (level >= lo) & (level <= tops[-1]) & ~bright_all  # outside: as it is
    x, y, bw, bh = cv2.boundingRect((~outside).view(np.uint8))
    if not (bw and bh):
        return result
    box = _grow((slice(y, y + bh), slice(x, x + bw)), 1, *level.shape)
    # with spots: never less than without (a lit gap's soft edges, which plain_top would otherwise cut off), and
    # lighter than the plain shadow around by SPOT_MARGIN of its spread, worked out on the whole field
    without = _lit_ground(level, outside, depth) if spots else None
    plain_top = _plain_top(level, ~outside & finite & ~bright_all)[box] if spots else np.inf
    spot_top = np.minimum(hi, plain_top)
    level, outside, finite, bright = level[box], outside[box], finite[box], bright_all[box]
    lit = result[box].copy()
    free = ~outside & finite & ~bright
    cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    for top in tops:
        open_b = free & (level <= top) & ~lit & ~bright
        open_l = open_b & (level >= lo)
        for _ in range(2 * sum(level.shape)):
            to_b = (cv2.dilate(bright.view(np.uint8), cross) > 0) & open_b
            to_l = (cv2.dilate(lit.view(np.uint8), cross) > 0) & open_l & ~to_b  # a tie: the bright thing's
            if not (to_b.any() or to_l.any()):
                break
            bright, lit = bright | to_b, lit | to_l
            open_b &= ~(to_b | to_l)
            open_l &= ~(to_b | to_l)
        if spots:  # the cells up to this level that no bright thing has reached: lit spots (also brighter ones)
            lit |= free & (level <= np.minimum(top, spot_top)) & ~bright
    # Each cell of the painting steps to its lowest side neighbor while that's lower: the steepest way down.
    # Going only down, it never loops, so where each way leads is found by doubling how far is looked each
    # round (a cell outside the painting or of a bright thing, or with nothing lower beside it, stays where it is).
    padded = np.pad(np.where(finite, level, np.inf), 1, constant_values=np.inf)
    sides = np.stack([padded[:-2, 1:-1], padded[2:, 1:-1], padded[1:-1, :-2], padded[1:-1, 2:]])
    which = np.argmin(sides, axis=0)
    down = ~outside & ~bright & finite & (np.take_along_axis(sides, which[None], 0)[0] < level)
    wid = level.shape[1]
    here = np.arange(level.size).reshape(level.shape)
    step = np.where(down, here + np.array([-wid, wid, -1, 1])[which], here).ravel()
    ends_lit = lit.ravel()
    for _ in range(max(1, int(level.size).bit_length())):
        ends_lit = ends_lit | ends_lit[step]
        step = step[step]
    ends_lit = ends_lit.reshape(level.shape)
    if spots:  # a slope is a soft edge, lighter than the plain shadow: not plain shadow whose texture leads down there
        ends_lit &= lit | (level <= plain_top)
    result[box] = ends_lit
    return result if without is None else result | without


def _lighting(log_y: np.ndarray) -> np.ndarray:
    """The lighting, without the surface detail: a rolling-guidance smoothing of log-luminance. It starts
    from a blur, which has no texture, and each pass guides the original by it, so large steps such as
    shadow edges come back sharp while small-scale texture never does."""
    h, w = log_y.shape
    radius = max(2, round(max(h, w) * 0.012))
    step = max(1, radius // 4)
    size = (max(1, w // step), max(1, h // step))
    small = cv2.resize(log_y, size, interpolation=cv2.INTER_AREA) if step > 1 else log_y
    r = max(1, radius // step)
    guide = cv2.GaussianBlur(small, (0, 0), r / 2.0)
    for _ in range(4):
        a, b = _guided_coeffs(guide, small, r, 0.01)
        guide = a * guide + b
    # Bring it to full size following the picture's own edges: where the smoothed lighting has an edge,
    # the output takes the sharp full-size edge; where the picture only has texture, it stays flat.
    a, b = _guided_coeffs(small, guide, 1, 0.01)  # a small window, so the edge band stays thin
    if step > 1:
        a = cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)
        b = cv2.resize(b, (w, h), interpolation=cv2.INTER_LINEAR)
    return a * log_y + b


def _extent(m: np.ndarray) -> tuple[slice, slice] | None:
    """Rows and columns of the smallest box holding every pixel where m isn't 0, or None when there's none."""
    x, y, bw, bh = cv2.boundingRect((m > 0).view(np.uint8))
    return (slice(y, y + bh), slice(x, x + bw)) if bw and bh else None


def _grow(box: tuple[slice, slice], by: int, h: int, w: int) -> tuple[slice, slice]:
    rows, cols = box
    return slice(max(0, rows.start - by), min(h, rows.stop + by)), slice(max(0, cols.start - by), min(w, cols.stop + by))


def _within(inner: tuple[slice, slice], outer: tuple[slice, slice]) -> tuple[slice, slice]:
    """inner's place in an array cut out at outer."""
    return tuple(slice(i.start - o.start, i.stop - o.start) for i, o in zip(inner, outer))


def _pasted(part: np.ndarray, box: tuple[slice, slice], h: int, w: int) -> np.ndarray:
    """An h x w picture that is `part` in box and 0 elsewhere."""
    full = np.zeros((h, w) + part.shape[2:], part.dtype)
    full[box] = part
    return full


def _field_size(h: int, w: int) -> tuple[int, int]:
    """(width, height) the surrounding light level is filled in at."""
    f = min(1.0, FIELD_MAX_SIDE / max(h, w))
    return max(1, round(w * f)), max(1, round(h * f))


@dataclass(frozen=True, eq=False)
class Lighting:
    """What lift_shadows measures on the picture alone, whatever is painted: the slow part of a lift. Made by
    measure() and passed back for each brush stroke on the same picture. One full-size channel (4 bytes a
    pixel), the rest at the field size; read-only, as it's shared between strokes."""

    light: np.ndarray  # log of the lighting, full size
    light_s: np.ndarray  # it at the field size
    tint: np.ndarray  # each channel's log ratio to luminance, at the field size


def measure(bgr: np.ndarray) -> Lighting:
    """The picture's Lighting for lift_shadows(bgr, ..., lighting); bgr as there."""
    return _measure(cv2.pow(np.clip(bgr, 0.0, 1.0).astype(np.float32, copy=False), GAMMA))


def _measure(lin: np.ndarray) -> Lighting:
    """lin: the clipped picture in linear light."""
    h, w = lin.shape[:2]
    y = lin @ LUMA
    light = _lighting(np.log(y + 1e-4))
    fs = _field_size(h, w)
    lin_s = cv2.resize(lin, fs, interpolation=cv2.INTER_AREA)
    y_s = cv2.resize(y, fs, interpolation=cv2.INTER_AREA)
    tint = np.log(lin_s + 1e-4) - np.log(y_s + 1e-4)[..., None]
    lighting = Lighting(light, cv2.resize(light, fs, interpolation=cv2.INTER_AREA), tint)
    for a in (lighting.light, lighting.light_s, lighting.tint):
        a.flags.writeable = False  # a change would carry over into the next stroke's result
    return lighting


def lift_shadows(bgr: np.ndarray, edit: ShadowEdit, lighting: Lighting | None = None) -> np.ndarray:
    """bgr: float32 HxWx3 in 0..1 (gamma-encoded). Returns the picture with the painted shadows lifted.
    lighting: measure(bgr), kept from an earlier call for the same picture, to skip measuring it again; the
    result is the same either way.

    The lighting and the light level around the painting are measured on the whole picture, but every
    full-size step after that is worked out only where it can change something: inside the painted area
    (where the mask isn't 0), plus a margin where how far the shadow reaches is measured. So the cost
    follows the size of the painting, and the result is the same as working out the whole picture."""
    if not edit.active:
        return bgr
    h, w = bgr.shape[:2]
    if lighting is not None and lighting.light.shape != (h, w):
        raise ValueError("the lighting was measured on a picture of another size")
    s = edit.strength / 100.0
    m = render_mask(edit, h, w)
    box = _extent(m)
    if box is None:
        return bgr

    result = np.clip(bgr, 0.0, 1.0).astype(np.float32, copy=False)  # the painted box is replaced at the end
    if lighting is None:
        lin = cv2.pow(result, GAMMA)
        lighting = _measure(lin)
        lin = lin[box]
    else:
        lin = cv2.pow(result[box], GAMMA)  # pixel by pixel: the same as the whole picture's, cut to the box
    light, light_s, tint = lighting.light, lighting.light_s, lighting.tint

    # The light level around the painted area, filled in across it.
    fs = _field_size(h, w)
    m_s = cv2.resize(m, fs, interpolation=cv2.INTER_AREA)
    outside = (m_s < 0.02).astype(np.float32)
    if outside.sum() < 1:  # painted everywhere: lift toward the brighter parts of the picture
        outside = (light_s >= np.percentile(light_s, 70)).astype(np.float32)
    around = _fill(light_s, outside)
    around = cv2.resize(cv2.GaussianBlur(around, (0, 0), 1.5), (w, h), interpolation=cv2.INTER_LINEAR)

    # A shadow dims everything under it by about the same factor, whatever the surfaces are, so lift by one
    # depth measured for the whole painted area, not by each pixel's own difference from the surroundings: a
    # dark tyre isn't pushed up to a bright car's level, and a bright object in the shadow gets the full lift.
    # The depth is the median over painted pixels clearly darker than their surroundings, so lit bits painted
    # over don't water it down. With hardly any of those, there's no shadow to lift.
    # From here on, full-size work is done in the painted area's box (outside it the mask is 0, so nothing
    # changes there), and in a wider box for the shadow's reach (`part`), which is worked on at the small size:
    # wide enough for the closing there (it reads 2 * CLOSE_RADIUS around), its blur (radius 12) and the
    # resizing on either side.
    reach = _grow(box, int(np.ceil((2 * CLOSE_RADIUS + 4) * max(w / fs[0], h / fs[1]))) + 2, h, w)
    inner = _within(box, reach)
    gap_r = around[reach] - light[reach]
    gap, mb = gap_r[inner], m[box]
    painted_px = mb > 0.5
    dark = gap[painted_px & (gap > 0.3)]
    if dark.size < max(16, 0.1 * painted_px.sum()):
        return bgr
    depth = float(np.clip(np.median(dark), 0.0, MAX_LIFT))
    # How much each pixel is in the shadow. Near the edge of the painting, where the shadow's soft fringe and
    # any lit bits painted over are, it's the pixel's own share of the depth. Deeper inside, a bright object in
    # the shadow gets the full share too (below).
    radius = max((st.radius for st in edit.strokes if not st.erase), default=0.0) * max(fs)
    band = max(1.0, 0.35 * radius)
    inside = cv2.distanceTransform((m_s > 0.5).astype(np.uint8), cv2.DIST_L2, 3)
    deep = np.clip((inside - band) / band, 0.0, 1.0)
    deep = cv2.resize(cv2.GaussianBlur(deep, (0, 0), 1.0), (w, h), interpolation=cv2.INTER_LINEAR)[box]
    # A pixel's own share of the depth is too low on a bright object in the shadow, whose edges the lighting
    # takes for less shadow, but right on the shadow's soft edge, where the light really fades in. They differ
    # in what's around them: the object is a dip or notch in the share with shadow on both sides, the soft edge
    # a slope down to lit ground. Two fills raise the dips and notches but leave a slope as it is:
    # - a grayscale closing fills any dip or notch narrower than 2 * CLOSE_RADIUS, also an object that reaches
    #   out of the shadow (a person standing at its edge) or has a lit-looking middle; but not lit ground
    #   joined to the lit ground outside the painting, or its soft edges (_lit_ground, which keeps such objects
    #   out of it): a lit gap between two painted shadows, or one cutting into a shadow, is a notch too, and
    #   filling it gave it and its edges the full depth (before 0.5.2);
    # - _fill_dips fills a dip of any size that shadow encloses (one that no path from plainly lit ground, or
    #   from outside the painting, reaches without climbing).
    # As before, the less plainly a spot is shadow, the less it's filled. (Taking every neighborhood that is a
    # third shadow as all shadow gave the soft edge the full depth: a bright ring around the shadow wherever
    # the painting ran past it.) Unless edit.spots, a lit spot that shadow encloses (sun through leaves) is a dip
    # like a bright object in the shadow, and filled; with it, the lit ground _lit_ground finds is a seed of
    # _fill_dips as well. Still lifted like a bright object: a lit gap so narrow that its middle is still mostly
    # in the shadow's soft edge, often one in a shallow shadow on ground whose texture is about as strong as the
    # shadow, the part of a lit gap near a bright thing in it or at its mouth, and partly one under lighting that
    # changes a lot across the picture.
    part_r = np.clip(gap_r / depth, 0.0, 1.0)
    part_s = cv2.resize(_pasted(part_r, reach, h, w), fs, interpolation=cv2.INTER_AREA)
    not_lit = np.clip((cv2.GaussianBlur(part_s, (0, 0), 3.0) - 0.1) / 0.25, 0.0, 1.0)
    window = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * CLOSE_RADIUS + 1,) * 2)
    closed = cv2.morphologyEx(part_s, cv2.MORPH_CLOSE, window)
    level = _lit_level(light_s, m_s < 0.02, depth)  # from the whole picture's lighting: the same in the box
    seeds = (not_lit <= 0.0) | (m_s < 0.02)  # outside the painting: seeds, so the fill never reaches past `reach`
    if level is not None:
        lit = _lit_ground(level, m_s < 0.02, depth, edit.spots)
        closed = np.where(lit, part_s, closed)
        if edit.spots:  # the lit spots themselves; their soft edges are slopes down to them, which it leaves as they
            seeds |= lit & (level <= LIT_LEVEL[1])  # are (all the slopes would keep it out of noisy plain shadow)
    filled = np.maximum(closed, _fill_dips(part_s, seeds))
    filled = part_s + (filled - part_s) * not_lit
    # Close to the full depth counts as the full depth: in the shadow's core such small differences are the
    # measurement's wobble, so the core and anything in it are lifted by one depth, as a real light would
    # (about a twentieth of the depth more at most, on the inner side of a soft edge).
    filled += (1.0 - filled) * np.clip((filled - 0.85) / 0.1, 0.0, 1.0)
    filled = cv2.resize(filled, (w, h), interpolation=cv2.INTER_LINEAR)[box]
    share = np.maximum(part_r[inner], deep * filled)
    lift = depth * share * (s * mb)
    out = lin * np.exp(lift)[..., None]

    # Match the shadow's tint (often blue from the sky) to the surroundings. It's measured in a band on
    # each side of the edge (inside the shadow, and outside the painting), where the surface is usually
    # the same material, with medians so a brightly
    # colored object in the shadow doesn't skew it; then applied in proportion to each pixel's lift.
    lift_s = cv2.resize(_pasted(lift, box, h, w), fs, interpolation=cv2.INTER_AREA)
    k = np.ones((2 * max(2, round(max(fs) * 0.06)) + 1,) * 2, np.uint8)
    painted = (m_s > 0.02).astype(np.uint8)
    ring = (cv2.dilate(painted, k) > 0) & (outside > 0)
    shaded = (lift_s > 0.5 * s * depth).astype(np.uint8)
    shaded = (shaded > 0) & (cv2.erode(shaded, k) == 0)  # the shadow's inner edge band
    if shaded.sum() >= 8 and ring.sum() >= 8:
        cast = np.median(tint[shaded], axis=0) - np.median(tint[ring], axis=0)
        cast = np.clip(cast - cast @ LUMA, -0.5, 0.5)
        amount = np.clip(lift / max(s * depth, 1e-3), 0.0, 1.0)
        out *= np.exp(-cast[None, None, :] * amount[..., None])

    # Roll off highlights that the lift pushed past white, keeping their color.
    peak = out.max(axis=2)
    soft = np.where(peak > 0.9, 0.9 + 0.1 * (1.0 - np.exp(-(peak - 0.9) / 0.1)), peak)
    k = np.divide(soft, peak, out=np.ones_like(peak), where=peak > 1e-6)
    k = 1.0 + (k - 1.0) * np.clip(lift / 0.1, 0.0, 1.0)  # only where something was lifted
    out *= k[..., None]
    result[box] = cv2.pow(np.clip(out, 0.0, 1.0), 1.0 / GAMMA)
    return result
