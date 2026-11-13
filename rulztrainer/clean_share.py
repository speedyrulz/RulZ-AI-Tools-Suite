"""Clean-end share schedule (after Fizgig's MOSS-Image "Slide"): the share of training steps at the clean end.

The clean end is a noise level (sigma) below 0.5: nearly clean images, where faces and fine detail are refined. The
noisy half shapes composition. Every timestep_sampling mode puts some share of its steps there: MOSS-Image's own
pretraining curve (timestep_sampling: reference) about 11-13%, RulzTrainer's default logit-normal (mean 0) 50%. With
clean_share set, the sampled curve is moved along the logit axis until that share of its steps (after timestep_shift,
the sigma range and the 1000-step grid) lands below 0.5; its shape stays.

clean_share takes Fizgig's notation, percentages and 1-based epochs of the run's own image queue:
  "40"              40% for the whole run
  "75:5:40"         75% for the first 5 epochs, then 40%
  "60:7:30:15"      60% until epoch 7, then down in a straight line to 30% by epoch 15, 30% after
  "60:7:30:8:3"     60% until epoch 7, 30% at epoch 8, then the higher share (60%) with every 3rd epoch at the lower
                    one (30%), at half the learning rate (Fizgig's MOSS default from 28 Oct to 8 Nov 2026, Slide)
  "11:7:40:8:3"     11% until epoch 7, 40% at epoch 8, then 40% with every 3rd epoch at 11%, at half the learning
                    rate (Fizgig's MOSS default since 10 Nov 2026, for Muon)
  "16.67,75,33.33,75,50,75"
                    one share per epoch, separated by commas, the last held to the end (Fizgig's MiniMax-H3-Image
                    Likeness optimization, H3_LIKENESS); every epoch at the full learning rate

With a fifth number R, every epoch after the slide trains at the higher of the two shares and every Rth one at the
lower, whichever share comes first, at half the learning rate: those epochs reset structure, but at the full rate
they cost likeness each time. A full fine-tune counts the epochs of the schedule in rotations (a cycle of every
window), so each window of a rotation trains at the same share and the schedule lands at the same point of the run
whatever the card holds; its default is "11:4:30:8:2" (the model's own ~11% until rotation 4, so body shape and what
the model knows hold).

Fizgig measured Slide on MOSS-Image, against MOSS's own ~11% curve: the first Slide (60:7:30:15:2, back to 60% every
2nd epoch) led from epoch 12 on its hardest dataset and held 71% likeness at epoch 60; 60:7:30:8:3 then held
likeness about 5 points over 60:7:30:15:2 from epoch 9 to 20 on a 53-photo dataset. A sudden drop (75 to 40) dipped
likeness two epochs later.

Fizgig's default moved twice once its MOSS presets trained on Muon, which it says trains differently enough from AdamW
that the old order was no longer the best: on 8 November 2026 to the reverse, 30:7:60:8:3 (composition first, then
likeness), and on 10 November to 11:7:40:8:3 (SLIDE_MUON), which starts at about the model's own share. Fizgig reports
that starting at 11 stopped the hair darkening late in a run and gave higher late likeness, and that it was tuned
together with its Element boost (the face and hair counting 9x early in the run), which RulzTrainer does not have. It
still offers 60:7:30:8:3 as "slower to train but known to work well".

MiniMax-H3-Image trains on its 6 sampling noise levels only, so there the share is the steps on the last one (0.232;
1 in 6 = 16.67% is the even draw). Fizgig's Likeness optimization (9.7.0, 12 Nov 2026), H3_LIKENESS: the even share,
then 75%, 33%, 75%, 50%, 75% for epochs 1 to 6 (the low share climbing to 50 on the odd epochs, 75 on the even ones),
75% after. Fizgig trains it on Automagic v4 at 2.5e-4 with its ceiling at 2.5e-4, at 0.25 MP, for 30 epochs (a run
peaked around epoch 15), and reports its best likeness on three people's sets with it, the face shape coming earliest
and the backgrounds clean. RulzTrainer's own MiniMax-H3-Image race (1 MP, AdamW 1e-4, 1000 steps) was a tie between the
even draw and 40%; the schedule itself is not measured here.

Unmeasured here but for one Krea 2 run of 60:7:30:8:3 (it tied the plain run), and RulzTrainer's default already sits
at 50%: race a schedule against "" (off) on your own data."""
from __future__ import annotations

import math

import torch

SLIDE = "60:7:30:8:3"  # Fizgig's MOSS-Image LoRA default until 8 November 2026 (measured on AdamW)
SLIDE_MUON = "11:7:40:8:3"  # its default since 10 November 2026, for Muon
FT_SLIDE = "11:4:30:8:2"  # its full fine-tune default (in rotations)
H3_LIKENESS = "16.67,75,33.33,75,50,75"  # its MiniMax-H3-Image Likeness optimization, per epoch, 75 held after
HALF_LR = 0.5  # the learning rate of the lower-share epochs after the slide
CLEAN_SIGMA = 0.5
_SOLVE_N = 16384  # draws per bisection step (fixed seed: the same offset every time)


class PerEpoch(tuple):
    """A per-epoch schedule: one share per epoch from epoch 1, the last held after."""


def parse(spec) -> tuple | None:
    """(first, until, rest, end, every) in percent / epochs, a PerEpoch of shares for a comma list, or None for ""
    (off). Raises ValueError."""
    text = str(spec or "").replace("%", "").replace(" ", "").strip()
    if not text:
        return None
    if "," in text:
        try:
            shares = PerEpoch(float(x) for x in text.strip(",").split(","))
        except ValueError:
            raise ValueError(f"clean_share {spec!r}: a per-epoch list is shares separated by ',' "
                             "(16.67,75,33.33,75,50,75)") from None
        if not shares or any(not 0.0 < s < 100.0 for s in shares):
            raise ValueError(f"clean_share {spec!r}: shares are percentages between 0 and 100")
        return shares
    try:
        parts = [float(x) for x in text.split(":")]
    except ValueError:
        raise ValueError(f"clean_share {spec!r}: numbers separated by ':' (40, 75:5:40, 60:7:30:8:3)") from None
    if len(parts) == 1:
        parts = [parts[0], 0.0, parts[0]]
    if len(parts) not in (3, 4, 5):
        raise ValueError(f"clean_share {spec!r}: give one share (40), share:epochs:share (75:5:40), "
                         "share:epoch:share:epoch (60:7:30:15) or with a fifth number, every how many epochs after "
                         "the slide every how many epochs train at the lower share, the rest at the higher (60:7:30:8:3)")
    first, until, rest = parts[:3]
    end = parts[3] if len(parts) >= 4 else until
    every = int(parts[4]) if len(parts) == 5 else 0
    for share in (first, rest):
        if not 0.0 < share < 100.0:
            raise ValueError(f"clean_share {spec!r}: shares are percentages between 0 and 100")
    if until < 0 or end < until or every < 0:
        raise ValueError(f"clean_share {spec!r}: the epochs must not go backwards")
    return first, until, rest, end, every


def share_at(schedule, epoch: int) -> float | None:
    """The clean-end share in percent for a 1-based epoch (None: no schedule)."""
    if schedule is None:
        return None
    e = max(1, int(epoch))
    if isinstance(schedule, PerEpoch):
        return round(schedule[min(e, len(schedule)) - 1], 2)
    first, until, rest, end, every = schedule
    if e <= until:
        share = first
    elif e >= end:
        share = rest
    else:
        share = first + (rest - first) * (e - until) / (end - until)
    if every > 0 and e > end:  # after the slide: the higher share, every Nth epoch the lower one
        share = min(first, rest) if _lower_epoch(schedule, e) else max(first, rest)
    return round(share, 2)


def _lower_epoch(schedule, e: int) -> bool:
    if isinstance(schedule, PerEpoch):
        return False
    first, until, rest, end, every = schedule
    return every > 0 and e > end and (e - end) % every == 0


def lr_scale_at(schedule, epoch: int) -> float:
    """The learning-rate factor for a 1-based epoch: HALF_LR on the lower-share epochs after the slide, else 1."""
    if schedule is None or isinstance(schedule, PerEpoch) or schedule[0] == schedule[2]:
        return 1.0
    return HALF_LR if _lower_epoch(schedule, max(1, int(epoch))) else 1.0


def epoch_of(images_done: int, epoch_images: int) -> int:
    """The 1-based epoch an image count falls in."""
    return int(images_done // max(1, int(epoch_images))) + 1


def measured_share(cfg, offset: float = 0.0, tokens: int = 4096, arch=None, n: int = _SOLVE_N) -> float:
    """The percentage of the configured sampler's steps below sigma 0.5, with the curve moved by `offset` (logit)."""
    from .trainer import sample_sigmas

    gen = torch.Generator("cpu").manual_seed(1234)
    s = sample_sigmas(cfg, n, gen, offset, tokens=tokens, arch=arch)
    return float((s < CLEAN_SIGMA).float().mean()) * 100.0


def offset_for(cfg, share: float, tokens: int = 4096, arch=None) -> tuple[float, float]:
    """(logit offset, share it reaches): the move along the logit axis that puts `share`% of the configured sampler's
    steps below sigma 0.5. A larger offset means noisier steps, so the share falls as the offset rises: bisect."""
    lo, hi = -12.0, 12.0
    if measured_share(cfg, hi, tokens, arch) > share or measured_share(cfg, lo, tokens, arch) < share:
        best = lo if abs(measured_share(cfg, lo, tokens, arch) - share) < abs(
            measured_share(cfg, hi, tokens, arch) - share) else hi
        return best, measured_share(cfg, best, tokens, arch)  # out of reach (a sigma range above 0.5, say)
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if measured_share(cfg, mid, tokens, arch) > share:
            lo = mid
        else:
            hi = mid
    off = 0.5 * (lo + hi)
    return off, measured_share(cfg, off, tokens, arch)


class CleanShare:
    """A run's (or race configuration's) schedule: the logit offset for the epoch a step falls in, solved once per
    share and image size and kept. `unit` names what the schedule counts in the log (a fine-tune's rotations)."""

    def __init__(self, cfg, arch=None, log=None):
        self.cfg, self.arch, self.log = cfg, arch, log
        self.schedule = parse(cfg.clean_share)
        self._solved: dict[tuple, float] = {}
        self._last = None
        self.unit = "epoch"

    @property
    def active(self) -> bool:
        return self.schedule is not None

    def offset(self, images_done: int, epoch_images: int, tokens: int) -> float:
        if self.schedule is None:
            return 0.0
        epoch = epoch_of(images_done, epoch_images)
        share = share_at(self.schedule, epoch)
        bucket = int(round(math.sqrt(max(1, tokens)) / 8.0))  # image sizes within ~8 tokens a side share an offset
        key = (share, bucket)
        if key not in self._solved:
            off, got = offset_for(self.cfg, share, tokens, self.arch)
            self._solved[key] = off
            if abs(got - share) > 1.0 and self.log:
                self.log.warning("clean_share: %g%% at the clean end is out of reach with these timestep settings; "
                                 "%.1f%% instead", share, got)
        if self.log and share != self._last:
            half = lr_scale_at(self.schedule, epoch) != 1.0
            self.log.info("Clean-end share: %g%% of steps below sigma 0.5 from %s %d (logit offset %+.2f)%s", share,
                          self.unit, epoch, self._solved[key],
                          " - at half the learning rate (the lower share, after the slide)" if half else "")
            self._last = share
        return self._solved[key]

    def lr_scale(self, images_done: int, epoch_images: int) -> float:
        if self.schedule is None:
            return 1.0
        return lr_scale_at(self.schedule, epoch_of(images_done, epoch_images))
