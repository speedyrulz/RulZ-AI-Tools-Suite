"""A configuration race with one configuration per GPU at the same time (race_parallel). See race.py for the idea.

The trainer keeps one model copy per GPU (data parallel). In this mode the copies do not train one LoRA together:
each configuration of a wave trains its own LoRA on its own GPU's copy, the same steps at the same moment, and the
wave ends with each configuration's probe, as in the one-at-a-time race. A configuration left over by an odd count
trains alone, over every GPU when its step has batches enough (gradient_accumulation 2 or more) and the split is
faster, else on the first GPU. The winner then trains on alone the same way.

Every configuration owns its parameters (race.Variant.build_master); the copies' wrappers carry (race.attach) the
weights of the configuration that trains there, the same storage on the GPU it lives on. Rendering (probes, samples,
saves) goes through the shared code: the configuration's weight average is loaded into the first copy's wrappers and
mirrored to the others (_rp_load_render), and the trainer's per-run attributes are bound to it (_rp_bound).

Not supported here: the identity loss (one shared module), torch.compile (the copies' threads would trace at once),
saved training states."""
from __future__ import annotations

import contextlib
import logging
import time
from types import SimpleNamespace

import torch

from . import race as race_mod
from .multigpu import SpeedMeter, job_order, split_batches, step_seconds

log = logging.getLogger("rulztrainer")
SPLIT_MIN_GAIN = 0.05  # a split must be this much faster than the first GPU alone to be used


def forward_backward(job):
    """Forward + backward of one batch of one configuration on one model copy (the trainer's `work`, with the
    configuration's settings in the job). Returns (loss value, per-image multipliers, None)."""
    from . import lora as lora_mod
    from .trainer import loss_weight

    (rmodel, rinj, rdev), batch, latents, embeds, mask, sigma, noise, regions, cur_step, n_total, ctx = job
    cfg = ctx.cfg
    latents = latents.to(rdev, torch.float32)
    embeds, mask, sigma, noise = embeds.to(rdev), mask.to(rdev), sigma.to(rdev), noise.to(rdev)
    s = sigma.view(-1, 1, 1, 1)
    noisy = (1 - s) * latents + s * noise
    target = noise - latents
    noisy_in = noisy.to(torch.bfloat16)
    pred = rmodel(noisy_in, sigma * 1000.0, embeds, mask).float()
    sq = (pred - target).pow(2)
    if regions is not None and cfg.region_weights:
        sq = sq * regions.weight.to(rdev)
    per = sq.mean(dim=(1, 2, 3))
    if regions is not None and cfg.region_weights and cfg.region_prior_weight > 0 and bool(regions.prior.any()):
        lora_mod.set_multiplier(rinj, 0.0)
        with torch.no_grad():
            base_pred = rmodel(noisy_in, sigma * 1000.0, embeds, mask).float()
        lora_mod.set_multiplier(rinj, 1.0)
        pm = regions.prior.to(rdev).float()
        prior = ((pred - base_pred).pow(2) * pm).sum(dim=(1, 2, 3)) / (pm.sum(dim=(1, 2, 3)) * pred.shape[1]).clamp(min=1)
        per = per + cfg.region_prior_weight * prior
    weights = loss_weight(cfg, sigma)
    look = ctx.look(batch, cur_step) if getattr(ctx, "look", None) else None
    if look is not None:  # look_warmup: the look filter's outliers ease in
        weights = weights * torch.tensor(look, device=weights.device, dtype=weights.dtype)
    mults: list[float] = []
    if ctx.loss_watch is not None:
        per_cpu, sig_cpu = per.detach().cpu().tolist(), sigma.detach().cpu().tolist()
        for it, lv, sv in zip(batch, per_cpu, sig_cpu):
            ctx.loss_watch.observe(cur_step, it.key, float(lv), float(sv))
            mults.append(ctx.loss_watch.multiplier(it.key))
        if cfg.per_image_lr:
            weights = weights * torch.tensor(mults, device=weights.device, dtype=weights.dtype)
    loss = (per * weights).mean() / n_total  # the step's update is the mean over all of its batches
    loss.backward()
    return loss.item(), mults, None


def work_list(entry):
    """One model copy's batches of a step, one after another: [(result, seconds, image tokens)]."""
    out = []
    for job in entry[1]:
        t0 = time.perf_counter()
        res = forward_backward(job)  # its .item() waits for the GPU, so this is the batch's own time
        out.append((res, time.perf_counter() - t0, job[2].shape[2] * job[2].shape[3]))
    return out


class RaceParallelMixin:
    """The trainer's parallel race (see the module docstring); attributes come from Trainer."""

    # ------------------------------------------------------------ setup and binding
    def _rp_init(self) -> None:
        n = len(self.replicas)
        self.rp_speed = SpeedMeter(n)  # seconds per batch of each model copy, from the waves' steps
        self.rp_lanes: list[tuple] = []  # [(configuration, [model copies])] training now
        self.rp_done = 0  # optimizer steps of every configuration together (the progress counter)
        self.rp_times: list[float] = []  # wall seconds per (lockstep) step
        self.rp_t0 = time.time()
        for v in self.race.variants:
            v.build_master("cpu")
        self.race.active = self.race.variants[0]

    @contextlib.contextmanager
    def _rp_bound(self, v, writeback: bool = True):
        """The trainer's per-run attributes (settings, probe controller, image queue, curriculum, loss watch) as
        configuration `v`'s, for the shared code that renders, saves and starts a configuration; the weight average is
        off (the average is loaded into the wrappers instead) and the identity loss is not used."""
        st = v.state
        keep = (self.cfg, self.controller, self.ema, self.loss_watch, self.batcher, self.batch_iter,
                self.curriculum_w, self.ts_logit_shift, self.all_items, self.id_loss, self.race.active)
        self.cfg, self.controller, self.ema = v.cfg, st.get("controller"), None
        self.loss_watch = st.get("loss_watch")
        self.batcher, self.batch_iter = st.get("batcher"), st.get("batch_iter")
        self.curriculum_w, self.ts_logit_shift = st.get("curriculum_w"), st.get("ts_logit_shift", 0.0)
        self.all_items = st.get("items", self.all_items)
        self.id_loss = None
        self.race.active = v
        try:
            yield
        finally:
            if writeback:
                st.update(curriculum_w=self.curriculum_w, ts_logit_shift=self.ts_logit_shift, batcher=self.batcher,
                          batch_iter=self.batch_iter, loss_watch=self.loss_watch)
            (self.cfg, self.controller, self.ema, self.loss_watch, self.batcher, self.batch_iter, self.curriculum_w,
             self.ts_logit_shift, self.all_items, self.id_loss, self.race.active) = keep

    def _rp_view(self, v) -> dict:
        """The configuration's LoRA as renders and saves use it: its weight average when it has one."""
        return v.view(use_average=True)

    @torch.no_grad()
    def _rp_load_render(self, v) -> None:
        """Put the configuration's (averaged) weights in the first copy's wrappers and mirror them to the others,
        for a probe, a sample or a save."""
        src = {name: dict(ns.adapter_params()) for name, ns in self._rp_view(v).items()}
        tmp = SimpleNamespace(weights=src, names=v.names, kind=v.kind, rank=v.rank, alpha=v.alpha, factor=v.factor,
                              scale=v.scale, dropout=v.dropout, block=v.block, constraint=v.constraint, dora=v.dora)
        race_mod.swap_in(tmp, self.race.wrappers)
        self.dp.sync_weights()

    def _rp_place(self, lanes) -> None:
        """Bring each lane's configuration to its GPU (its parameters, optimizer state and weight average), start it on
        its first turn, and make its model copies carry its weights."""
        for v, reps in lanes:
            v.move_master(self.replicas[reps[0]][2])
            if not v.started:
                with self._rp_bound(v, writeback=False):
                    self._race_start_variant(v)
            for r in reps:
                race_mod.swap_in(v, self.replicas[r][1])
            race_mod.configure_backward(v.cfg, [self.replicas[r][0] for r in reps])
        self.rp_lanes = list(lanes)

    def _rp_park(self, v) -> None:
        """A configuration that is not training: its parameters, optimizer state and average on the CPU."""
        v.move_master("cpu")

    # ------------------------------------------------------------ planning a wave
    def _rp_split_reps(self) -> list[int]:
        """The model copies a lone configuration trains over: all of them when its step has a batch for each and the
        measured speeds say that is faster than the first copy alone, else just the first."""
        n, ga = len(self.replicas), max(1, int(self.race_base.gradient_accumulation))
        k = min(n, ga)
        if k < 2:
            return [0]
        if self.rp_speed.ready():
            times = self.rp_speed.times()[:k]
            counts = split_batches(times, ga)
            if step_seconds(times, counts) >= ga * times[0] * (1 - SPLIT_MIN_GAIN):
                log.info("Race: a lone configuration trains on %s alone (splitting its %d batches over %d GPUs is "
                         "not faster)", self.replicas[0][2], ga, k)
                return [0]
        return list(range(k))

    def _rp_plan(self, wave: list) -> list[tuple]:
        """[(configuration, [model copies])]: one copy each for a wave of several, the split or the first for one."""
        if len(wave) >= 2:
            return [(v, [i]) for i, v in enumerate(wave[:len(self.replicas)])]
        return [(wave[0], self._rp_split_reps())]

    def _rp_counts(self, reps: list[int]) -> list[int]:
        """Batches per model copy of a lane's step: the lane's gradient_accumulation batches, split by speed."""
        ga = max(1, int(self.race_base.gradient_accumulation))
        if len(reps) == 1:
            return [ga]
        if self.rp_speed.ready():
            return split_batches([self.rp_speed.times()[r] for r in reps], ga)
        counts = [ga // len(reps)] * len(reps)
        for i in range(ga - sum(counts)):
            counts[i] += 1
        return counts

    # ------------------------------------------------------------ one lockstep step
    def _rp_step(self, lanes) -> None:
        """One optimizer step for every lane at the same moment: the batches of each, one model copy per GPU in
        threads, then each configuration's own optimizer step."""
        from .dataset import load_batch
        from .trainer import current_lr, draw_sigmas, free_cuda, step_grad_norm

        t_step = time.perf_counter()
        while True:
            entries, owners, step_images = [], [], {}
            sigmas: dict[int, list] = {}  # lane -> its micro-batches' sigmas (Keel's observe)
            for v, reps in lanes:
                st, cfg = v.state, v.cfg
                counts = self._rp_counts(reps)
                n_total = sum(counts)
                look = None
                if getattr(self, "look_outliers", None):  # look_warmup: this lane's weights for the outliers
                    look = (lambda batch, step, _cfg=cfg, _b=st.get("batcher"): self._look_factors(
                        batch, step, _cfg, _b, _cfg.batch_size * _cfg.gradient_accumulation))
                ctx = SimpleNamespace(cfg=cfg, loss_watch=st.get("loss_watch"), look=look)
                per_rep: list[list] = [[] for _ in reps]
                step_sigma = None  # keel_pair_sigma: the lane's one sigma this step
                for k in job_order(counts):
                    batch = next(st["batch_iter"])
                    step_images[v.idx] = step_images.get(v.idx, 0) + len(batch)
                    latents, embeds, mask, regions = load_batch(batch, cfg, st["rng"])
                    tok = latents.shape[2] * latents.shape[3]
                    shift = st["ts_logit_shift"] + (self._clean_offset(  # clean_share: the epoch's share
                        cfg, st["step"], tok, st.get("batcher"), cfg.batch_size * cfg.gradient_accumulation)
                        if cfg.clean_share else 0.0)
                    sigma, step_sigma = draw_sigmas(cfg, latents.shape[0], st["gen"], shift, tok, self.arch, step_sigma,
                                                    st.get("ts_strata"))
                    sigmas.setdefault(v.idx, []).append(sigma)
                    noise = torch.randn(latents.shape, generator=st["gen"])
                    per_rep[k].append((self.replicas[reps[k]], batch, latents, embeds, mask, sigma, noise, regions,
                                       st["step"] + 1, n_total, ctx))
                for k, r in enumerate(reps):
                    entries.append((r, per_rep[k]))
                    owners.append(v.idx)
            try:
                lists = self.dp.run(work_list, entries) if len(entries) > 1 else [work_list(entries[0])]
                break
            except torch.OutOfMemoryError:
                for _, reps in lanes:
                    for r in reps:
                        for m in self.replicas[r][1].values():
                            for p in m.adapter_params().values():
                                p.grad = None
                if not self._shrink_kept_blocks():
                    raise
                free_cuda()
        wall = time.perf_counter() - t_step
        for (r, _), lst in zip(entries, lists):
            if self.rp_done > 2:  # the first steps carry start-up costs, not the GPUs' speed
                for _, sec, tok in lst:
                    self.rp_speed.add(r, sec, tok)
        by_lane: dict[int, float] = {}
        for owner, lst in zip(owners, lists):
            by_lane[owner] = by_lane.get(owner, 0.0) + sum(res[0] for res, _, _ in lst)
        for v, reps in lanes:
            st, cfg = v.state, v.cfg
            race_mod.collect_grads(v, [self.replicas[r][1] for r in reps])
            grad_norm = step_grad_norm(st["params"], cfg.max_grad_norm)
            observe = getattr(st["optimizer"], "observe", None)
            if observe is not None:  # Keel: the step's noise levels (its bands)
                for s in sigmas.get(v.idx, ()):
                    observe(s)
            st["optimizer"].step()
            st["optimizer"].zero_grad(set_to_none=True)
            if st["ema"] is not None:
                st["ema"].update(step_images[v.idx])
            st["scheduler"].step()
            for r in reps:
                race_mod.attach(v, self.replicas[r][1])
            st["step"] += 1
            st["losses"].append(by_lane[v.idx])
            st["grad_norms"].append(grad_norm)
        self.rp_done += len(lanes)
        self.rp_times.append(wall)
        self._rp_log(lanes, by_lane, current_lr)

    def _rp_log(self, lanes, by_lane, current_lr) -> None:
        from .trainer import format_grad_norm, round_norms

        elapsed = time.time() - self.rp_t0
        total = self.race.estimate["total"]
        eta = elapsed / max(1, self.rp_done) * max(0, total - self.rp_done)
        for k, (v, reps) in enumerate(lanes):
            st = v.state
            losses = st["losses"]
            avg = sum(losses[-50:]) / len(losses[-50:])
            lr_now = current_lr(st["optimizer"], st["scheduler"])
            if st["step"] % 5 == 0 or self.rp_done <= len(lanes):
                last = self.rp_times[-5:]
                where = "+".join(str(self.replicas[r][2]) for r in reps)
                log.info("[%s] step %d/%d loss %.4f avg %.4f lr %.2e grad %s on %s %.1fs/it (last %d steps %.2fs "
                         "each)", v.name, st["step"], v.cfg.steps, by_lane[v.idx], avg, lr_now,
                         format_grad_norm(st["grad_norms"][-1], v.cfg.max_grad_norm), where,
                         elapsed / max(1, len(self.rp_times)), len(last), sum(last) / len(last))
            if k == 0:  # the Monitor's loss chart follows the first lane
                self.status.update(step=self.rp_done, total=total, loss=by_lane[v.idx], avg_loss=avg, elapsed=elapsed,
                                   eta=eta, lr=lr_now, losses=losses[-2000:], grad_norm=st["grad_norms"][-1],
                                   grad_norms=round_norms(st["grad_norms"][-2000:]), grad_clip=v.cfg.max_grad_norm,
                                   step_times=[round(t, 3) for t in self.rp_times[-2000:]], race=self._race_status(),
                                   race_step=st["step"])

    # ------------------------------------------------------------ events of a lane after a step
    def _rp_events(self, v) -> bool:
        """Loss-watch epoch end, checkpoint and sample of a lane after its step. True when something loaded other
        weights into the wrappers (the lanes then carry theirs again)."""
        st, cfg = v.state, v.cfg
        step, hit = st["step"], False
        watch = st.get("loss_watch")
        if watch is not None and watch.epoch_complete():
            with self._rp_bound(v):
                summary = watch.end_epoch(step)
                log.info("Loss watch epoch %d [%s]: %s (median residual %.3f, %d/%d active)", summary["epoch"], v.name,
                         summary["counts"], summary["median"] or 0.0, summary["active"], summary["total"])
                self.status.update(loss_watch=summary)
                self._refresh_batcher_weights()
        if step % cfg.save_every == 0 and step < cfg.steps:
            with self._rp_bound(v):
                self.save(self._rp_view(v), step)
        if cfg.sample_prompts and cfg.sample_every > 0 and step % cfg.sample_every == 0 and step < cfg.steps:
            with self._rp_bound(v):
                self._rp_load_render(v)
                self.sample_images(self.replicas[0][0], step)
            hit = True
        return hit

    def _rp_stop_requested(self) -> bool:
        return self.stop_file.exists()

    # ------------------------------------------------------------ manual drops
    def _rp_manual_drops(self, lanes: list) -> bool:
        """The Monitor's Drop button (RACE_DROP in the job folder): drop each named configuration, one of the lanes or
        not. A lane that is dropped leaves `lanes`; the others go on where they are. Returns True when anything
        changed (a lane left, or the race is decided)."""
        request = self.job_dir / "RACE_DROP"
        if not request.exists():
            return False
        try:
            names = [x.strip() for x in request.read_text(encoding="utf-8").splitlines() if x.strip()]
            request.unlink()
        except OSError:
            return False
        rr = self.race
        changed = False
        for name in names:
            v = next((x for x in rr.variants if x.name == name), None)
            if v is None or not v.alive or rr.schedule.done:
                log.warning("Race: cannot drop %r (not in the race any more)", name)
                continue
            rr.schedule.drop(v.idx)
            log.info("Race: %s dropped by request at its step %d", v.name, v.state.get("step", 0))
            self._race_drop(v)
            lanes[:] = [(x, reps) for x, reps in lanes if x is not v]
            changed = True
            if rr.schedule.done:
                break
        if names:
            self._race_report()
        return changed

    # ------------------------------------------------------------ the race
    def _rp_probe(self, v):
        """The configuration's probe at the end of its turn; returns its smoothed combined score and whether the
        controller asks to stop (only the winner may)."""
        st = v.state
        with self._rp_bound(v):
            self._rp_load_render(v)
            stop = self.run_probe(self.replicas[0][0], st["step"], st["optimizer"], st["scheduler"], self._rp_view(v),
                                  ema=st.get("ema"), sync_replicas=False)  # a pull moves v's own parameters
            score = self.controller.combined_ema if self.controller is not None else None
        return score, stop

    def _rp_train_lanes(self, lanes: list, target: int | None) -> str:
        """Train the lanes in lockstep until each reaches `target` steps (a probe, or the run's end for the winner).
        Returns "done", "stop" (a stop was requested) or "changed" (a lane was dropped by hand)."""
        while lanes and any(v.state["step"] < (target if target is not None else v.cfg.steps) for v, _ in lanes):
            self._rp_step(lanes)
            moved = False
            for v, _ in list(lanes):
                moved = self._rp_events(v) or moved
            if moved:
                self._rp_place(lanes)  # events loaded other weights: every lane carries its own again
            if self._rp_manual_drops(lanes):
                return "changed"
            if self._rp_stop_requested():
                log.info("Stop requested at step %d", lanes[0][0].state["step"])
                return "stop"
        return "done"

    def _race_parallel_main(self, items, backprop) -> None:
        """The whole parallel race and the winner's training to the end of the run."""
        from .trainer import free_cuda

        rr, base = self.race, self.race_base
        model = self.replicas[0][0]
        self._rp_init()
        v0 = rr.variants[0]
        if base.sample_prompts and base.sample_before_training:
            with self._rp_bound(v0):
                self._rp_load_render(v0)
                self.sample_images(model, 0)
        if base.probe_at_start:
            with self._rp_bound(v0):
                self._rp_load_render(v0)
                self.run_probe(model, 0, None, None, self._rp_view(v0))
            for v in rr.variants[1:]:  # every configuration starts from the same base model
                v.state["controller"].update(0, *self.last_probe)
        model.train()
        self.dp.train(True)
        self._setup_checkpointing(items, backprop)
        self.status.update(stage="training", step=0)
        self.rp_t0 = time.time()
        pe = base.probe_every
        stopped = False
        while rr.running and not stopped:
            wave = [rr.variants[i] for i in rr.schedule.wave(len(self.replicas))]
            lanes = self._rp_plan(wave)
            log.info("Race round %d: %s", rr.schedule.round, "; ".join(
                f"{v.name} on {'+'.join(str(self.replicas[r][2]) for r in reps)}" for v, reps in lanes))
            self._rp_place(lanes)
            self.status.update(race=self._race_status())
            target = (lanes[0][0].state.get("step", 0) // pe + 1) * pe
            outcome = self._rp_train_lanes(lanes, target)
            while outcome == "changed" and lanes and rr.running and any(v.state["step"] < target for v, _ in lanes):
                outcome = self._rp_train_lanes(lanes, target)  # one was dropped by hand: the others finish their turn
            if outcome == "stop":
                stopped = True
                break
            for v, reps in list(lanes):
                if not v.alive or rr.schedule.done:
                    continue
                self._rp_probe(v)
                self._race_record(v, v.state.get("controller"))
                self._rp_log_round(rr)
                for i in rr.schedule.just_dropped:
                    self._race_drop(rr.variants[i])
            for v, _ in lanes:
                if v.alive and not rr.schedule.done:
                    self._rp_park(v)
            self.rp_lanes = []
            free_cuda()
            self._race_report()
        if stopped:
            self._rp_stop_save(lanes)
            self._race_report()
            self.status.update(stage="finished", done=True)
            return
        self._rp_winner(rr.variants[rr.schedule.alive[0]], model)

    def _rp_log_round(self, rr) -> None:
        if rr.schedule.history and getattr(self, "_rp_logged_rounds", 0) < len(rr.schedule.history):
            self._rp_logged_rounds = len(rr.schedule.history)
            log.info("Race round %s", race_mod.describe_round(rr.schedule.history[-1], rr.schedule))

    def _rp_stop_save(self, lanes) -> None:
        log.warning("Stopped before the race was decided; the final saves are the configurations that were training "
                    "(the others are in race.json and their _dropped files)")
        for v, _ in lanes:
            with self._rp_bound(v):
                self.save(self._rp_view(v), v.state["step"], final=True)

    def _rp_winner(self, win, model) -> None:
        """One configuration left: it trains on alone (over every GPU when that is faster) as the run, with the run's own
        file names, to the end of the run."""
        from .trainer import free_cuda

        rr = self.race
        rr.active = win
        self._race_finish()
        pe = win.cfg.probe_every
        lanes = self._rp_plan([win])
        self._rp_place(lanes)
        stop = False
        while win.state["step"] < win.cfg.steps and not stop:
            target = min(win.cfg.steps, (win.state["step"] // pe + 1) * pe)
            outcome = self._rp_train_lanes(lanes, target)
            if outcome == "stop":
                break
            final_too = rr.schedule.keeps_all  # the last one is scored at its last step like the others were
            if win.state["step"] % pe == 0 and (win.state["step"] < win.cfg.steps or final_too) \
                    and "controller" in win.state:
                _, stop = self._rp_probe(win)
                if stop:
                    log.info("Likeness controller: stopping at step %d", win.state["step"])
                self._rp_place(lanes)
        step = win.state["step"]
        with self._rp_bound(win):
            self._rp_load_render(win)
            path = self.save(self._rp_view(win), step, final=True)
            if win.cfg.sample_prompts:
                self.sample_images(model, step)
        free_cuda()
        self._race_decide()
        self._race_report()
        self.status.update(stage="finished", done=True)
        log.info("Done. Final LoRA: %s", path)
