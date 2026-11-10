# Experimental settings: test results

What the experimental settings did to face and body likeness on Krea 2, measured in November 2026. The settings tips
follow from this page. Datasets are described only as far as a result needs: no names, prompts or file names.

All of it is **one small set of a generated fictional character**, 20 photos at 0.25 MP. A setting that helps here may
not help on another set, and most rows ran on one training seed.

## Setup

- AdamW 8-bit at 1e-4, rank 32, 2000 steps, 8 probe seeds, one probe every 200 steps. Each row changed one thing.
- Rows raced with no drops, sharing their start, image order and noise. Settings that cannot be race rows ran alone.
- **Combined** is the mean of the FaceNet face score and the body score. Rows are read on their mean from step 1000 on,
  because a best score rewards one lucky probe.
- FaceNet is the held-out judge: nothing trains on it. Identity features train on ArcFace, so a rise on ArcFace alone is
  not a gain.
- **Copy** is the closest match to a training photo, limit 0.50. High likeness with high copy is memorising.
- The body probe uses a scene and pose unlike the data, so it shows whether the subject carries over.

## How far to trust a gap

Plain AdamW ran 11 times across these tests and the earlier reference runs, on two seeds. Combined averaged **30.9**,
from 27.9 to 37.1. The same row and seed gave 29.3, 37.1 and 32.7 in three races.

- Within one race, two identical rows differed by 1.0 and by 2.5 points. Between races they differed by up to 7.8.
- Paired tests over 8 probe seeds overstate confidence, because all 8 share one set of weights. An identical twin reached
  t = 2.3.
- One more run stayed at plain settings until step 1600, because resolution step-up had not yet moved, and read 37 to 42
  at a seed where plain reads 26 to 30. The code showed no difference in its data, so it is probably a plain run's upper
  tail, though a cause not found is possible.

So a single row is not evidence below about +15 points, and a result needs a second seed.

## Results

Mean from step 1000 on. Peak copy is the highest at any probe.

| Row | Seed | Combined | Face (FaceNet) | Body | Peak copy |
|---|---|---|---|---|---|
| Plain AdamW | 42 | 29.3 / 31.8 / 37.1 / 32.7 | 46.1 to 55.9 | 12.4 to 18.2 | 0.42 |
| Plain AdamW | 1234 | 27.9 / 28.9 | 44.2 / 46.6 | 11.6 / 11.2 | 0.40 |
| **OFT** | 42 | **43.4** | 66.1 | 20.7 | 0.59 |
| **OFT** | 1234 | **42.9** | 58.1 | 27.7 | 0.43 |
| OFT + trigger attention | 1234 | 43.8 | 61.3 | 26.3 | 0.49 |
| OFT + Isolation | 42 | 45.1 | 60.0 | 30.1 | 0.58 |
| Isolation | 1234 | 54.5 | 68.0 | 41.0 | 0.68 |
| Isolation | 42 | 32.0 | 47.1 | 17.0 | 0.47 |
| Trigger attention | 42 | 39.0 | 53.6 | 24.3 | 0.45 |
| Trigger attention | 1234 | 33.6 | 53.2 | 14.0 | 0.43 |
| Keel | 42 | 37.0 | 52.7 | 21.4 | 0.39 |
| Keel | 1234 | 33.3 | 54.3 | 12.4 | 0.44 |
| Content timesteps | 42 | 34.9 | 53.5 | 16.4 | 0.43 |
| Content timesteps | 1234 | 27.3 | 43.6 | 11.1 | 0.35 |
| Keel FB (face gain 1) | 42 | 35.8 | 51.5 | 20.1 | 0.38 |
| DoRA | 42 | 35.1 | 53.5 | 16.7 | 0.40 |
| Self-training | 42 | 34.4 | 51.0 | 17.7 | 0.43 |
| Body alignment | 42 | 34.2 | 49.3 | 19.1 | 0.38 |
| Identity timesteps | 42 | 33.2 | 52.0 | 14.3 | 0.37 |
| Clean-end share 60:7:30:8:3 | 42 | 31.2 | 47.4 | 14.9 | 0.36 |
| Learning rate from likeness | 42 | 30.4 | 47.1 | 13.7 | 0.40 |
| LoKr factor 8 | 42 | 29.5 | 50.3 | 8.8 | 0.43 |
| Identity alignment | 42 | 29.3 | 45.0 | 13.7 | 0.37 |
| Direct identity loss 0.2 | 42 | 27.5 | 46.2 | 8.7 | 0.36 |
| Reward fine-tuning, 150 steps | 1234 | 28.7 before, 27.9 after | 44.5 before | 13.2 before | 0.43 |
| Even noise, 16 buckets | 42 | 26.8 | 38.4 | 15.3 | 0.38 |
| Band loss | 42 | 19.0 | 33.2 | 4.8 | 0.33 |

## What it means

- **OFT is the one confirmed gain.** It led its own base rows by 14.1 and 15.0 points on two seeds, on all 8 probe seeds,
  on face and on body, and it was still rising at step 2000. Copy crossed the limit once, at the last probe on one seed.
  Read its best file with its copy score, and look at the pictures.
- **Isolation is unsettled.** 54.5 then 32.0, the widest spread here: one run was at 49 by step 600, the other reached 46
  only by step 1800. It repeated one fault on both seeds: headshot probes came back in a training photo's room. Its first
  run's body followed the unseen scene with a closer build than OFT's, but the second did not reach that score. Its main
  job, keeping the face off strangers, was not measured.
- **Trigger attention speeds up the start.** Steps 200 to 800 averaged 34.5 and 28.2 against 19.5 to 24.1 for plain, on
  both seeds. Its off-subject share reached about zero by step 200, so the loss has little left to do after that. Late
  gains were small, and its body gain did not repeat.
- **Keel's face gain repeated, its body gain did not.** FaceNet was 52.7 and 54.3 against 44.2 to 48.7 for the plain rows
  beside it. Faces came out very alike across seeds. Keel FB tied Keel. Gradient accumulation was 1, so the consensus
  gate did nothing.
- **Direct identity loss hurt.** ArcFace read 7.7 points over FaceNet, the gap a loss trained on ArcFace would be
  expected to open, body fell to 8.7, and the built-in guard switched it off at step 2000.
- **Band loss is poor at this learning rate**, not shown broken: it was still rising at step 2000.
- **Even noise lost 10 FaceNet points** on all 8 seeds against its own base, unexplained and not repeated.
- **Ties or no sign of a gain, one seed each:** LoKr factor 8, DoRA, identity timesteps, clean-end share, body alignment,
  identity alignment, learning rate from likeness.
- **Combining OFT with trigger attention or Isolation** tied OFT alone on combined and raised copy to 0.49 and 0.58.
- **Content timesteps' climb did not repeat.**

## Not settled

- **Self-training** kept 2 of 32 pictures, so it barely ran. Its bar was the floor of 60.
- **Resolution step-up** reached 0.5 MP only at step 1600, then nothing clearly moved and steps took twice as long.
  Nothing can be credited to it. A fair test has to reach the higher resolution much earlier.
- **Reward fine-tuning** has one probe after its stage, 1 point lower. 150 steps took about 3.5 hours, 80 to 100 s each.

## Qwen-Image 2.1: optimizers and OFT

Measured in November 2026 on a different set from the Krea 2 rows above: 62 generated photos of one fictional
character at 0.25 MP, with long captions that name the scene, pose, lighting and clothing. Qwen-Image 2.1 Base int8
with its turbo LoRA for probes and Fizgig's training adapter, rank 32, 2000 steps, seed 42, 8 probe seeds, lr 1e-4
(Keel's and Automagic's start). Each row is **one separate run** (not a race, so training maps could run). The probe
prompts describe the character, so the step-0 probes already scored high (raw ArcFace 64.5 of a ceiling of 78) and copy
was 0.54 before any training: on these runs copy over 0.50 means little until it climbs well past 0.54.

| Row | Best smoothed combined | Combined from step 1000 | FaceNet from step 1000 | ArcFace from step 1000 | Body probes scored from step 1000 | Peak copy | s/step |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AdamW 8-bit, LoRA | 28.0 (step 1000) | 23.3 | 86.2 | 73.0 | 4 of 6 | 0.59 | 0.42 |
| Muon, LoRA | 29.2 (1000) | 25.9 | 86.1 | 72.4 | 3 of 6 | 0.57 | 0.8 |
| Keel, LoRA | **44.7 (1400)** | **38.2** | **87.9** | 72.0 | 3 of 6 | 0.61 | 0.67 |
| Muon, OFT | 51.4 (1200), see below | 19.3 | 85.7 | 64.5 | 1 of 6 | 0.61 | 0.45 |
| AdamW 8-bit, OFT | 45.9 (1200), see below | 16.9 | 83.5 | 62.7 | 3 of 6 | 0.64 | 0.45 |
| AdamW 8-bit, OFT, lr 5e-5 | 34.2 (1200) | 33.9 | 87.0 | 71.3 | **6 of 6** | 0.62 | 0.45 |

- **Keel led clearly**: 12 to 15 points over AdamW and Muon from step 1000, the best FaceNet at every probe from step 800,
  and still rising at step 1400. That repeats Keel's FaceNet gain on Krea 2. Its lr climbed to the 1e-3 ceiling by step
  500. One run; a second seed should confirm it. Its loss over the watched training pictures fell 44% against 15-17% for
  AdamW and Muon, and its copy rose to 0.61 against one photo of a near-duplicate pair, so it fits the set hardest.
- **Muon tied AdamW** on every score at twice the time per step. By step 2000 most of its faces had broad smiles.
- **OFT overtrained on Qwen-Image 2.1 at lr 1e-4, with either optimizer.** Combined jumped early (20-31 at step 200
  against 9-14 for LoRA) and copy reached 0.61-0.64 at step 200, then ArcFace fell below the untrained model's (to
  about 60) and FaceNet slid from step 1200. Training photos' backgrounds and outdoor scenes leaked into the probes,
  headshots included. Both OFT rows' best scores come from one probe seed rendering a standing full-body pose like the
  training photos' (body 97-107), not the seated scene asked for: read them by their means, not their bests. On Krea 2
  OFT was the one confirmed gain; on this model it needs a lower lr (next point).
- **OFT at half the lr (5e-5) fixed most of that.** Second to Keel from step 1000 (33.9 against 38.2), FaceNet 87.0,
  ArcFace held at 70-73 instead of falling, and the only row whose body probe kept its framing: scored at all six late
  probes, with five or six of eight seeds in the outfit the prompt named, where every LoRA row had gone to portraits.
  No training-photo backgrounds in the headshots. Copy still touched 0.62 once (step 1800), on the same face seed and
  near-duplicate photo as Keel's 0.61, and the scene drifted from the prompt's to rooms like the training photos'. One run.
- **Every row lost the full-body framing**: the body probe went to waist-up and then head-and-shoulders portraits by
  step 1000-1600, so the body score could not be read at several probes. Its outfit followed the set's tops, not the
  prompt's. That happened with every optimizer and both adapters, so it is not the optimizer's doing.
- **Training maps** ran on all five: about 2.5 s a capture on LoRA, 3.4-3.5 s on OFT.

## Qwen-Image 2.1 fine-tune: region weighting, steering and held-out photos

November 2026, one RTX 5090: a community fine-tune of Qwen-Image 2.1 (int8) with the family's turbo LoRA and training
adapter, the 62-picture set with long captions (59 train, 3 held out with `holdout_images: 3`), AdamW 8-bit at lr 1e-4,
rank 32, 0.25 MP, seed 42, 2000 steps, 8 probe seeds, a region sweep every 4 epochs. Each row is **one run**. *Learned*
is the share of a region's starting error that is gone at fixed noise (the region sweep); *subject* is face, hair and
skin together.

| Row | Training pictures learned: subject / face / background | Held-out subject, best (step) | Held-out at step 2000: subject / face | Gap at step 2000 | Best smoothed combined | FaceNet from step 1000 | Peak copy | All 8 body renders lost from |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Plain | 21.4 / 30.3 / 15.0 | 9.0 (944) | 5.8 / 11.8 | +15.6 | 50.0 (1400) | 87.5 | 0.63 | step 1600 |
| Class weights: face 3, hair 1.5, skin 1.5, clothes 0.5, background 0.3 | 26.5 / 46.7 / 9.3 | 8.6 (472) | 4.4 / 7.0 | +22.1 | 52.9 (1400) | 87.9 | 0.64 | step 1400 |
| Region steering, defaults | 21.7 / 31.6 / 15.0 | 8.8 (472) | 6.0 / 13.2 | +15.7 | 46.3 (1600) | 87.8 | 0.70 | step 1600 |
| Region steering read from the sweeps, held-out rule 0.05 | 21.3 / 30.3 / 14.9 | 9.0 (944) | 5.8 / 12.0 | +15.5 | 52.3 (1400) | 88.7 | 0.63 | step 1600 |

- **The measuring side worked and is cheap.** The region watch added no measurable time to a step (0.39 s). A sweep of
  all 62 pictures took 24 s. Runs that were still training alike read within half a point of each other on the sweeps
  (held-out subject 8.5, 8.4, 8.2 at step 236) while their combined probe scores differed by 10 (41.9, 31.5, 38.3 at
  step 400): same-seed runs are not identical, and the sweeps see through that where one probe does not.
- **The held-out photos showed where learning the person stopped.** Most of it was there by step 236, it peaked at step
  944 and then fell in every run, while the training pictures kept improving. FaceNet on the probes peaked at steps
  600 to 1000. The combined score's best came later (step 1400 in three runs) on a body score from 2 to 4 of the 8
  renders, after the framing had mostly gone: a late best on few renders is not to be trusted.
- **Fixed class weights memorised faster.** The training faces were learned to 47% against 30%, the held-out faces ended
  at 7% against 12%, the framing went 200 steps sooner and likeness was level.
- **Region steering changed nothing measurable**, read from the steps or from the sweeps. Its weights were mild (clothes
  typically 0.82, background 0.91), no picture triggered a per-picture rule, and a region's loss followed its own
  weight only weakly (clothes at 0.5 beside a face at 3 still learned 22% against 24%).
- **The held-out rule cancelled itself.** All five parts overfitted together, so it turned all of them down alike (0.78
  to 0.87), which a per-picture mean of 1 turns back into nearly nothing.
- **The bleed number told nothing here**: 0.98-0.99 at every noise level all run, because the long captions describe the
  person with or without the trigger word.
- **None of the four kept the full-body framing**: 2 to 4 of 8 renders lost by step 600 to 1000, all 8 by step 1400 to
  1600.

### Five more runs: three more seeds, a background-only weight, half the learning rate

Same day, same model, set, settings and held-out photos. Each row changes **one thing** from the plain row above (seed
42, lr 1e-4), which is repeated here.

| Row | Training pictures learned: subject / face / background | Held-out subject, best (step) | Held-out at step 2000: subject / face | Gap at step 2000 | FaceNet from step 1000 | Copy at step 2000 | Body renders without a body, of 8, at steps 800 / 1200 / 1600 / 2000 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Plain, seed 42 | 21.4 / 30.3 / 15.0 | 9.0 (944) | 5.8 / 11.8 | +15.6 | 87.5 | 0.63 | 2 / 5 / 8 / 8 |
| Plain, seed 43 | 20.6 / 28.8 / 14.5 | 9.0 (708) | 5.8 / 12.2 | +14.8 | 89.1 | 0.55 | 1 / 1 / 0 / 0 |
| Plain, seed 44 | 20.5 / 28.7 / 14.4 | 8.8 (708) | 6.6 / 14.0 | +13.9 | 89.3 | 0.64 | 0 / 1 / 4 / 8 |
| Plain, seed 45 | 20.6 / 30.1 / 14.5 | 9.4 (708) | 6.7 / 13.0 | +14.0 | 88.2 | 0.60 | 4 / 3 / 8 / 8 |
| Class weights: background 0.3 only | 25.1 / 36.8 / 10.7 | 8.7 (944) | 4.7 / 11.9 | +20.4 | 88.6 | 0.60 | 2 / 8 / 7 / 8 |
| Learning rate 5e-5 | 16.3 / 22.5 / 10.7 | 8.9 (1180) | 8.2 / 14.3 | +8.1 | 88.3 | 0.58 | 0 / 0 / 8 / 8 |

- **The sweeps repeat across seeds; single probes do not.** Over the four plain seeds the held-out subject differed by
  a standard deviation of 0.2 to 0.7 points at every sweep and the training subject by 0.1 to 0.4. All four peaked on
  the held-out photos at steps 708 to 944 (8.8 to 9.4%) and fell to 5.8 to 6.7% by step 2000. Their combined probe score
  at step 1000 read 48.5, 47.6, 49.5 and 63.9; FaceNet from step 1000 spread over 1.8 points. That spread is the
  yardstick for everything else on this page's Qwen-Image 2.1 rows.
- **Against that yardstick both fixed weights are worse and steering is nothing.** Face 3 and background-only 0.3 end
  the held-out subject at 4.4 and 4.7% (seeds 5.8 to 6.7) with gaps of 22 and 20 points (seeds 14 to 16); the two
  steering runs sit inside the seeds on every reading. Turning only the background down still fitted the training
  faces further (37% against 29 to 30%): what is taken from the background goes to the person.
- **Half the learning rate was the one change that helped.** The held-out subject peaked later, at the same height, and
  held (8.2% at step 2000); the gap ended at 8 points; FaceNet and ArcFace on the probes stayed inside the seeds' range,
  so no likeness was lost. Rank 32 at 1e-4 is twice the rate of the Qwen-Image 2.1 recipe scaled to that rank, which
  fits. One run.
- **Whether the body probe keeps following its prompt depends on the seed.** Seed 43 kept all eight full-body renders
  to the end, seeds 42 and 45 had lost all eight by step 1600, seed 44 by step 2000. The seeds fit the photos equally
  (the sweeps agree), so the difference is in what the LoRA does to the prompt, not in how much it learned.
- **At half the learning rate the same seed lost its framing at the same step, all at once**: none of 8 through step
  1200, 2 at step 1400, all 8 at step 1600 (seen in the renders: full-body pictures at step 1400, head-and-shoulders
  portraits in different clothes at step 1600). A lower rate did not prevent it and did not move it.
- **The best checkpoint was chosen on a probe that had lost at least half its body renders in 8 of the 9 runs**, the
  seed-43 run being the exception: a body score from one to four renders (62 to 94), a normalized body score above 100
  (104 to 115), and a smoothed body score that stays where it was when no render shows a body, so a rising face score
  keeps setting new bests. `likeness_best_min_kept` guards against this; it is off by default.

Not settled: three held-out photos (at least one with a near-duplicate still in training), one set, one model family;
half the learning rate on one seed only; why one seed keeps its framing. Whether a seed that does well here does well
on another set is not tested and not expected: the seed fixes the order of the photos and the noise drawn for them,
which change with the set.

### The best checkpoint with lost renders counted as zero (3.4.0), replayed

The nine runs' recorded probes, fed through the likeness controller again: with `likeness_framing` off (the rule the
runs trained with; it reproduces their recorded bests to the decimal) and on (a body render without a measurable body
counts as zero against the first probe, a render stops at the top of the scale).

| Row | Best as the run recorded it | Best with lost renders counted | Held-out subject peaked at |
| --- | --- | --- | --- |
| Plain, seed 42 | 50.0 at step 1400 | 39.0 at step 800 | step 944 |
| Class weights: face 3 to background 0.3 | 52.9 at step 1400 | 34.2 at step 400 | step 472 |
| Region steering, defaults | 46.3 at step 1600 | 30.9 at step 1000 | step 472 |
| Region steering read from the sweeps | 52.3 at step 1400 | 42.8 at step 1400 | step 944 |
| Plain, seed 43 (kept its framing) | 52.6 at step 1800 | 52.4 at step 1800 | step 708 |
| Plain, seed 44 | 73.6 at step 1800 | 51.6 at step 1600 | step 708 |
| Plain, seed 45 | 70.3 at step 1800 | 44.1 at step 1200 | step 708 |
| Class weights: background 0.3 only | 77.8 at step 1800 | 41.6 at step 1000 | step 944 |
| Learning rate 5e-5 | 38.7 at step 2000 | 32.5 at step 1200 | step 1180 |

- Seven bests move earlier, to probes where most of the body renders still showed a body; the run that kept its framing
  is unchanged, so the rule costs nothing where nothing was lost.
- Two still land on a probe with half the renders gone (seed 44 at step 1600, steering from the sweeps at step
  1400): the four renders that were left each scored at the top of the scale, which is what a copy of a training
  photo does.
  `likeness_best_min_kept` is the stricter guard for that.
- This is a replay. The rule did not act on these runs, and it has not yet decided a training.

## Probe prompt check: the framing by training seed

November 2026. The Qwen-Image 2.1 fine-tune, set and settings of the section above, plain training (no region
weighting), with **only the training seed changed**: 42, 43 and 44. The probe seeds are the same in every run, and the
step-0 probes are identical. The prompt check of 3.2.0 (`rulztrainer/promptcheck.py`) was run afterwards on the saved
probe renders, 8 seeds of a headshot face probe and 8 of a sitting full-body body probe per step, with the trainer's
own face and pose detectors on a CPU. No run was repeated and nothing acted on the check.

Body probe renders that still show the full-body frame, of the 8 the base model drew that way:

| Step | Seed 42 | Seed 43 | Seed 44 |
| --- | --- | --- | --- |
| 0 (base model) | 8 of 8 | 8 of 8 | 8 of 8 |
| 200 | 8 of 8 | 8 of 8 | 8 of 8 |
| 400 | 8 of 8 | 8 of 8 | 8 of 8 |
| 600 | 7 of 8 | 7 of 8 | 8 of 8 |
| 800 | 7 of 8 | 7 of 8 | 8 of 8 |
| 1000 | 6 of 8 | 7 of 8 | 7 of 8 |
| 1200 | 3 of 8 | 7 of 8 | 6 of 8 |
| 1400 | 2 of 8 | 8 of 8 | 7 of 8 |
| 1600 | 0 of 8 | 8 of 8 | 3 of 8 |
| 1800 | 0 of 8 | 8 of 8 | 2 of 8 |
| 2000 | 0 of 8 | 8 of 8 | 0 of 8 |

- **The face probe stayed a headshot in all 8 renders of every probe of every seed** (the largest face 38% to 62% of
  the picture's height against a threshold of 34%). Every render had one face, and the pose model found a person in
  every one of the 528.
- **The body probe's drift depends on the seed.** Seed 42 lost most of its full-body renders by step 1200 and all of
  them by step 1600. Seed 44 held until step 1400 and lost all of them by step 2000. Seed 43 kept all 8 from step 1400
  to the end; one of its renders had moved closer from step 600 to 1200 and came back. The region sweeps read seeds
  42 and 43 as fitting the photos about equally (the same gap between training and held-out pictures), so those two
  differ in how the LoRA treats the prompt, not in how much it has learned.
- **The check agrees with the body scorer's own count** of renders without a measurable body (seed 42: 2, 2, 4, 5, 6,
  8, 8, 8 at steps 600 to 2000 against 1, 1, 2, 5, 6, 8, 8, 8 drifted here; seed 44: 1, 1, 2, 4, 6 at steps 1000 to
  1800 against 1, 2, 1, 5, 6). It differs by a render or two because it reads the framing (face size and hips), not
  whether enough of the body can be measured to score it.
- **At step 0 every body render reads as full body** with a face of 16% to 20% of the height and both hips in the
  picture, but knees in only half of them and feet in none: the base model draws this sitting "full-body" prompt from
  head to thigh. That is why `full` asks for the hips and not for the feet.
- **As a race** (the three runs fed to the race schedule as seed rows, `race_drop_drift: 0.5` over 2 probes,
  `race_drop_lowest` off): seed 42 leaves after the step-1400 probe, seed 44 after the step-1800 probe, and seed 43 is
  the one left. This replays the recorded probes through the schedule's code; no race was trained.

Not settled: the thresholds (34% and 24% of the picture's height) were set on these same renders, so the agreement
above is a fit, not a test; other models, sets and prompts may sit elsewhere. Three seeds on one set do not say how
often a seed drifts. `race_drop_drift` and `likeness_best_min_kept` are unit-tested and have not decided a real run.
The extra prompts (`probe_checks`) have not been rendered in a real run.
