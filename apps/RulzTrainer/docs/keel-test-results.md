# Keel: test results

What has actually been measured with Keel, kept apart from the design in
[keel-optimizer.md](keel-optimizer.md). Each new run or race is added here, and the settings tips follow from this
page. Datasets are described only as far as a result needs: no names, prompts or file names.

Scores are on the trainer's normalized scale, where 0 is the base model and 100 is the mean of the references. Numbers
from different runs are not comparable when their probes, scale or weights differ.

## Where things stand (2026-11-05)

| Question | Answer | Strength |
|---|---|---|
| `automagic_groups` for Keel: `single` or `by_type`? | `single` | One set, 8 seeds, p = 0.03 |
| `automagic_max_lr` for Keel: 1e-3, 5e-4 or 3e-4? | No measurable difference at 1000 steps | One set, 8 seeds |
| Does the body carry to a pose and scene unlike the data? | Yes, at 1000 steps, all three ceilings | One set, by eye and by score |
| Is 2000 steps too many? | Not shown: scores were still rising near the end | One set |
| Keel against AdamW 8-bit | **Not measured** | |
| Hull against AdamW 8-bit | No measurable difference | One set, 4 runs each |
| `keel_fb` against `keel` | **Not measured** | |
| Face gain, consensus gate and vote, lookahead, band weights | **Not measured** | |
| Gauge preconditioner, cautious mask, paired sigma, each alone | **Not measured** | |
| Keel on face likeness | **Not measured** | |

The Wizard keeps recommending AdamW 8-bit until a race on a suitable set says otherwise.

## Keel races

Both ran locally on Qwen-Image 2.1 at 1 MP with Keel's default mechanisms, batch size 1, gradient accumulation 1
(so the consensus gate had nothing to vote on and did nothing), seed 42 and 8 probe seeds, on a small body-only set
of generated images. Probes were scored on the anchored body score alone.

### Learning-rate groups (2000 steps)

| Steps | `single` | `by_type` | Gap | Seeds favouring `single` |
|---|---|---|---|---|
| 100 to 500 | 14.6 | 10.8 | +3.8 | 6 of 8 |
| 600 to 1000 | 13.1 | 12.3 | +0.8 | 5 of 8 |
| 1100 to 1500 | 21.7 | 19.1 | +2.7 | 7 of 8 |
| 1600 to 2000 | 26.7 | 22.3 | +4.5 | 7 of 8 |
| Best smoothed | 27.2 (step 1900) | 22.8 (step 1900) | +4.4 | |

The paired test over the 8 seeds gives p = 0.03 for the second half.

- Under Keel `by_type` means separate rates for the attention and feed-forward layers.
- Both reached the 1e-3 ceiling by step 200. `single` then eased to 1.4e-4 by step 2000, while `by_type` stayed near
  5e-4.
- The two look almost the same seed by seed: the gap is real in the numbers, but subtle.
- Both best scores came at step 1900 of 2000, so this race does not show where training should stop.
- The probe matched the shot type of the data, and by step 600 every seed had become the same pose and backdrop.
  Part of the late rise may be the LoRA reproducing its photos, at about 118 passes over each one.

### Learning-rate ceiling (1000 steps)

This race used a seated, three-quarter body probe in a setting the data does not have.

| `automagic_max_lr` | Steps 100 to 300 | Steps 400 to 600 | Steps 700 to 1000 | Best smoothed |
|---|---|---|---|---|
| 1e-3 | 21.9 | 26.3 | 32.0 | 33.1 |
| 3e-4 | 21.0 | 22.7 | 31.1 | 32.2 |
| 5e-4 | 19.3 | 22.9 | 28.9 | 30.2 |

| Comparison over steps 700 to 1000 | Gap | Seeds favouring the first | Paired p |
|---|---|---|---|
| 1e-3 vs 3e-4 | +0.9 | 4 of 8 | 0.65 |
| 1e-3 vs 5e-4 | +3.1 | 5 of 8 | 0.50 |
| 3e-4 vs 5e-4 | +2.2 | 5 of 8 | 0.54 |

- The race named 1e-3 the winner, but no gap held up across seeds: this is a tie.
- The dip at steps 500 to 900 in the first race did not come back at 1e-3, so the ceiling is not a supported cause.
- The lower ceilings held their rates, so the racers really did train at a third and a half of the top rate.
- All three set their best at the last probe: 1000 steps was not enough to see them level off.
- On this probe poses and backgrounds stayed varied and the body carried over, unlike the first race's collapse.
- Single seeds moved by up to 20 points between racers, so 8 seeds cannot separate gaps of about 3 points here.

## Reference runs on vast.ai: Hull and AdamW

Keel has not run on vast.ai. These runs are Hull against AdamW 8-bit, recorded here for their baseline and for what
they show about noise. All used 0.25 MP, 2000 steps and 8 probe seeds on a generated, fictional character, with the
second face scorer and the copy check on. A race of two rows took about 80 minutes and cost about $1.

| Race | Row | Seed | Best smoothed | Peak copy score (limit 0.50) |
|---|---|---|---|---|
| 1 | AdamW | 42 | 38.8 | 0.39 for the race |
| 1 | Hull, step 2e-4 | 42 | 42.4 | 0.39 for the race |
| 2 | AdamW | 1234 | 45.0 | about 0.43 |
| 2 | Hull, step 2e-4 | 1234 | 57.2 | **0.61, over the limit** |
| 3 | Hull, step 2e-4 | 1234 | 40.0 | 0.41 |
| 3 | Hull, step 1e-4 | 1234 | 30.2 | 0.41 |

- **Run-to-run spread.** The same settings and seed gave 57.2 and 40.0 in different races, so a gap of up to about 15
  points between two single runs can be chance. What caused this spread is not established. Fresh weights are seeded
  by layer name (`race.py`), so it does not come from the row's name.
- **High scores can hide copying.** The 57.2 was the only run over the copy limit, and part of its lead probably came
  from memorising one photo.
- **Hull step.** 1e-4 was clearly worse than 2e-4 and did not reduce copying.
- **Hull against AdamW.** Without the flagged run, the two are level.
- **Watermarks.** A mark in the training images came through in some probes. Remove marks before a final LoRA.

### Repeat runs: 4 of Hull, 4 of AdamW (2026-11-05)

Same setup and character as above, with Hull's step at 2e-4 and AdamW's learning rate at 1e-4. These were separate runs,
not race rows, all on one RTX PRO 5000 (48 GB), at about an hour a run. Each optimizer ran seeds 42 and 1234 twice, with
Hull and AdamW alternating in the queue.

Best smoothed score, the highest smoothed combined score at any probe:

| Optimizer | Seed 42 | Seed 1234 |
|---|---|---|
| Hull | 32.5, 29.3 | 32.9, 39.3 |
| AdamW | 37.3, 32.3 | 30.2, 34.7 |

Means over the 4 runs, standard deviation in brackets, with the difference's approximate 95% range:

| Measure | Hull | AdamW | Hull minus AdamW |
|---|---|---|---|
| Best smoothed score | 33.5 (4.2) | 33.6 (3.1) | -0.1 ± 6.4 |
| Combined score from step 1000 on | 30.6 (5.5) | 30.9 (1.5) | -0.3 ± 7.0 |
| Body score from step 1000 on | 16.4 (2.0) | 14.6 (2.7) | +1.8 ± 4.1 |
| Face score from step 1000 on | 59.5 (5.5) | 60.8 (2.6) | -1.3 ± 7.5 |
| Peak copy score | 0.45 (0.07) | 0.41 (0.03) | +0.03 ± 0.09 |

- **Hull against AdamW.** A tie on every measure. This shows no gain from Hull, but it does not show the two are equal:
  a true gap of about 6 points either way would still fit these runs.
- **Identical runs differ.** The same settings and seed gave gaps of 3.2 to 6.4 points in best score, and up to 11.3 on
  the combined score from step 1000 on. Hull's four runs covered 10 points. Changing the seed added no visible spread
  beyond that, though only two seeds ran. Training does not repeat exactly on one card, and the cause is not known.
- **Copying.** The highest-scoring Hull run was the only one over the copy limit, 0.55 at step 1400, back to 0.40 by step
  2000. That is the second time Hull's top run has crossed the limit, so read a high Hull score with its copy score.
- **Test size.** The two spreads pool to about 3.7 points. To catch a true 5-point gap 4 times in 5 would take about 9
  runs of each row, and a 3-point gap about 24 [computed]. Single pairs of runs can't settle gaps that small.

## How to run the next tests

- Compare rows in a race, on 8 probe seeds, and repeat with a second training seed before believing a gap under about
  15 points. Identical runs differed by up to 10 points, so plan on several runs of each row for smaller gaps.
- Use a probe unlike the data as well as one like it: the first shows whether the subject carries over, and the second
  alone can reward memorising.
- Read the copy score beside the likeness score. A best file flagged by the copy check is not a clean win.
- Change one thing per row, and note in this file whether a mechanism was active: the consensus gate needs gradient
  accumulation of 2 or more, and `keel_fb`'s body share needs a face gain above 0.
- Test on fictional or synthetic characters, or adults who agreed to it.
