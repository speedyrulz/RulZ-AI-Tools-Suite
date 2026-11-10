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
