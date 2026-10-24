![RulZ](simplecut/assets/logo_light.png)

# RulZ Simple Video Clip Extractor

A small Windows video trimmer: frame-accurate stepping, keyframe navigation,
In/Out marks, and export either **lossless** (stream copy, when In and Out are
on keyframes) or **re-encoded with NVIDIA NVENC** to MP4.

![RulZ Simple Video Clip Extractor with a 16:9 crop box and a 2.5 second fixed-length selection](docs/screenshot.png)

## Download

Grab the latest `RulZ-Simple-Video-Clip-Extractor-v*-win64.zip` from the
[Releases page](https://github.com/speedyrulz/rulz-simple-video-clip-extractor/releases/latest),
unzip it anywhere and run `RulZ Clip Extractor.exe`. No Python is needed, but
ffmpeg and ffprobe must be on your PATH (see below).

To build that zip yourself:

```
.venv\Scripts\python -m pip install pyinstaller
.venv\Scripts\python packaging\build.py
```

## Requirements

- Windows 10/11, Python 3.10+ on PATH
- `ffmpeg` and `ffprobe` on PATH (`winget install Gyan.FFmpeg`)
- An NVIDIA GPU with current drivers for NVENC export

## Run

Double-click `run.bat`. It creates a `.venv` on first run, installs the
dependencies from `requirements.txt` (PySide6, PyAV, numpy), and starts the app.
You can also pass a video: `run.bat "C:\path\to\video.mp4"`.

Manual alternative:

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python main.py
```

## Controls

| Action              | Button | Key           |
| ------------------- | ------ | ------------- |
| Go to start         | ⏮      | Home          |
| Previous keyframe   | ◀K     | Shift+Left    |
| Previous frame      | ◀      | Left          |
| Play / pause        | ▶      | Space         |
| Mute                | 🔊     | M             |
| Next frame          | ▶      | Right         |
| Next keyframe       | K▶     | Shift+Right   |
| Go to end           | ⏭      | End           |
| Mark In             |        | I             |
| Mark Out            |        | O             |
| Export              |        | Ctrl+E        |
| Preferences         |        | Ctrl+,        |
| Quick export        |        | Ctrl+Shift+E  |
| Open last export    |        | Ctrl+Shift+O  |
| Save frame as PNG   | 💾     | S             |
| Rotate 90° clockwise |       | R             |
| Flip horizontally   |        | F             |
| Close video         |        | Ctrl+W        |
| Toggle crop box     |        | C             |
| Toggle fixed length |        | L             |
| Match colour to reference photo | | B        |
| Move clip box (fixed length) |  | Ctrl + the playback keys |
| Next clip (fixed length) |   | N             |

Playback has sound. The speaker button mutes (M) and the slider next to it
sets the volume; both are remembered between runs. During playback the audio
is the master clock and the video frames follow it, so they stay in sync.
Files without an audio track play against a wall clock. Frame stepping and
scrubbing are silent.

Click or drag on the timeline to seek. Light ticks on the timeline are
keyframes. A red "● keyframe" tag shows next to the time when the current
frame is a keyframe. The In/Out range is shaded dark red, with a white IN and
a red OUT marker.

## Saving a frame

Press **S** (or the "Save frame PNG" button) to write the current frame at
full source resolution as a PNG. It saves immediately, with no dialog, as
`<video name>_<timecode>_f<frame>.png` in the output folder; an existing
name gets a `_1`, `_2`, ... suffix instead of being overwritten. The status
bar shows the saved path.

## Rotation and flip

Press **R** (or the Rotate button) to turn the picture 90° clockwise; each
press adds another 90° and the fourth press returns to upright. Press **F**
(or the Flip button) to mirror the picture left-to-right. Both always act on
what you see on screen, in any order: rotate always turns the current view
clockwise and flip always mirrors the current view. The label next to the
buttons shows the stored rotation and whether the view is mirrored. Both
reset when a new file is opened. Sources that carry their own rotation flag
(phone recordings) are shown upright automatically.

Exports keep the rotation and flip:

- **Frame PNG**: pixels are rotated and mirrored.
- **NVENC re-encode**: rotation and flip are baked into the pixels.
- **Lossless**: pixels cannot change without re-encoding. Rotation alone is
  written as a display-rotation flag in the container (MP4, MOV, MKV), which
  every mainstream player honors. A flip has no reliably honored flag, so
  lossless is unavailable while the view is mirrored; use NVENC.

## Open last export

**Open last export** (button beside Quick export, or File → Open last export,
Ctrl+Shift+O) closes the current video and opens the file most recently
written by Export or Quick export. It is remembered between runs and greys
out if that file no longer exists.

## Close video

**File → Close video** (Ctrl+W) unloads the current file and returns the app
to its empty state: marks, crop, fixed-length selection, rotation and flip are
all cleared.

## Crop

Press **C** (or the Crop button) to show a crop box over the video. Drag
inside the box to move it, drag an edge or corner to resize it, or drag
outside it to draw a new box. The size and position (in source pixels, after
rotation) are shown above the box and in the crop panel.

- **Aspect** locks the box shape: Free, 16:9, 9:16, 4:3, 3:4, 1:1, 21:9,
  2.39:1, 3:2, 2:3, or Custom (type `W:H`, e.g. `5:4`, or a decimal).
  Changing the aspect refits the current box around its centre.
- **Output**: "Crop size" exports the box at its native pixel size. "Scale to"
  lets you type an output width and height; the box is then locked to that
  shape. With an aspect preset chosen, editing one dimension fills in the other.
- Rotating the video rotates the box with it (and flips the aspect, so 16:9
  becomes 9:16).
- Crop and scale need re-encoding, so the export dialog only offers NVENC
  while the crop is on. Frame PNGs are cropped (and scaled) too.
- The filter order on export is rotate, crop, scale, then frame-rate change.

## Match colour to a reference photo

Stop on a frame that shows the person, press **B** (or "Match colour…" on
the crop row) and choose a reference photo of the same person (JPEG, PNG,
HEIC and most other formats). Drag a box over the person's skin in each
image, face or cheek works best. Inside each box only skin-coloured pixels are
used (shown with a green tint), so hair, eyes, teeth and background do not
skew the result; untick the option to use every pixel in the box instead. No
box means the whole image. The Result panel shows the corrected frame live and
the line below shows the skin averages before, in the reference, and after.

The correction is a per-channel exposure and colour balance plus a contrast
(gamma) term, fitted in linear light so the video's skin matches the
reference's average skin colour and its spread of brightness. "Match
contrast" turns the gamma term off and keeps only the balance, and
**Strength** blends between the original and the full correction.

Apply makes the correction part of the view: the preview, frame PNGs and every
export use it. It is implemented as three lookup tables, which ffmpeg applies
as a 1D LUT, so the export matches what you see. Lossless export is
unavailable while it is active. "Clear" removes it, and opening another
video starts clean. The last reference photo is remembered.

`samples/skin_cast.mp4` is a deliberately blue, under-exposed version of
`samples/skin_reference.png`; matching the two recovers the original skin tone.

## Fixed-length clips

Press **L** (or the Fixed length button) to pick a clip by length instead of
by In/Out marks. Type the length in **seconds** (to 0.01 s) or in **frames**;
the other box updates to match. Whichever you typed last is what gets exported
exactly. A length longer than the video is clamped to the video.

While fixed length is on:

- A red box on the timeline shows the selection. Drag it to move the clip,
  or click anywhere outside it to jump the clip start there. Clicking inside
  the box seeks within the clip. Shift-drag scrubs.
- Play, next/previous frame, next/previous keyframe, Home and End all stay
  inside the selection and stop at its last frame. Pressing play at the end
  restarts from the selection start.
- Mark In moves the selection to start at the current frame; Mark Out moves it
  to end just before the current frame.
- A **Move clip** row appears with six buttons that mirror the playback row
  but move the selection box instead of the playhead: to the start of the
  video, back one keyframe, back one frame, forward one frame, forward one
  keyframe, to the end of the video. The shortcuts are the playback ones with
  Ctrl held: Ctrl+Home, Ctrl+Shift+Left, Ctrl+Left, Ctrl+Right,
  Ctrl+Shift+Right, Ctrl+End. The playhead follows to the new clip start.
- The same row has **Start at playhead**, which moves the clip so it begins
  at the frame you are looking at (the I key does the same in this mode).
- **Next clip** (N) moves the selection to begin on the first frame the
  current clip does not include, so exporting, pressing N, and exporting again
  gives back-to-back clips with no overlap and no gap. When there is no room
  for another full clip it moves to the end and says so in the status bar.
- **Playhead to clip start** (Home) jumps the playhead back to the first
  frame of the selection without moving the selection.
- Export uses the selection. Lossless is still offered if both ends happen to
  land on keyframes (and no crop is active); otherwise NVENC.
- Crop works at the same time.

Turning fixed length off restores your In/Out marks.

## Settings

The **Settings** menu sets the default output folder used for both clips and
frame PNGs. "Save next to source video" clears it (the default). "Open output
folder" opens the folder in Explorer. The choice is remembered between runs.

**Settings → Preferences…** (Ctrl+,) opens a window with four tabs:

- **Export**: codec, CQ, preset, audio handling and frame rate the Export
  dialog opens with, and whether it pre-selects Lossless when the In and Out
  points allow it.
- **Quick export**: by default Quick export uses the same settings as Export.
  Untick "Use the same settings as Export" to give it its own codec, CQ,
  preset, audio and frame rate. "Cut losslessly … when the clip allows it"
  makes Quick export stream-copy instead of re-encoding when the selection is
  keyframe-aligned and no crop, flip or colour match is active.
- **Crop & clip**: the aspect ratio the crop box starts locked to, and the
  default fixed clip length in seconds or frames.
- **Output**: save next to the source video or to a chosen folder.

OK saves everything and applies it to the current window straight away.
"Restore built-in defaults" resets the first three tabs to the shipped values
(H.264, CQ 18, p5, copy audio, 24 fps, Quick export same as Export, Free
aspect, 2.00 s) and leaves the output folder alone.

## Marks and export

- **In** is the first frame of the clip.
- **Out** is *exclusive*: the clip ends just before the Out frame. This lets
  you go to a keyframe, press O, and get a clip that ends cleanly at that GOP
  boundary. Pressing O on the very last frame includes it (Out = end of video).
- If In and Out are both keyframes (or the start/end of the file), the export
  dialog offers **Lossless**: all streams are stream-copied with no
  re-encoding, into the same container type as the source.
- Otherwise the export re-encodes with **NVENC** (h264, hevc, or av1) to MP4.
  Cut points are frame-accurate. Audio is copied when MP4-compatible,
  otherwise converted to AAC. Quality is controlled by the CQ value
  (lower = better; default 18, 18–24 is typical).
- The NVENC options also include a **Frame rate** box, which defaults to
  **24**. "Same as source" keeps the original timing; picking or typing a rate
  (24, 30, 60, 24000/1001, 48, ...) resamples the clip by duplicating or
  dropping frames. NTSC presets use exact fractions, so 23.976 is really
  24000/1001. Audio is unaffected.

### Quick export

**Quick export** (Ctrl+Shift+E) skips the dialog. It exports the current
selection (In/Out marks or the fixed-length box), or the **whole video** when
nothing is selected, straight to the output folder using the default NVENC
settings from Preferences (out of the box: H.264, CQ 18, preset p5, 24 fps,
audio copied when MP4-compatible).
The current rotation, flip, crop and scale are applied. It re-encodes unless the
Quick export tab in Preferences allows lossless cuts and the clip qualifies. Progress and a Cancel button appear in the status bar, the file
is named like a normal export (`_clip_full` for the whole video), and an
existing name gets a numeric suffix rather than being overwritten.

Note on lossless cuts: audio packets do not line up exactly with video
keyframes, so a lossless clip may begin with a few milliseconds of audio
before the first video frame. A/V sync is preserved.

NVENC exports write a keyframe every second so the clips step and cut cleanly
when opened again. They seek the source to the keyframe before the cut and
trim precisely on the output side, so a copied audio track contains only the
clip's audio. (Trimming on the input side alone keeps every audio packet from
the keyframe onward, hidden by an edit list, and players such as MPC-HC and
Windows Explorer then report the clip as far longer than it is.) If you have
clips exported before this fix, this repairs one without touching the video:

    ffmpeg -i clip.mp4 -map 0 -c:v copy -c:a aac -b:a 192k -movflags +faststart clip_fixed.mp4

## Layout

- `simplecut/probe.py` — ffprobe wrapper, keyframe list
- `simplecut/decoder.py` — PyAV frame-accurate decode/seek/step
- `simplecut/timeline.py` — timeline widget (marks, fixed-length box)
- `simplecut/cropview.py` — video preview with interactive crop box
- `simplecut/export.py` — ffmpeg command builder + async job
- `simplecut/app.py` — main window and export dialog
- `simplecut/defaults.py` — stored preferences and the Preferences window
- `simplecut/theme.py` — RulZ name, logo assets and the black-and-red theme
- `simplecut/assets/` — logo, icon and theme images
- `packaging/` — Windows release build script and the notices shipped in the zip
- `simplecut/colormatch.py` — skin-tone matching maths and dialog

## License

The source code is released under the [MIT License](LICENSE).

The RulZ name and logo are excluded from the MIT License. All rights are
reserved. The logo files are `simplecut/assets/logo_dark.png`,
`simplecut/assets/logo_light.png` and `simplecut/assets/icon256.png`. They
may not be reused in other projects. If you fork or redistribute this
project, remove or replace the RulZ branding. See
[simplecut/assets/LOGO-NOTICE.md](simplecut/assets/LOGO-NOTICE.md).
