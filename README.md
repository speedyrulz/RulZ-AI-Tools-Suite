# RulZ Image Tournament

A Windows app that finds your favorite image by single-elimination tournament. It shows two images at a time. You keep one, and the other is moved to a Losers folder (or recycled, deleted, or left alone). This repeats until one champion is left.

![RulZ](assets/logo-light.png)

## Run

Double-click **`launch.bat`**. It builds the app first if needed. You can also drop images or folders onto `launch.bat` to start a tournament with them.

The app needs nothing beyond Windows 10/11 (.NET Framework 4.8 is built in). You can also run `bin\RulZ Image Tournament.exe` directly, or pass images and folders to it on the command line.

## Install (optional)

Double-click **`install.bat`** to install the app for your user account. It doesn't need admin rights. It:

- builds the app and copies it to `%LOCALAPPDATA%\Programs\RulZ Image Tournament`
- adds Start Menu and Desktop shortcuts
- adds a **Send to** shortcut, so you can right-click images or a folder and choose *Send to → RulZ Image Tournament*

Run `install.bat` again to update the installed copy after changing the code. `uninstall.bat` removes the app and shortcuts, and asks before deleting your settings. It never touches your images or Losers folders.

## Build

```powershell
.\build.ps1          # builds bin\RulZ Image Tournament.exe
.\build.ps1 -Test    # also runs the bracket logic tests
```

It uses the C# compiler that ships with Windows, so no SDK or Visual Studio is needed. The logo is embedded in the `.exe`, so it remains a single file.

`tools\make_logo.py` (needs Pillow) builds the app's artwork from `assets\logo-source.webp`. It makes a transparent wordmark with white lettering for the dark UI (`logo.png`), a light-background version (`logo-light.png`), and the "R" icon (`app.ico`). Rerun it, then `build.ps1`, if the logo changes.

## How the bracket works

Images are paired in order: 1 v 2, 3 v 4, 5 v 6… When every match in a round is done, the winners meet in the same order in the next round.

If a round has an odd number of images, the last one advances automatically (a bye). In the next round, the image that got the bye goes **first**, so it faces the winner of 1 v 2. The image left over at the end of that round gets the next bye, which keeps the same image from getting a free pass every round.

Example with 7 images: round 1 is 1v2, 3v4, 5v6, and 7 gets the bye. Round 2 is 7 v W(1,2) and W(3,4) v W(5,6). Round 3 is the final.

Images you add during a tournament join the end of the current round. Images you add after a champion is decided challenge the champion.

## Controls

| Action | Mouse / button | Keyboard |
|---|---|---|
| Keep left image | **Keep Left** | `←`, `A`, `1` |
| Keep right image | **Keep Right** | `→`, `D`, `2` |
| Undo last pick | **Undo** | `Ctrl+Z`, `Backspace` |
| Zoom (both images) | Mouse wheel (zooms toward the cursor) | `+` / `-` |
| Pan (both images) | Click and drag | |
| Reset zoom | Double-click, **Fit** | `F`, `0`, `Home` |
| Side by side / split | **Side by Side**, **Split** | `V` or `S` toggles |
| Move the split line | Drag the gold divider | |
| Add images / folder | **Add Images…**, **Add Folder…**, or drag and drop | `Ctrl+O`, `Ctrl+Shift+O` |

Zoom and pan stay in sync. Both images show the same spot measured as a fraction of each image's size from its center, so images of different resolutions still line up. Past 300% zoom, true pixels are shown without smoothing.

## Options

- **Loser:** sets what happens to the image you didn't pick:
  - *Move to Losers folder* (default). Leave the folder box empty to use a `Losers` folder next to each image. An absolute path sends every loser to one folder. A relative name, such as `Rejects`, is created next to each image. A file with the same name already there is never overwritten: the loser is saved as `name (2).jpg`.
  - *Send to Recycle Bin*
  - *Delete permanently* (asks you to confirm first)
  - *Leave in place*, if you only want to find a favorite
- **Undo** works for *Move* and *Leave in place*, and moves the file back. It is disabled after recycling or deleting.
- **Subfolders** scans folders recursively and skips the Losers folder.
- **Shuffle** randomizes the order of newly added images.
- **Keep zoom** keeps the zoom and position from one match to the next. This is handy for comparing the same detail across a burst of similar shots.

Progress is saved after every pick. If the app is closed mid-tournament, it offers to resume the next time it starts. Settings and progress are stored in `%APPDATA%\RulZ Image Tournament`.

## Supported formats

JPEG, PNG, BMP, GIF (first frame), TIFF, WebP, ICO, and JPEG XR. HEIC/HEIF and AVIF also work when the matching Windows codec extensions are installed. Camera rotation (EXIF orientation) is applied automatically.
