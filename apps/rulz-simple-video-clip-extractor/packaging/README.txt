RulZ Simple Video Clip Extractor  -  Windows build
==================================================

Run "RulZ Clip Extractor.exe". No Python installation is needed.

Requirements
------------
- Windows 10 or 11, 64-bit.
- ffmpeg and ffprobe on your PATH. They are used for probing videos and for
  every export. Install with:   winget install Gyan.FFmpeg
  then open a new window so the PATH change takes effect.
- An NVIDIA GPU with current drivers for NVENC (re-encoded) exports.
  Lossless cuts and frame PNGs work without one.

You can also drag a video file onto the .exe, or onto the app window.

Keep the "_internal" folder next to the .exe; the app needs it.

Dark or light look: Settings > Light mode. The first launch asks once.

Settings are stored per user in the Windows registry under
HKEY_CURRENT_USER\Software\SimpleCut. Delete that key to reset everything.

Full documentation, keyboard shortcuts and source code:
https://github.com/speedyrulz/RulZ-AI-Tools-Suite/tree/main/apps/rulz-simple-video-clip-extractor

Licenses: see LICENSE.txt, LOGO-NOTICE.md and THIRD-PARTY-NOTICES.txt.
