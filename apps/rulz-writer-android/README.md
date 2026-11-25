# RulZ Writer for Android

Write in the Writing mode of [RulZ LLM Suite](../rulz-llm-suite) from your Android phone: open a project, read and edit
scenes, let the AI continue, rewrite or draft, keep your story bible, tracked state and consistency checks, and
interview your characters. The AI and your files stay on the computer; the phone is the screen.

The phone reaches the computer over [Tailscale](https://tailscale.com), so it works from anywhere without opening
anything to the internet. It needs Android 7.0 or later.

Design and the message contract: `apps/rulz-llm-suite/docs/phone-access-design.md` and
`apps/rulz-llm-suite/src/shared/phone.ts`.

## Getting started

1. Install Tailscale on the computer and on the phone, and sign in to the same account on both.
2. On the computer, update RulZ LLM Suite to **0.26.0** or later, open **Settings**, and under **Phone access** turn on
   **Let my phone connect**. The first time, Windows may ask whether RulZ LLM Suite may use your networks: allow it.
   RulZ LLM Suite has to be running on the computer; its window may be in the tray.
3. Install this app on the phone: scan the **Phone app** QR code on LLM Suite's page in the RulZ AI Tools Suite
   launcher, or download the APK (`RulZ-Writer-<version>.apk`) from the suite's
   [Releases](https://github.com/speedyrulz/RulZ-AI-Tools-Suite/releases) page and open it (allow installing from that
   source). Or build it yourself, see *Building* below.
4. Scan the QR code from the computer's Settings with the phone's camera and tap the link, or tap **Add computer**, then
   **No camera? Type it in instead**, and paste the pairing link or type the address, port (7421) and pairing code shown
   next to the QR code.

The app remembers the computer and reconnects by itself, also after the phone sleeps or switches networks (more slowly
while the app is in the background, to save battery). After a wrong pairing code it doesn't try again by itself, since
every try counts towards the computer's limit; **Try again** on the banner does. The app's data (the computer's address
and pairing code) is kept in the app's private storage and left out of Android backups.

## What you can do

**Adventures** lets you play branching stories on your computer's model. Start with an idea or a Writing project,
choose a path or type an action, and use the map to return to earlier passages. Export shares the current path.

- **Projects**: the projects in the computer's Writing folder; open one or start a new one.
- **Write**: pick a scene from **Chapters and scenes** and write in it. Your text is saved on the computer about a
  second after you stop typing, with the same safety rules as the desktop: a save made on top of a file that changed
  on the computer is a conflict, and you see the computer's text next to yours and choose which to keep. If the file
  changes on the computer while you have unsaved edits, a banner says so. Text that couldn't be saved yet stays on the
  phone and is offered back after a restart.
- **Chapters and scenes**: status, rename, move up and down, add, delete (deleted scenes and chapters go to the
  project's trash on the computer; **Trash** restores them), edit a scene's beats.
- **Write with AI**: Continue, Draft scene, Dialogue, Expand, Condense and Rewrite (tighter, more vivid, change tone,
  show don't tell, your own way), with a note and one to four alternatives. Expand, Condense and Rewrite work on the
  passage you select. A job card shows the progress and the text as it is written, with **Stop**. A result shows the
  alternatives and what each changes; **Accept** puts it in the scene and saves it, **Reject** forgets it. If you
  changed the scene after starting, Accept says "The text changed since this was written." instead of overwriting.
  Results are kept on the computer until you accept or reject them, so they are there again after the phone slept.
- **Summaries and outline proposals**: summarize a scene or chapter, plan chapters, plan the scenes of a chapter;
  accept them into the outline. Names the AI used that aren't in the story bible are listed.
- **Bible**: entries by kind; open one to edit its file (the front matter is part of the text and is saved as it is),
  add one, delete one.
- **State**: what is true at the current scene; **Update state** lets the AI propose changes to tick and apply; add,
  end or remove facts by hand.
- **Check**: the consistency check of the scene; tap **Show in the text** to select the passage, or dismiss a finding.
- **Interview**: talk to a character; **Add to voice notes** adds a line to the character's Voice section.
- **The model's settings** (0.3.0, RulZ LLM Suite 0.35.0): a line under the title names the model that writes and which
  of its Writing (or Adventure) settings apply. When it runs with other load settings, **Reload** loads it again with
  them; **Keep as is** hides the bar. Nothing reloads by itself, and the settings are changed on the computer.
- **Models on a rented computer** (vast.ai) are marked **Rental**; your text is sent there when you choose one. They are
  loaded on the computer only.
- **Settings**: the model of the open project; notifications when a job finishes while the app is in the background,
  **Stay connected in the background** (Android shows a small ongoing notification for it); this phone's look and name;
  switch or forget computers.

## Limits

- Proposals made on the phone show only on the phone; the computer sees the text once you accept it.
- If the computer's window has unsaved edits in a scene and you save that scene from the phone, the window shows its
  usual "changed on disk" warning.
- One AI job at a time from the phone, and not while the computer's window is generating on the same model.
- LLM Suite must be running on the computer.
- Desktop only: the writing benchmark, recording and exporting training data, deleting projects, emptying the trash,
  the file history.

## Security

- The computer listens only on its Tailscale address (100.x.y.z), never on your home network or the internet. Tailscale
  encrypts the connection (WireGuard), and only devices in your tailnet can reach it.
- Every connection must start with the pairing code. After several wrong codes from one device, the computer stops
  listening to it for a while. **Make a new code** on the computer disconnects every phone using the old one.
- The computer answers only the Writing requests listed in the contract; the chat, tools and settings of RulZ LLM Suite
  are not reachable from the phone.
- The pairing code is kept in the app's private storage on the phone.

## How it works

With phone access on, the computer serves a list of typed requests and events over a WebSocket at
`ws://<tailscale-ip>:7421/rulz-writer`. This app sends them (`src/lib/remote.ts`). The messages:

```
phone → computer   { t: 'hello', token, device, protocol }       first, within a few seconds
                   { t: 'invoke', id, channel, args }
computer → phone   { t: 'welcome', protocol, version, name }  |  { t: 'denied', reason }
                   { t: 'result', id, ok, value | error }
                   { t: 'event', channel, payload }
```

`src/shared/` is copied from the desktop app (`npm run sync-shared`, which looks for it next to this folder, in
`apps/rulz-llm-suite`): the contract `phone.ts` and the files it uses. The logic that matters for safety is in
`src/lib/` as plain functions with tests: saving and conflicts (`autosave.ts`), the accept check (`accept.ts`), the job
list (`jobs.ts`), outline and state operations (`outlineOps.ts`, `stateOps.ts`).

The app is React, Tailwind and zustand in a [Capacitor](https://capacitorjs.com) shell, like RulZ Coding Agent for
Android.

## Building

Needs Node.js 22+, Java 21 and the Android SDK (platform 36 and build-tools; Android Studio installs them).

```
npm install
npm run android:apk        # builds the web part, syncs it into android/, and runs Gradle
```

The APK is `android/app/build/outputs/apk/debug/app-debug.apk`. Copy it to the phone and open it (allow installing from
that source), or with the phone plugged in and USB debugging on: `adb install -r <path to the apk>`.

Set `ANDROID_HOME` to the SDK folder, or create `android/local.properties` with
`sdk.dir=C:\\Users\\you\\AppData\\Local\\Android\\Sdk`. To work on it in Android Studio: `npx cap open android`.

## Releasing

Bump `version` in `package.json` and `versionName` (and `versionCode`, by one) in `android/app/build.gradle`; the two
must agree. Pushing the tag `writer-android-v<version>` makes the suite's Apps workflow build the APK and publish it
as `RulZ-Writer-<version>.apk` on a release; the launcher's QR code then points to it.

## Development

```
npm run dev          # the app in a desktop browser at http://127.0.0.1:5199
npm test             # unit tests (pairing links, the connection, saving, accepting, jobs, outline and state)
npm run typecheck
```

In a browser, saved computers go to `localStorage`, and a pairing link can be given as `?pair=<link>`.

## Later

Not in this version: the file history and restore, a chapter-wide consistency check, notifications after Android stops
the app (push notifications would need a cloud service), a built-in QR scanner (the phone's camera app does it),
several computers at once.

## License

MIT, except the RulZ name and logo; see [LOGO-NOTICE.md](LOGO-NOTICE.md).
