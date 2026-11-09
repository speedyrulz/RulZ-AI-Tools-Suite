# RulZ Coding Agent for Android

Watch and control [RulZ Coding Agent](https://github.com/speedyrulz/rulz-coding-agent) on your computer from your
Android phone. Give it tasks, answer its "Is this OK?" questions, follow its steps, look at the files it changed,
run your app, undo, all from the phone, while the AI and your projects stay on the computer.

The phone reaches the computer over [Tailscale](https://tailscale.com), so it works from anywhere without opening
anything to the internet.

## Getting started

1. Install Tailscale on the computer and on the phone, and sign in to the same account on both.
2. On the computer, update RulZ Coding Agent to **0.6.0** or later, open **Settings** and turn on **Phone access**.
   The first time, Windows may ask whether RulZ Coding Agent may use your networks: allow it.
3. Install this app on the phone (`app-debug.apk`, see *Building* below).
4. Scan the QR code from the computer's Settings with the phone's camera and tap the link, or tap **No camera? Type it
   in instead** and type the address, port and pairing code shown under the QR code.

The app remembers the computer and reconnects by itself, also after the phone sleeps or switches networks.

## What you can do

- **Projects**: the project open on the computer shows at the top; open another, create one, or browse the
  computer's folders with **Open folder**. The computer has one open project at a time, so opening one here opens it
  there too, and the computer's window follows.
- **Chat**: what you asked and what the agent says, live. Type the next task and press the arrow; the red square stops
  the task. Questions from the agent and "I'm stuck" pauses show here as on the computer. The model's thoughts show
  live in a **Thinking…** box above its reply, which folds away into **Thought process** when the answer starts (tap it
  to open or close); turn on **Keep thinking open after the answer** in Settings to leave it open. When thinking is off
  or its budget is 0 in Model settings, the status says so instead. Thoughts need RulZ Coding Agent 0.10.0 on the
  computer, and keeping them open needs 0.11.0.
- **Is this OK?**: approval requests come up as a sheet over whatever you're looking at. **Allow**, **Allow for this
  task** or **Don't allow**; the back button leaves the app rather than refusing by accident.
- **Steps** and **Files**: each step with its command and output, and each changed file as a diff. When the task used
  a draft model (MTP), **Drafts** above the steps shows how many drafted tokens the model accepted and the time that
  saved; tap it for the counts (needs RulZ Coding Agent 0.10.0).
- **App**: **Run my app** on the computer and watch its output; **Fix this** when it crashes. For web apps, **Open on
  this phone** opens the address with the computer's Tailscale address (the dev server must accept outside visits:
  `--host` for Vite).
- **⋮ menu**: Share my app, Undo last task, History (go back to before any task), Notes, Settings.
- **Notifications** (Settings > Notifications): when a task finishes (with the agent's last words), and when it waits
  for you: a permission request, a question or a pause. Tap one to go to the project. They come only while the app
  isn't on screen, and a waiting notification disappears once it's answered, here or on the computer. To get them
  while the app is closed, **Stay connected in the background** keeps the connection open, which Android shows as a
  small ongoing notification that says what the agent is doing (working, waiting for your OK, a question, paused);
  while a task runs, the phone also stays awake so messages arrive right away. If they stop after the phone has been
  locked a while, use **Battery settings** to let the app run without battery limits. Settings explains why
  notifications can't show, if Android blocks them, and shows the last one sent.
- **Model quick switch**: tap the model under the project name to switch models (not during a task), with a shortcut to
  Model settings.
- **Model settings** (Settings > AI model): the same as on the computer. **How it writes** covers temperature, top P,
  top K, repeat penalty, longest answer and thinking (with a warning that a budget of 0 skips thinking), saved per model. **How it loads** covers context length, GPU
  layers, flash attention, KV cache, GPU split sliders, the draft model (automatic, built-in MTP layers or any draft
  file in RulZ's models folder) and draft (MTP) tuning, with RulZ's memory estimate as you change them; saving
  restarts a loaded model. This needs RulZ Coding Agent 0.8.0 on the computer; the GPU split, draft tuning and
  estimate need RulZ LLM Suite 0.3.7, and the draft model choice needs RulZ Coding Agent 0.9.2 and RulZ LLM Suite 0.3.9.
- **Reverse engineering** (⋮ menu, and a card on a new project's first screen): start the tasks RulZ LLM Suite's
  Reverse Engineering page offers, run on the computer in the open project: analyse a program or folder with the
  Suite's question templates, clone pieces of it or all of it as new code, or start one of the training kit's
  **Practice** tasks (with how often each was solved; the answer is checked when the agent finishes). **Done** lists
  every RE session with its grade. **Record for training** records them in RulZ LLM Suite. Needs RulZ Coding Agent
  0.18.0 and RulZ LLM Suite 0.15.0 on the computer.
- **Settings**: the AI model, keeping thinking open, approval level, instructions and extra tools (these change the
  computer's settings);
  this phone's look and name; switch or forget computers. The RulZ LLM Suite address and API key, and phone access
  itself, can only be changed on the computer.

## Security

- The computer listens only on its Tailscale address (100.x.y.z), never on your home network or the internet.
  Tailscale encrypts the connection (WireGuard), and only devices in your tailnet can reach it.
- Every connection must start with the pairing code. After 10 wrong codes from one device, the computer stops
  listening to it for 10 minutes. **Make a new code** on the computer disconnects every phone using the old one.
- The phone never receives the pairing code back, and can't change phone access, open dialogs on the computer, open
  its browser, or show files in Explorer.
- The pairing code is kept in the app's private storage on the phone.

## How it works

The computer app's window talks to its main process through a list of typed requests and events
(`src/shared/ipc.ts`). With phone access on, the computer also serves those same requests over a WebSocket at
`ws://<tailscale-ip>:7420/rulz` (`src/main/remote/` in the desktop app). This app sends them over that socket instead,
so the phone and the computer's window see exactly the same state and events. The messages:

```
phone → computer   { t: 'hello', token, device, protocol }       first, within 10 s
                   { t: 'invoke', id, channel, args }
computer → phone   { t: 'welcome', protocol, version, name }  |  { t: 'denied', reason }
                   { t: 'result', id, ok, value | error }
                   { t: 'event', channel, payload }
```

`src/shared/` is copied from the desktop app (`npm run sync-shared`, which looks for it in
`../RulZ AI Tools Suite/apps/rulz-coding-agent`); `protocol` tells the two apps apart when they
don't match.

The app is React, Tailwind and zustand (like the desktop app) in a [Capacitor](https://capacitorjs.com) shell.

## Building

Needs Node.js 22+, Java 21 and the Android SDK (platform 36 and build-tools; Android Studio installs them).

```
npm install
npm run android:apk        # builds the web part, syncs it into android/, and runs Gradle
```

The APK is `android/app/build/outputs/apk/debug/app-debug.apk`. Copy it to the phone and open it (allow installing
from that source), or with the phone plugged in and USB debugging on: `adb install -r <path to the apk>`.

Set `ANDROID_HOME` to the SDK folder, or create `android/local.properties` with `sdk.dir=C:\\Users\\you\\AppData\\Local\\Android\\Sdk`.
To work on it in Android Studio: `npx cap open android`.

## Development

```
npm run dev          # the app in a desktop browser at http://127.0.0.1:5199
npm test             # unit tests (pairing links, the connection)
npm run typecheck
```

In a browser, saved computers go to `localStorage`, and a pairing link can be given as `?pair=<link>`. To try it
against a computer app on the same machine, start RulZ Coding Agent with `RULZ_AGENT_REMOTE_HOST=127.0.0.1` so phone
access listens there, and add the computer as `127.0.0.1`.

## Later

Not in this version: answering approvals from the notification itself, notifications after Android stops the app
(push notifications would need a cloud service), a built-in QR scanner (the phone's camera app does it), several
computers at once.

## License

MIT, except the RulZ name and logo; see [LOGO-NOTICE.md](LOGO-NOTICE.md).
