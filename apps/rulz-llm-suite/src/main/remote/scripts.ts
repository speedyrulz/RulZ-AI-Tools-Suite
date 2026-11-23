// The bash scripts sent to a rental (docs/vast-design.md, sections 2.5, 4-7). Pure functions: text in, text out.
// Each is sent on stdin to `bash -s` (see frameScript in ssh.ts) and reports with `@rz KEY=VALUE` marker lines, so
// banners or noise printed by the image's shell cannot be mistaken for an answer. Scripts exit with codes under 100
// (255 is ssh's own). Every value that comes from TypeScript goes through shq() or is checked to be a whole number.
import type { RemoteGpu } from '@shared/remote'
import { assertFileName, assertInstanceId, assertSha256, assertSlug, assertTag, marker, markers, shq } from './ssh'

/**
 * Tagged template for bash text: raw (backslashes stay as written) and `@{` stands for `${` so TypeScript does not
 * read bash parameter expansions as its own. Interpolated values must already be quoted or numeric.
 */
function bash(strings: TemplateStringsArray, ...vals: Array<string | number>): string {
  return String.raw(strings, ...vals)
    .split('@{')
    .join('${')
}

/** Drops comment lines, blank lines and indentation (the onstart script has a 4000 character limit). */
export function compactBash(text: string): string {
  return text
    .split('\n')
    .map((l) => l.trim())
    .filter((l) => l && !l.startsWith('#'))
    .join('\n')
}

/** In an argument list, stands for the port the instance picks. */
export const PORT_PLACEHOLDER = '@RZ_PORT@'
/** llama-server is ended by its supervisor when keepalive is older than this. */
export const SERVER_MAX_IDLE_SECONDS = 6 * 3600

/** The remote folder: /workspace/rulz-llm, or /root/rulz-llm when there is no /workspace. RZ_ROOT is for tests. */
const PRELUDE = bash`
emit(){ printf '@rz %s=%s\n' "$1" "$2"; }
R=@{RZ_ROOT:-}
[ -n "$R" ] || { R=/workspace/rulz-llm; [ -d /workspace ] || R=/root/rulz-llm; }
mkdir -p "$R"
r=$(cd "$R" 2>/dev/null && pwd -P) && [ -n "$r" ] && R=$r
`

/** A variable from the environment, /etc/environment or /proc/1/environ (vast writes the container's variables to all three). */
const ENV_FN = bash`
ev(){ local v=@{!1:-}; [ -n "$v" ] || v=$(sed -n "s/^\(export \)\{0,1\}$1=//p" /etc/environment 2>/dev/null | head -1 | tr -d "\"'"); [ -n "$v" ] || v=$(tr '\0' '\n' </proc/1/environ 2>/dev/null | sed -n "s/^$1=//p" | head -1); printf '%s' "$v"; }
vid(){ local i; i=$(ev CONTAINER_ID); i=@{i#C.}; case "$i" in ''|*[!0-9]*) i=$(ev VAST_CONTAINERLABEL); i=@{i#C.};; esac; case "$i" in ''|*[!0-9]*) return 1;; esac; printf '%s' "$i"; }
vapi(){ local k i x=(); k=$(ev CONTAINER_API_KEY); i=$(vid) || { echo none; return; }; [ -n "$k" ] || { echo nokey; return; }
[ "$1" = PUT ] && x=(-X PUT -H 'Content-Type: application/json' -d '{"state":"stopped"}')
printf 'header = "Authorization: Bearer %s"\n' "$k" | curl -sS -m 30 --config - "@{x[@]}" -o /dev/null -w '%{http_code}' "https://console.vast.ai/api/v0/instances/$i/" 2>/dev/null; }
`

/**
 * Prints 1 when a compute process that is not this app's llama-server may be on a GPU, else 0. The safe side for
 * stopping is "foreign" whenever unsure, so:
 * - a pid nvidia-smi lists that is visible here and is not one of our servers (its /proc/<pid>/exe is not under our
 *   engine folder) is foreign;
 * - inside a container nvidia-smi may list host pids that do not exist here, or none at all. Such a pid counts only as
 *   far as we run servers that were not already matched by pid (our own loading server is never foreign);
 * - GPU memory used above what our live servers were expected to take (run/<id>/expect, MiB, plus 12% and 1.5 GiB)
 *   is foreign, which also covers processes in other containers. A server with an unknown expectation skips this test;
 * - nvidia-smi failing counts as foreign.
 */
const FOREIGN_FN = bash`
rzf(){ local f p x a=0 m=0 n=0 ex=0 unk=0 s used o=" "
for f in "$R"/run/*/pid; do [ -f "$f" ] || continue; p=$(cat "$f" 2>/dev/null)
case "$(readlink "/proc/$p/exe" 2>/dev/null)" in "$R"/engine/*) o="$o$p "; n=$((n+1)); x=$(cat "@{f%pid}expect" 2>/dev/null); case "$x" in ''|*[!0-9]*) unk=1;; *) ex=$((ex+x));; esac;; esac; done
s=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null) || { echo 1; return; }
while read -r p; do [ -n "$p" ] || continue; a=$((a+1))
case "$o" in *" $p "*) m=$((m+1));; *) [ -e "/proc/$p" ] && { echo 1; return; };; esac; done <<<"$s"
[ $((a-m)) -gt $((n-m)) ] && { echo 1; return; }
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '{t+=$1}END{print t+0}')
[ "$unk" = 0 ] && [ "$used" -gt $((ex+ex/8+1536)) ] && { echo 1; return; }
echo 0; }
`

/** Emits ALIVE, PID, PORT, HASH of the server run/<id> (alive = the pid exists and is a binary of our engine). */
const STATUS_FN = bash`
rzs(){ local D="$R/run/$1" p; p=$(cat "$D/pid" 2>/dev/null); local al=0
if [ -n "$p" ] && kill -0 "$p" 2>/dev/null && ! grep -q '^State:[[:space:]]*Z' "/proc/$p/status" 2>/dev/null; then case "$(readlink "/proc/$p/exe" 2>/dev/null)" in "$R"/engine/*) al=1;; esac; fi
emit ALIVE $al; emit PID "$p"; emit PORT "$(cat "$D/port" 2>/dev/null)"; emit HASH "$(cat "$D/hash" 2>/dev/null)"; }
`

const GPU_FN = bash`
rzg(){ command -v nvidia-smi >/dev/null 2>&1 || { emit NVSMI 0; return; }
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | sed 's/ *, */|/g' | while IFS= read -r l; do emit GPU "$l"; done
emit NVSMI 1; }
`

// ---------- Watchdog (design 2.5) ----------

/**
 * Stops the instance through the account API when its limits say so. Limits are in `$R/limits` (idle minutes, cap as
 * an epoch second, origin), each 0 = off. Every 30 s it samples the busiest GPU; a stop needs keepalive older than
 * the idle limit and 10 low samples in a row (about 5 minutes), or the cap passed, and for idle always, and for the cap
 * on an attached instance, no foreign compute process (rzf). Retries once a minute while the API refuses. The key is
 * only in a curl config on stdin and never logged. RZ_TICK, RZ_RETRY and RZ_N exist for tests.
 */
const WATCHDOG_BODY = bash`
R=@{RZ_ROOT:-}
[ -n "$R" ] || { R=/workspace/rulz-llm; [ -d /workspace ] || R=/root/rulz-llm; }
r=$(cd "$R" 2>/dev/null && pwd -P) && [ -n "$r" ] && R=$r
mkdir -p "$R"
exec >>"$R/watchdog.log" 2>&1
pf=$R/watchdog.pid
o=$(cat "$pf" 2>/dev/null)
if [ -n "$o" ] && kill -0 "$o" 2>/dev/null && grep -qa watchdog "/proc/$o/cmdline" 2>/dev/null; then exit 0; fi
echo $$ >"$pf"
${ENV_FN}
${FOREIGN_FN}
say(){ echo "$(date -u +%FT%TZ) $*"; }
low=0; n=0; ls=0
while :; do
idle=0; cap=0; org=app
. "$R/limits" 2>/dev/null
now=$(date +%s)
ka=$(stat -c %Y "$R/keepalive" 2>/dev/null || echo "$now")
u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | sort -n | tail -1)
if [ "@{u:-100}" -le 10 ] 2>/dev/null; then low=$((low+1)); else low=0; fi
why=
if [ "$cap" -gt 0 ] && [ "$now" -ge "$cap" ]; then why=cap
elif [ "$idle" -gt 0 ] && [ $((now-ka)) -gt $((idle*60)) ] && [ "$low" -ge 10 ]; then why=idle; fi
if [ -n "$why" ] && [ $((now-ls)) -ge "@{RZ_RETRY:-60}" ]; then
ls=$now
if [ "$why$org" != capapp ] && [ "$(rzf)" = 1 ]; then say "$why limit reached, but another GPU process may be running: not stopping"
else say "stopping the instance ($why limit)"; say "stop request: http $(vapi PUT)"; fi
fi
n=$((n+1)); [ -n "@{RZ_N:-}" ] && [ "$n" -ge "$RZ_N" ] && exit 0
sleep "@{RZ_TICK:-30}"
done
`

/** The watchdog program as installed (compact). */
export const WATCHDOG_PROGRAM: string = compactBash(WATCHDOG_BODY)

const intIn = (n: number, min: number, max: number, what: string): number => {
  if (!Number.isInteger(n) || n < min || n > max) throw new Error(`${what} is out of range.`)
  return n
}

export interface WatchdogLimits {
  /** 0 = no idle stop. */
  idleMinutes: number
  /** Epoch second; null = no cap. */
  capAt: number | null
  origin: 'app' | 'attached'
  /** Cap in hours for the next boot (onstart reads it from `boot`); 0 = none. */
  capHours: number
}

const limitsText = (l: WatchdogLimits): string => `idle=${intIn(l.idleMinutes, 0, 10080, 'Idle minutes')}\ncap=${l.capAt === null ? 0 : intIn(l.capAt, 0, 4102444800, 'The cap')}\norg=${l.origin === 'attached' ? 'attached' : 'app'}\n`

/** Writes watchdog.sh when it differs and stops the old process, so the next start runs the new text. */
const INSTALL_WATCHDOG = bash`
cat >"$R/watchdog.sh.new" <<'RZWDOG'
${WATCHDOG_PROGRAM}
RZWDOG
if ! cmp -s "$R/watchdog.sh.new" "$R/watchdog.sh" 2>/dev/null; then
mv "$R/watchdog.sh.new" "$R/watchdog.sh"
o=$(cat "$R/watchdog.pid" 2>/dev/null); if [ -n "$o" ] && grep -qa watchdog "/proc/$o/cmdline" 2>/dev/null; then kill "$o" 2>/dev/null; sleep 1; fi
else rm -f "$R/watchdog.sh.new"; fi
wd(){ local o; o=$(cat "$R/watchdog.pid" 2>/dev/null); [ -n "$o" ] && kill -0 "$o" 2>/dev/null && grep -qa watchdog "/proc/$o/cmdline" 2>/dev/null; }
wd || { setsid nohup bash "$R/watchdog.sh" >/dev/null 2>&1 </dev/null & sleep 1; }
`

/** Writes the limits and (re)starts the watchdog, touches keepalive, then asks the API for its own instance. */
export function armScript(limits: WatchdogLimits): string {
  return (
    bash`
${PRELUDE}
${ENV_FN}
printf '%s' ${shq(limitsText(limits))} >"$R/limits.new" && mv "$R/limits.new" "$R/limits"
printf 'bi=%s\nbh=%s\n' ${intIn(limits.idleMinutes, 0, 10080, 'Idle minutes')} ${intIn(limits.capHours, 0, 24 * 90, 'The cap hours')} >"$R/boot"
touch "$R/keepalive"
${INSTALL_WATCHDOG}
wd && emit WATCHDOG_RUNNING 1 || emit WATCHDOG_RUNNING 0
c=$(vapi GET)
emit VERIFY "$c"
if wd && [ "$c" = 200 ]; then printf 'ok' >"$R/watchdog.state"; emit ARMED 1
else
case "$c" in none) m="This instance has no CONTAINER_ID in its environment.";; nokey) m="This instance has no CONTAINER_API_KEY, so it cannot stop itself.";; 401|403) m="The instance's own API key was refused (http $c).";; *) m="The instance could not reach vast.ai (http $c).";; esac
wd || m="The watchdog did not start."
printf 'error: %s' "$m" >"$R/watchdog.state"; emit ARMED 0; emit ERR "$m"
fi
`
  )
}

/** Turns both limits off (an attached instance); the watchdog process, if any, keeps running and then stops nothing. */
export function disarmScript(): string {
  return bash`
${PRELUDE}
if [ -f "$R/limits" ] || [ -f "$R/watchdog.pid" ]; then printf 'idle=0\ncap=0\norg=attached\n' >"$R/limits"; fi
printf 'bi=0\nbh=0\n' >"$R/boot"
printf 'off' >"$R/watchdog.state"
emit ARMED 0
`
}

/** Only the check that the instance can stop itself. */
export function verifyScript(): string {
  return bash`
${PRELUDE}
${ENV_FN}
c=$(vapi GET); emit VERIFY "$c"
`
}

/**
 * The `onstart` script of a rental made by the app. It runs on every boot (also after a restart from vast's site):
 * writes the limits (the cap counts from this boot; idle minutes and cap hours come from `boot`, which arm and disarm
 * keep current, else from the values the rental was made with), touches keepalive, writes the watchdog and starts it. It never
 * exits and never fails the image's own startup. Under 4000 characters, watchdog included.
 */
export function onstartScript(limits: { idleMinutes: number; capHours: number | null }): string {
  const idle = intIn(limits.idleMinutes, 0, 10080, 'Idle minutes')
  const capH = limits.capHours === null ? 0 : intIn(limits.capHours, 1, 24 * 90, 'The cap hours')
  return (
    compactBash(bash`
R=/workspace/rulz-llm; [ -d /workspace ] || R=/root/rulz-llm
mkdir -p "$R" 2>/dev/null
bi=${idle}; bh=${capH}; [ -f "$R/boot" ] && . "$R/boot" 2>/dev/null
case "$bi" in ''|*[!0-9]*) bi=${idle};; esac
case "$bh" in ''|*[!0-9]*) bh=${capH};; esac
c=0; [ "$bh" -gt 0 ] && c=$(( $(date +%s) + bh * 3600 ))
printf 'idle=%s\ncap=%s\norg=app\n' "$bi" "$c" >"$R/limits" 2>/dev/null
touch "$R/keepalive" 2>/dev/null
cat >"$R/watchdog.sh" <<'RZW'
${WATCHDOG_PROGRAM}
RZW
setsid nohup bash "$R/watchdog.sh" >/dev/null 2>&1 </dev/null &
`) + '\n'
  )
}

// ---------- Keepalive ----------

/** Held open over ssh: touches keepalive now and for every line read; EOF (the connection ends) ends the loop. */
export function keepaliveLoopScript(): string {
  return bash`
${PRELUDE}
K="$R/keepalive"; touch "$K"
while IFS= read -r _; do touch "$K"; done
`
}

export function touchScript(): string {
  return bash`
${PRELUDE}
touch "$R/keepalive"
`
}

// ---------- Probe ----------

export function probeScript(): string {
  return bash`
${PRELUDE}
${GPU_FN}
${FOREIGN_FN}
emit ROOT "$R"
rzg
emit DISK_FREE "$(df -PB1 "$R" 2>/dev/null | awk 'NR==2{print $4}')"
for d in "$R"/engine/*/; do [ -f "$d.ok" ] && emit ENGINE "$(basename "$d")"; done
for d in "$R"/models/*/; do [ -s "$d.complete" ] && emit MODEL "$(basename "$d")"; done
wp=$(cat "$R/watchdog.pid" 2>/dev/null)
if [ -n "$wp" ] && kill -0 "$wp" 2>/dev/null && grep -qa watchdog "/proc/$wp/cmdline" 2>/dev/null; then emit WATCHDOG_RUNNING 1; else emit WATCHDOG_RUNNING 0; fi
emit WATCHDOG_STATE "$(head -c 300 "$R/watchdog.state" 2>/dev/null | tr '\n' ' ')"
emit FOREIGN "$(rzf)"
exit 0
`
}

export function parseGpus(out: string): RemoteGpu[] {
  const gpus: RemoteGpu[] = []
  for (const v of markers(out, 'GPU')) {
    const [i, name, total, used, util] = v.split('|')
    const index = Number(i)
    if (!Number.isInteger(index)) continue
    gpus.push({ index, name: name ?? '', totalMiB: Number(total) || 0, usedMiB: Number(used) || 0, utilization: Number(util) || 0 })
  }
  return gpus
}

export interface ProbeMarkers {
  root: string
  gpus: RemoteGpu[]
  nvidiaSmi: boolean
  diskFreeBytes: number | null
  engines: string[]
  /** Model folder names with a completion marker. */
  modelSlugs: string[]
  watchdogRunning: boolean
  watchdogState: string
  foreign: boolean
}

export function parseProbe(out: string): ProbeMarkers {
  const disk = Number(marker(out, 'DISK_FREE'))
  return {
    root: marker(out, 'ROOT') ?? '',
    gpus: parseGpus(out),
    nvidiaSmi: marker(out, 'NVSMI') === '1',
    diskFreeBytes: Number.isFinite(disk) && marker(out, 'DISK_FREE') ? disk : null,
    engines: markers(out, 'ENGINE'),
    modelSlugs: markers(out, 'MODEL'),
    watchdogRunning: marker(out, 'WATCHDOG_RUNNING') === '1',
    watchdogState: (marker(out, 'WATCHDOG_STATE') ?? '').trim(),
    foreign: marker(out, 'FOREIGN') === '1'
  }
}

export function freeSpaceScript(): string {
  return bash`
${PRELUDE}
emit DISK_FREE "$(df -PB1 "$R" 2>/dev/null | awk 'NR==2{print $4}')"
`
}

// ---------- Engine ----------

export interface EngineInstallInput {
  tag: string
  llama: { url: string; sha256: string; size: number }
  cudart: { url: string; sha256: string; size: number }
}

const GITHUB_URL_RE = /^https:\/\/(github\.com\/ggml-org\/llama\.cpp\/releases\/download|objects\.githubusercontent\.com)\/[A-Za-z0-9._~%/?=&+-]+$/

/**
 * Downloads both packages (curl, retried, resumed with -C -), checks the sha256 digests, unpacks next to the final
 * folder and moves it into place, finds llama-server wherever the package keeps it, writes `run-env` (LD_LIBRARY_PATH
 * with every folder that holds .so files), then runs --version and --list-devices. `.ok` is written last.
 */
export function engineInstallScript(input: EngineInstallInput): string {
  const tag = assertTag(input.tag)
  for (const p of [input.llama, input.cudart]) {
    if (!GITHUB_URL_RE.test(p.url)) throw new Error('The engine download address is not a GitHub release address.')
    assertSha256(p.sha256)
    if (!Number.isSafeInteger(p.size) || p.size < 0) throw new Error('The engine download size is not valid.')
  }
  return bash`
set -u
${PRELUDE}
TAG=${shq(tag)}
E="$R/engine/$TAG"
if [ -f "$E/.ok" ]; then emit ENGINE_OK "$TAG"; exit 0; fi
W="$R/engine/.dl-$TAG"
mkdir -p "$W"
say(){ emit STAGE "$*"; }
need=
command -v curl >/dev/null 2>&1 || need="$need curl"
[ -e /etc/ssl/certs/ca-certificates.crt ] || need="$need ca-certificates"
ldconfig -p 2>/dev/null | grep -q 'libgomp\.so\.1' || need="$need libgomp1"
if [ -n "$need" ]; then
say "Installing$need"
export DEBIAN_FRONTEND=noninteractive
{ apt-get update -qq && apt-get install -y libgomp1 curl ca-certificates; } >"$W/apt.log" 2>&1 || { emit ERR "Could not install the packages with apt-get: $(tail -n 2 "$W/apt.log" | tr '\n' ' ')"; exit 12; }
fi
have(){ [ -f "$1" ] && printf '%s  %s\n' "$2" "$1" | sha256sum -c --status 2>/dev/null; }
fetch(){
local url=$1 f=$2 sha=$3 label=$4 size=$5 i rc wp
have "$f" "$sha" && return 0
for i in 1 2 3 4 5 6; do
say "Downloading $label"
( while sleep 2; do emit PROG "$label|$(stat -c%s "$f" 2>/dev/null || echo 0)|$size"; done ) & wp=$!
curl -fsSL --retry 3 --retry-delay 2 --connect-timeout 20 -C - -o "$f" "$url"; rc=$?
kill "$wp" 2>/dev/null; wait "$wp" 2>/dev/null
if have "$f" "$sha"; then emit PROG "$label|$size|$size"; return 0; fi
if [ "$rc" = 0 ] || [ "$i" -ge 3 ]; then rm -f "$f"; fi
sleep @{RZ_RETRY_SLEEP:-3}
done
emit ERR "Could not download $label (network or checksum)"; exit 20
}
fetch ${shq(input.llama.url)} "$W/llama.tgz" ${shq(input.llama.sha256)} llama.cpp ${input.llama.size}
fetch ${shq(input.cudart.url)} "$W/cudart.tgz" ${shq(input.cudart.sha256)} "CUDA runtime" ${input.cudart.size}
say "Unpacking"
rm -rf "$W/x" "$E"; mkdir -p "$W/x"
{ tar -xzf "$W/llama.tgz" -C "$W/x" && tar -xzf "$W/cudart.tgz" -C "$W/x"; } || { emit ERR "Could not unpack the downloads."; exit 21; }
mv "$W/x" "$E" || { emit ERR "Could not move the engine into place."; exit 21; }
S=$(find "$E" -name llama-server -type f 2>/dev/null | head -1)
[ -n "$S" ] || { emit ERR "llama-server is not in the downloaded package."; exit 22; }
chmod +x "$S"
libs=$(find "$E" -name '*.so*' -printf '%h\n' 2>/dev/null | sort -u | tr '\n' ':')
printf 'RZ_SERVER=%q\n' "$S" >"$E/run-env"
printf 'export LD_LIBRARY_PATH=%q"@{LD_LIBRARY_PATH:-}"\n' "$libs$(dirname "$S"):" >>"$E/run-env"
say "Checking llama-server"
v=$( ( . "$E/run-env"; "$RZ_SERVER" --version ) 2>&1 ) || { emit ERR "llama-server does not start: $(printf '%s' "$v" | tail -n 3 | tr '\n' ' ')"; exit 23; }
emit VERSION "$(printf '%s' "$v" | tail -n 1)"
d=$( ( . "$E/run-env"; "$RZ_SERVER" --list-devices ) 2>&1 ) || { emit ERR "llama-server cannot list devices: $(printf '%s' "$d" | tail -n 3 | tr '\n' ' ')"; exit 24; }
emit DEVICES "$(printf '%s' "$d" | tr '\n' ';' | head -c 400)"
touch "$E/.ok"
rm -rf "$W"
emit ENGINE_OK "$TAG"
`
}

// ---------- Model files ----------

export interface HfFileInput {
  url: string
  name: string
  /** null = ask the server. */
  size: number | null
  /** null = not known; the size is still checked. */
  sha256: string | null
}

const HF_URL_RE = /^https:\/\/huggingface\.co\/[A-Za-z0-9._~%/+@-]+$/

// ---------- Owner of a model folder (design: docs/vast-design.md, section 6) ----------

/**
 * Only one script at a time may write into models/<slug>/. Every process of a download (the script, its subshells, every
 * curl) has `models/.own-<slug>` (next to the folder, so that removing the model keeps it) open as file descriptor 9,
 * which it inherits, so what an earlier script left running is found through /proc even when its parent is gone. take_dl opens the file first and then ends everybody else who has it open: of two
 * scripts that start together at most one stays. This is the normal case, not a rare one: ssh.run runs a script again on the next
 * route after a transport error, and a dropped or cancelled ssh does not end remote children. Needs emit.
 * The scan starts no process of its own (a child would have the file open too).
 */
const OWNER_FN = bash`
holds(){ local p; HOLD=(); for p in /proc/[0-9]*; do p=@{p#/proc/}; [ "$p" = "$$" ] || [ "$p" = "$BASHPID" ] || { [ "/proc/$p/fd/9" -ef "@{1%/*}/.own-@{1##*/}" ] && HOLD+=("$p"); }; done; return 0; }
take_dl(){ local i
mkdir -p "$1"; { exec 9>>"@{1%/*}/.own-@{1##*/}"; } 2>/dev/null || { emit ERR "The model folder on the instance cannot be written to."; exit 38; }
for i in $(seq 1 40); do
holds "$1"; [ "@{#HOLD[@]}" = 0 ] && return 0
kill -KILL "@{HOLD[@]}" 2>/dev/null; sleep 0.25
done
emit ERR "An earlier download on the instance would not stop."; exit 38; }
`

/** A file of this name recorded as complete from another address (another revision, say) is not kept. Needs $M. */
const SRC_FN = bash`
src_check(){ local rec; rec=$(awk -F'\t' -v n="$2" '$1==n{print $4}' "$M/.complete" 2>/dev/null)
if [ -n "$rec" ] && [ "$rec" != - ] && [ "$rec" != "$1" ]; then
rm -f "$M/$2" "$M/$2.part"
awk -F'\t' -v n="$2" '$1!=n' "$M/.complete" >"$M/.complete.tmp" 2>/dev/null && mv "$M/.complete.tmp" "$M/.complete"
fi; }
`

/**
 * Downloads files into models/<slug>/ with curl. The Hugging Face token never reaches the instance. The first line of
 * stdin is empty (public files), or, for one gated file, a curl config line with the short-lived signed address this
 * PC got for it: it reaches curl through a pipe from printf, a shell builtin, so it is not in argv, the environment or
 * a file. Exit 36: the signed address was refused (they expire after about an hour). Resumes with -C - into `<name>.part`, checks size and
 * sha256 (once more from scratch if a resumed file fails the hash), then renames.
 *
 * With a known size and at least 64 MB missing it fetches 16 MB chunks over 16 connections (one curl per chunk, ranged,
 * address on stdin) and appends them to `.part` in order, so `.part` is always a plain prefix of the file and the only
 * resume state. A server that answers a ranged request with 200 gets the single curl. A script whose stdout is gone
 * (the progress line cannot be printed) ends itself and its workers. It takes over the folder first (see OWNER_FN).
 * RZ_CHUNK, RZ_CONNS, RZ_MULTI_MIN and RZ_POLL are for tests.
 */
export function hfDownloadScript(slug: string, files: HfFileInput[]): string {
  assertSlug(slug)
  const calls = files.map((f) => {
    if (!HF_URL_RE.test(f.url)) throw new Error('A download address is not a Hugging Face address.')
    assertFileName(f.name)
    if (f.size !== null && (!Number.isSafeInteger(f.size) || f.size < 0)) throw new Error('A file size is not valid.')
    if (f.sha256 !== null) assertSha256(f.sha256)
    return `dl ${shq(f.url)} ${shq(f.name)} ${f.size === null ? '-' : f.size} ${f.sha256 === null ? '-' : shq(f.sha256)} 0`
  })
  return (
    bash`
set -u
${PRELUDE}
IFS= read -r CFG
M="$R/models/"${shq(slug)}
mkdir -p "$M"
${OWNER_FN}
${SRC_FN}
# With stdout gone, printf fails instead of killing the subshell that prints: that failure ends the whole script.
trap '' PIPE
take_dl "$M"
rm -rf "$M/.dl"
trap 'rm -rf "$M/.dl"' EXIT
CH=@{RZ_CHUNK:-16777216}; NC=@{RZ_CONNS:-16}; ACFG=
bye(){ holds "$M"; kill -KILL $$ @{HOLD[@]+"@{HOLD[@]}"} 2>/dev/null; exit 0; }
tree(){ cat /proc/[0-9]*/stat 2>/dev/null | awk -v r="$1" '{s=$0; sub(/^.*\) /,"",s); split(s,a," "); pp[$1]=a[2]} END{k[r]=1; do{c=0; for(p in pp) if(!(p in k) && (pp[p] in k)){k[p]=1; c=1}}while(c); for(p in k) print p}'; }
pbytes(){ local p c; p=$(stat -c%s "$1" 2>/dev/null || echo 0); c=$(stat -c%s "$M"/.dl/*[0-9] 2>/dev/null | awk '{t+=$1}END{print t+0}'); echo $((p+c)); }
curlc(){ if [ -n "$CFG" ]; then printf '%s\n' "$CFG" | curl --globoff --config - "@{@:2}"; else curl "@{@:2}" "$1"; fi; }
sha_ok(){ [ "$2" = - ] || [ "$(sha256sum "$1" | cut -d' ' -f1)" = "$2" ]; }
# The address a ranged fetch of a public file uses: where huggingface.co sends it (a signed storage address). Returns 0 = RES set,
# 1 = try again later, 2 = no ranges (single curl), 4 = refused, 5 = not found.
resolve(){ local o c
o=$(curl -sSL -r 0-0 --connect-timeout 20 -m 60 -o /dev/null -w '%{http_code} %{url_effective}' "$1" 2>/dev/null); c=@{o%% *}; RES=@{o#* }
case "$RES" in https://*) ;; *) RES=;; esac
case "$RES" in *[!!-~]*|*\"*|*\\*) RES=;; esac
case "$c" in 206) [ -n "$RES" ] && return 0; return 2;; 401|403) return 4;; 404) return 5;; 200|2??|3??) return 2;; *) return 1;; esac; }
# One chunk (name, offset, length, curl config line), up to 3 tries; the answer is written to <chunk>.rc: ok, whole (the server
# ignored the range), auth, missing or fail. --max-filesize keeps a server that sends the whole file for a range from doing so.
chunk(){ local f="$M/.dl/$1.$2" t c sz r=fail
for t in 1 2 3; do
rm -f "$f"
c=$(printf '%s\n' "$4" | curl --globoff --config - -sSL -r "$2-$(($2+$3-1))" --max-filesize "$3" --connect-timeout 20 --speed-limit 1000 --speed-time 60 -o "$f" -w '%{http_code}' 2>/dev/null)
sz=$(stat -c%s "$f" 2>/dev/null || echo 0)
case "$c" in
206) if [ "$sz" = "$3" ]; then r=ok; break; fi; r=fail;;
200) r=whole; break;;
401|403) r=auth; break;;
404) r=missing; break;;
*) r=fail;;
esac
sleep @{RZ_RETRY_SLEEP:-3}
done
rm -f "$f.rc.tmp"; printf '%s\n' "$r" >"$f.rc.tmp" && mv "$f.rc.tmp" "$f.rc"; }
# These three use the locals of mfetch (bash scopes by call).
launch(){ local off=$((start+$1*CH)) len; len=$((size-off)); [ "$len" -gt "$CH" ] && len=$CH
gn[$1]=$gen; ( chunk "$name" "$off" "$len" "$ACFG" ) & wp[$1]=$!; }
stop_workers(){ local k x; for ((k=app;k<next;k++)); do [ -f "$W/$name.$((start+k*CH)).rc" ] && continue; for x in $(tree "@{wp[k]}"); do kill -KILL "$x" 2>/dev/null; done; wait "@{wp[k]}" 2>/dev/null; done; rm -rf "$W"; }
# Fetches the rest of a file (after the bytes of .part) in chunks over several connections, appended to .part in order as
# they finish. Returns 0 = done, 1 = try again later, 2 = no ranges, 3 = the signed address was refused, 4 = refused, 5 = not found.
mfetch(){ local url=$1 name=$2 size=$3 part=$4 W="$M/.dl" start n next=0 app=0 gen=0 nres=0 r f rr k run
local -a wp=() gn=()
start=$(stat -c%s "$part" 2>/dev/null || echo 0)
n=$(( (size-start+CH-1)/CH ))
rm -rf "$W"; mkdir -p "$W"
if [ -n "$CFG" ]; then ACFG=$CFG; else resolve "$url"; rr=$?; [ "$rr" = 0 ] || { rm -rf "$W"; return "$rr"; }; ACFG="url = \"$RES\""; fi
while [ "$app" -lt "$n" ]; do
# One slow connection must not hold the others back: they go up to three rounds ahead of the chunk that is appended next.
run=0; for ((k=app;k<next;k++)); do [ -f "$W/$name.$((start+k*CH)).rc" ] || run=$((run+1)); done
# Chunks that were refused with an older address are fetched again side by side, not one by one when their turn comes.
for ((k=app+1;k<next&&run<NC;k++)); do
[ "@{gn[k]}" != "$gen" ] && read -r r 2>/dev/null <"$W/$name.$((start+k*CH)).rc" && [ "$r" = auth ] || continue
rm -f "$W/$name.$((start+k*CH)).rc"; launch "$k"; run=$((run+1))
done
while [ "$next" -lt "$n" ] && [ "$run" -lt "$NC" ] && [ $((next-app)) -lt $((NC*3)) ]; do launch "$next"; next=$((next+1)); run=$((run+1)); done
f="$W/$name.$((start+app*CH))"
if [ ! -f "$f.rc" ]; then
# A worker that is gone without an answer (killed, or a full disk) must not be waited for.
if ! kill -0 "@{wp[app]}" 2>/dev/null && [ ! -f "$f.rc" ]; then stop_workers; return 1; fi
sleep 0.2; continue
fi
r=$(cat "$f.rc")
case "$r" in
ok) if cat "$f" >>"$part"; then rm -f "$f" "$f.rc"; app=$((app+1)); else stop_workers; return 1; fi;;
whole) stop_workers; return 2;;
missing) stop_workers; return 5;;
auth)
if [ -n "$CFG" ]; then stop_workers; return 3; fi
if [ "@{gn[app]}" = "$gen" ]; then
# Only a chunk that was started with the newest address may ask for another one.
if [ "$nres" -ge 6 ]; then stop_workers; return 4; fi
nres=$((nres+1)); resolve "$url"; rr=$?; [ "$rr" = 0 ] || { stop_workers; return "$rr"; }
gen=$((gen+1)); ACFG="url = \"$RES\""
fi
rm -f "$f.rc"; launch "$app";;
*) stop_workers; return 1;;
esac
done
rm -rf "$W"; return 0; }
dl(){
local url=$1 name=$2 size=$3 sha=$4 again=$5 dest="$M/$2" part="$M/$2.part" cur code try wp rc ok=0 multi=1 mr nc
if [ "$size" = - ] && [ -z "$CFG" ]; then size=$(curlc "$url" -sIL --max-time 60 2>/dev/null | tr -d '\r' | awk 'tolower($1)=="content-length:"{v=$2} END{print v+0}'); [ "$size" = 0 ] && size=-; fi
# The file of this name may have come from another address (another revision, say): then do not keep it.
src_check "$url" "$name"
if [ -f "$dest" ]; then
cur=$(stat -c%s "$dest")
if [ "$size" != - ] && [ "$cur" = "$size" ] && sha_ok "$dest" "$sha"; then emit FILE_DONE "$name"; return 0; fi
rm -f "$dest"
fi
[ -f "$part" ] || : >"$part"
for try in 1 2 3 4 5 6; do
cur=$(stat -c%s "$part")
if [ "$size" != - ] && [ "$cur" -gt "$size" ]; then rm -f "$part"; : >"$part"; cur=0; fi
if [ "$size" != - ] && [ "$cur" = "$size" ]; then ok=1; break; fi
if [ "$multi" = 1 ] && [ "$size" != - ] && [ $((size-cur)) -ge "@{RZ_MULTI_MIN:-67108864}" ]; then
nc=$(( (size-cur+CH-1)/CH )); [ "$nc" -gt "$NC" ] && nc=$NC
emit STAGE "Downloading $name, $nc connections"
# Never lower than before and never over the size: a chunk is counted twice for an instant while it is appended.
( mx=0; while sleep "@{RZ_POLL:-2}"; do v=$(pbytes "$part"); [ "$v" -gt "$size" ] && v=$size; [ "$v" -lt "$mx" ] && v=$mx; mx=$v; emit PROG "$name|$v|$size" || bye; done ) & wp=$!
mfetch "$url" "$name" "$size" "$part"; mr=$?
kill "$wp" 2>/dev/null; wait "$wp" 2>/dev/null
case "$mr" in
0) ok=1; break;;
2) multi=0; continue;;
3) emit ERR "The download address of $name was refused (http 403)."; exit 36;;
4) emit ERR "Hugging Face refused the download of $name. A gated or private repository needs a token with access."; exit 31;;
5) emit ERR "Hugging Face has no file $name at this address (http 404)."; exit 32;;
esac
sleep @{RZ_RETRY_SLEEP:-3}
continue
fi
emit STAGE "Downloading $name"
( while sleep "@{RZ_POLL:-2}"; do emit PROG "$name|$(stat -c%s "$part" 2>/dev/null || echo 0)|$size" || bye; done ) & wp=$!
code=$(curlc "$url" -fsSL --retry 3 --retry-delay 2 --connect-timeout 20 --speed-limit 1000 --speed-time 60 -C - -o "$part" -w '%{http_code}' 2>/dev/null); rc=$?
kill "$wp" 2>/dev/null; wait "$wp" 2>/dev/null
case "$code" in
401|403) if [ -n "$CFG" ]; then emit ERR "The download address of $name was refused (http $code)."; exit 36; fi; emit ERR "Hugging Face refused the download of $name (http $code). A gated or private repository needs a token with access."; exit 31;;
404) emit ERR "Hugging Face has no file $name at this address (http 404)."; exit 32;;
esac
if [ "$rc" = 0 ]; then ok=1; break; fi
if [ "$code" = 416 ] && [ "$size" != - ] && [ "$(stat -c%s "$part" 2>/dev/null)" = "$size" ]; then ok=1; break; fi
sleep @{RZ_RETRY_SLEEP:-3}
done
# Without a successful transfer a partial file must never pass for the whole one (no size or checksum may be there to catch it).
if [ "$ok" != 1 ]; then emit ERR "$name could not be downloaded; the partial file is kept for the next try."; exit 35; fi
cur=$(stat -c%s "$part" 2>/dev/null || echo 0)
if [ "$size" != - ] && [ "$cur" != "$size" ]; then emit ERR "$name is incomplete ($cur of $size bytes)."; exit 33; fi
if [ "$sha" != - ]; then
emit STAGE "Checking $name"
if ! sha_ok "$part" "$sha"; then
rm -f "$part"
if [ "$again" = 0 ]; then dl "$url" "$name" "$size" "$sha" 1; return $?; fi
emit ERR "$name does not match its checksum."; exit 34
fi
fi
mv "$part" "$dest" && emit FILE_DONE "$name"
}
${calls.join('\n')}
exit 0
`
  )
}

/** The data line that follows hfDownloadScript on stdin: the signed address of its one file, or nothing. */
export function hfConfigLine(signedUrl: string | null): string {
  if (!signedUrl) return '\n'
  if (!/^https:\/\/[!-~]+$/.test(signedUrl) || /["\\]/.test(signedUrl)) throw new Error('The download address has characters that cannot be sent.')
  return `url = "${signedUrl}"\n`
}

/** Ends a download that still runs on the instance (a cancelled setup) and clears its chunks; `.part` stays for the next try. */
export function hfStopScript(slug: string): string {
  assertSlug(slug)
  return bash`
set -u
${PRELUDE}
M="$R/models/"${shq(slug)}
${OWNER_FN}
[ -d "$M" ] || { emit STOPPED 1; exit 0; }
take_dl "$M"
rm -rf "$M/.dl"
emit STOPPED 1
`
}

export interface CompleteFile {
  name: string
  size: number
  /** '-' when unknown. */
  sha256: string
  /** Where the file came from: its Hugging Face address, or '-' for an upload (its sha256 is the PC's). */
  source: string
}

/** Records files as complete in models/<slug>/.complete (name, size, sha256, source per line); each must exist with that size. */
export function markCompleteScript(slug: string, files: CompleteFile[]): string {
  assertSlug(slug)
  const adds = files.map((f) => {
    assertFileName(f.name)
    if (!Number.isSafeInteger(f.size) || f.size < 0) throw new Error('A file size is not valid.')
    if (f.sha256 !== '-') assertSha256(f.sha256)
    if (f.source !== '-' && !HF_URL_RE.test(f.source) && !/^pc:\d{1,16}$/.test(f.source)) throw new Error('A file source is not valid.')
    return `add ${shq(f.name)} ${f.size} ${shq(f.sha256)} ${shq(f.source)} || exit 61`
  })
  return bash`
set -u
${PRELUDE}
M="$R/models/"${shq(slug)}
mkdir -p "$M"; touch "$M/.complete"
add(){
[ "$(stat -c%s "$M/$1" 2>/dev/null)" = "$2" ] || { emit ERR "$1 is not complete on the instance."; return 1; }
awk -F'\t' -v n="$1" '$1!=n' "$M/.complete" >"$M/.complete.tmp"
printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" >>"$M/.complete.tmp"
mv "$M/.complete.tmp" "$M/.complete"
}
${adds.join('\n')}
emit COMPLETE 1
`
}

export interface CompleteEntry {
  name: string
  size: number
  sha256: string
  /** '-' = not recorded (an upload, or an older line). */
  source: string
}

/** The lines of a `.complete` file. */
export function parseComplete(text: string): CompleteEntry[] {
  const out: CompleteEntry[] = []
  for (const line of text.split(/\r?\n/)) {
    const [name, size, sha256, source] = line.split('\t')
    if (name && /^\d+$/.test(size ?? '')) out.push({ name, size: Number(size), sha256: sha256 || '-', source: source || '-' })
  }
  return out
}

/**
 * Sizes of the files of a model folder: FILE=name|size, PART=name|size, the .complete content (base64) and the free disk.
 * Chunks that a download left behind are cleared first, so that they do not count against the free disk.
 */
export function modelStateScript(slug: string): string {
  assertSlug(slug)
  return bash`
${PRELUDE}
M="$R/models/"${shq(slug)}
${OWNER_FN}
holds "$M"; [ "@{#HOLD[@]}" = 0 ] && rm -rf "$M/.dl"
for f in "$M"/*; do [ -f "$f" ] || continue; n=$(basename "$f"); case "$n" in *.part) emit PART "$n|$(stat -c%s "$f")";; *) emit FILE "$n|$(stat -c%s "$f")";; esac; done
emit COMPLETE_B64 "$(base64 -w0 "$M/.complete" 2>/dev/null)"
emit DISK_FREE "$(df -PB1 "$R" 2>/dev/null | awk 'NR==2{print $4}')"
`
}

/** Refuses (exit 60) while a live server of ours was started on this model; ends a download into the folder first (exit 38 when it will not stop). */
export function removeModelScript(slug: string): string {
  assertSlug(slug)
  return bash`
${PRELUDE}
${STATUS_FN}
S=${shq(slug)}
M="$R/models/$S"
${OWNER_FN}
for D in "$R"/run/*/; do
id=$(basename "$D"); [ -f "$D/args" ] || continue
if grep -qF "/models/$S/" "$D/args" 2>/dev/null; then
out=$(rzs "$id"); case "$out" in *"@rz ALIVE=1"*) emit ERR "A server is running on this model. Unload it first."; exit 60;; esac
fi
done
take_dl "$M"
rm -rf "$M"
emit REMOVED 1
`
}

// ---------- Uploads (transfer.ts) ----------

/** FINAL and PART: sizes of models/<slug>/<name> and its .part, -1 when missing. */
export function uploadStateScript(slug: string, name: string): string {
  assertSlug(slug)
  assertFileName(name)
  return bash`
${PRELUDE}
M="$R/models/"${shq(slug)}
mkdir -p "$M"
f="$M/"${shq(name)}
emit FINAL "$(stat -c%s "$f" 2>/dev/null || echo -1)"
emit PART "$(stat -c%s "$f.part" 2>/dev/null || echo -1)"
`
}

/** The bytes after the script on stdin are appended to the .part file. */
export function uploadAppendScript(slug: string, name: string): string {
  assertSlug(slug)
  assertFileName(name)
  return bash`
${PRELUDE}
M="$R/models/"${shq(slug)}
mkdir -p "$M"
f="$M/"${shq(name)}".part"
cat >>"$f"
emit PART "$(stat -c%s "$f" 2>/dev/null || echo -1)"
`
}

/** Compares the .part (or the final file with `of: 'final'`) with the expected size and sha256, then moves the .part into place. */
export function uploadFinishScript(slug: string, name: string, size: number, sha256: string, of: 'part' | 'final' = 'part'): string {
  assertSlug(slug)
  assertFileName(name)
  assertSha256(sha256)
  if (!Number.isSafeInteger(size) || size < 0) throw new Error('A file size is not valid.')
  return bash`
${PRELUDE}
M="$R/models/"${shq(slug)}
f="$M/"${shq(name)}
src="$f"${of === 'part' ? '".part"' : ''}
got=$(stat -c%s "$src" 2>/dev/null || echo -1)
emit SIZE "$got"
if [ "$got" != ${size} ]; then emit RESULT short; exit 0; fi
h=$(sha256sum "$src" | cut -d' ' -f1)
emit SHA "$h"
if [ "$h" != ${shq(sha256)} ]; then emit RESULT mismatch; exit 0; fi
${of === 'part' ? 'mv "$src" "$f"' : ':'}
emit RESULT ok
`
}

export function uploadDiscardScript(slug: string, name: string): string {
  assertSlug(slug)
  assertFileName(name)
  return bash`
${PRELUDE}
f="$R/models/"${shq(slug)}"/"${shq(name)}
rm -f "$f" "$f.part"
emit DISCARDED 1
`
}

// ---------- Server ----------

export interface ServerStartInput {
  id: string
  tag: string
  /** llama-server arguments with PORT_PLACEHOLDER where the port goes. */
  args: string[]
  hash: string
  /** GPU memory the server is expected to take, MiB; null = unknown. */
  expectMiB: number | null
  /** For tests. */
  maxIdleSeconds?: number
}

/** Free port, run folder, supervisor, setsid nohup. Reports PORT and PID, or ERR with the log's tail (base64 in LOG). */
export function serverStartScript(input: ServerStartInput): string {
  const id = assertInstanceId(input.id)
  const tag = assertTag(input.tag)
  assertSha256(input.hash)
  for (const a of input.args) if (/[\0\r\n]/.test(a)) throw new Error('An argument has a line break in it.')
  const expect = input.expectMiB === null ? '-1' : String(intIn(Math.round(input.expectMiB), 0, 10_000_000, 'The memory estimate'))
  return bash`
set -u
${PRELUDE}
ID=${shq(id)}
E="$R/engine/"${shq(tag)}
[ -f "$E/.ok" ] || { emit ERR "The engine is not installed on the instance."; exit 40; }
# A token of the user's account environment must not reach the server or its supervisor (a model could print its environment).
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_TOKEN
. "$E/run-env" || { emit ERR "The engine's environment file is broken."; exit 41; }
D="$R/run/$ID"
mkdir -p "$D"
old=$(cat "$D/pid" 2>/dev/null)
if [ -n "$old" ] && kill -0 "$old" 2>/dev/null; then case "$(readlink "/proc/$old/exe" 2>/dev/null)" in "$R"/engine/*) emit ERR "A server of this instance is still running."; exit 42;; esac; fi
P=
# A listening socket on the port shows in /proc/net/tcp (a connect test can hang behind a firewall).
for i in $(seq 1 60); do c=$((20000 + RANDOM % 20000)); if ! grep -qiE ":$(printf '%04X' "$c") [0-9A-F:]+ 0A" /proc/net/tcp /proc/net/tcp6 2>/dev/null; then P=$c; break; fi; done
[ -n "$P" ] || { emit ERR "No free port on the instance."; exit 43; }
A=()
for a in ${input.args.map((a) => shq(a)).join(' ')}; do [ "$a" = ${shq(PORT_PLACEHOLDER)} ] && a=$P; A+=("$a"); done
rm -f "$D/pid"; : >"$D/log"
printf '%s' "$P" >"$D/port"; printf '%s' ${shq(input.hash)} >"$D/hash"; printf '%s' ${shq(expect)} >"$D/expect"
printf '%s\n' "@{A[@]}" >"$D/args"
cat >"$D/sup.sh" <<'RZSUP'
D=$1; K=$2; R=$3; M=@{RZ_MAXAGE:-${input.maxIdleSeconds ?? SERVER_MAX_IDLE_SECONDS}}; p=$(cat "$D/pid")
t(){ sed 's/^.*) //' "/proc/$p/stat" 2>/dev/null | awk '{print $20}'; }
st=$(t)
ours(){ case "$(readlink "/proc/$p/exe" 2>/dev/null)" in "$R"/engine/*) [ -n "$st" ] && [ "$(t)" = "$st" ];; *) return 1;; esac; }
while ours; do
sleep "@{RZ_SUP_TICK:-60}"
[ -f "$K" ] || continue
a=$(( $(date +%s) - $(stat -c %Y "$K") ))
if [ "$a" -gt "$M" ]; then
echo "$(date -u +%FT%TZ) [rulz] keepalive is older than $M s, stopping the server" >>"$D/log"
ours && kill "$p" 2>/dev/null; sleep @{RZ_SUP_KILL:-15}; ours && kill -9 "$p" 2>/dev/null; break
fi
done
RZSUP
RZ_ID="$ID" setsid bash -c 'echo $$ >"$1"; shift; exec nohup "$@"' _ "$D/pid" "$RZ_SERVER" "@{A[@]}" >>"$D/log" 2>&1 </dev/null &
for i in $(seq 1 50); do [ -s "$D/pid" ] && break; sleep 0.1; done
setsid nohup bash "$D/sup.sh" "$D" "$R/keepalive" "$R" >/dev/null 2>&1 </dev/null &
sleep 1
p=$(cat "$D/pid" 2>/dev/null)
if [ -n "$p" ] && kill -0 "$p" 2>/dev/null; then emit PORT "$P"; emit PID "$p"; emit STARTED 1; exit 0; fi
emit LOG "$(tail -c 4000 "$D/log" 2>/dev/null | base64 -w0)"
emit ERR "llama-server stopped right after starting."
exit 44
`
}

export function serverStatusScript(id: string): string {
  assertInstanceId(id)
  return bash`
${PRELUDE}
${STATUS_FN}
rzs ${shq(id)}
`
}

/** Reads the server's log from a byte offset (negative: that many bytes from the end), at most 64 KiB, as base64. */
export function serverLogScript(id: string, offset: number): string {
  assertInstanceId(id)
  if (!Number.isSafeInteger(offset)) throw new Error('The log offset is not a whole number.')
  return bash`
${PRELUDE}
${STATUS_FN}
ID=${shq(id)}
rzs "$ID"
L="$R/run/$ID/log"
sz=$(stat -c%s "$L" 2>/dev/null || echo 0)
off=${offset}
[ "$off" -lt 0 ] && off=$((sz+off))
[ "$off" -lt 0 ] && off=0
if [ "$off" -gt "$sz" ]; then off=0; emit LOG_RESET 1; fi
n=$((sz-off)); [ "$n" -gt 65536 ] && n=65536
emit LOG "$(tail -c +$((off+1)) "$L" 2>/dev/null | head -c "$n" | base64 -w0)"
emit LOG_NEXT $((off+n))
`
}

/** TERM, wait, KILL, confirm it is gone; exit 50 when it is not. Only a process whose binary is in our engine folder is killed. */
export function serverStopScript(id: string): string {
  assertInstanceId(id)
  return bash`
${PRELUDE}
ID=${shq(id)}
D="$R/run/$ID"
t(){ sed 's/^.*) //' "/proc/$1/stat" 2>/dev/null | awk '{print $20}'; }
ours(){ case "$(readlink "/proc/$1/exe" 2>/dev/null)" in "$R"/engine/*) return 0;; esac; return 1; }
p=$(cat "$D/pid" 2>/dev/null)
found=
if [ -n "$p" ] && kill -0 "$p" 2>/dev/null && ours "$p"; then found=$p; else
# No usable pid file: our server carries RZ_ID in its environment.
for e in /proc/[0-9]*/environ; do q=@{e#/proc/}; q=@{q%/environ}; ours "$q" || continue
if tr '\0' '\n' <"$e" 2>/dev/null | grep -qx "RZ_ID=$ID"; then found=$q; break; fi; done
fi
if [ -z "$found" ]; then
if [ -n "$p" ] && kill -0 "$p" 2>/dev/null; then emit NOTOURS 1; fi
rm -f "$D/pid"; emit STOPPED 1; exit 0
fi
p=$found; st=$(t "$p")
gone(){ ! kill -0 "$p" 2>/dev/null || grep -q '^State:[[:space:]]*Z' "/proc/$p/status" 2>/dev/null; }
kill -TERM "$p" 2>/dev/null
for i in $(seq 1 @{RZ_STOP_WAIT:-40}); do gone && break; sleep 0.5; done
if ! gone; then
# The pid could have been reused while we waited: KILL only the same process.
if ours "$p" && [ "$(t "$p")" = "$st" ]; then kill -KILL "$p" 2>/dev/null; fi
for i in $(seq 1 20); do gone && break; sleep 0.5; done
fi
if gone; then rm -f "$D/pid"; emit STOPPED 1; exit 0; fi
emit ERR "llama-server (pid $p) is still running on the instance."
exit 50
`
}

/** Everything a load needs to know before it starts: engine, model files, GPUs, foreign processes, the run folder. */
export function launchCheckScript(input: { id: string; tag: string; slug: string }): string {
  assertInstanceId(input.id)
  assertTag(input.tag)
  assertSlug(input.slug)
  return bash`
${PRELUDE}
${GPU_FN}
${FOREIGN_FN}
${STATUS_FN}
emit ROOT "$R"
[ -f "$R/engine/"${shq(input.tag)}"/.ok" ] && emit ENGINE_OK 1 || emit ENGINE_OK 0
emit COMPLETE_B64 "$(base64 -w0 "$R/models/"${shq(input.slug)}"/.complete" 2>/dev/null)"
for f in "$R/models/"${shq(input.slug)}/*; do [ -f "$f" ] || continue; n=$(basename "$f"); case "$n" in *.part) ;; *) emit FILE "$n|$(stat -c%s "$f")";; esac; done
rzg
emit FOREIGN "$(rzf)"
for d in "$R"/run/*/; do [ -d "$d" ] || continue; i=$(basename "$d"); o=$(rzs "$i"); case "$o" in *"@rz ALIVE=1"*) emit RUN "$i|$(cat "$d/port" 2>/dev/null)|$(cat "$d/hash" 2>/dev/null)";; esac; done
exit 0
`
}
