// Pure builders for the bash scripts that run inside WSL. Flags checked against ComfyUI 0.45.0 (comfy/cli_args.py,
// folder_paths.py, utils/extra_config.py, main.py).
import type { OwnServerSettings } from '@shared/types'
import { shellQuote } from './wsl'

/** Quotes a path for bash but leaves a leading `~/` (or `~`) unquoted so bash expands it. */
export function bashPath(p: string): string {
  const t = p.trim()
  if (t === '~') return '~'
  if (t.startsWith('~/')) return `~/${shellQuote(t.slice(2))}`
  return shellQuote(t)
}

function joinPath(dir: string, rel: string): string {
  const d = dir.trim().replace(/\/+$/, '')
  return d === '' ? `/${rel}` : `${d}/${rel}`
}

export function modelsPath(s: OwnServerSettings): string {
  return s.modelsDir.trim() ? s.modelsDir.trim() : joinPath(s.comfyDir, 'models')
}

/** What identifies one launched process: the absolute base directory it was given, its pid and its start time. */
export interface ProcessIdentity {
  /** Absolute base directory (the value printed as RULZ_BASE). */
  baseDir: string
  /** Field 22 of /proc/<pid>/stat; null only for a process whose start time was never recorded. */
  startTime: string | null
}

/**
 * Bash function RULZ_ID_OK <pid> <absolute base> <start time or "">. True only when the process is alive, has that
 * start time (so a reused pid never matches), has an argument ending in main.py, and has `--base-directory` followed by
 * exactly the base path (or `--base-directory=<path>`). /proc/<pid>/cmdline is read as NUL-separated arguments, so
 * `.../comfy-backup` never matches `.../comfy`.
 */
export const IDENTITY_FUNCTION = [
  'RULZ_ID_OK() {',
  '  local pid="$1" want="$2" start="$3" a prev="" hasmain=0 hasbase=0 st',
  '  [ -r "/proc/$pid/cmdline" ] || return 1',
  '  if [ -n "$start" ]; then',
  `    st=$(sed 's/.*) //' "/proc/$pid/stat" 2>/dev/null | cut -d' ' -f20)`,
  '    [ "$st" = "$start" ] || return 1',
  '  fi',
  `  while IFS= read -r -d '' a; do`,
  '    case "$a" in *main.py) hasmain=1;; esac',
  '    if [ "$prev" = "--base-directory" ] && [ "$a" = "$want" ]; then hasbase=1; fi',
  '    if [ "$a" = "--base-directory=$want" ]; then hasbase=1; fi',
  '    prev="$a"',
  '  done < "/proc/$pid/cmdline"',
  '  [ "$hasmain" = 1 ] && [ "$hasbase" = 1 ]',
  '}'
].join('\n')

const START_TIME_OF = (pidExpr: string): string => `$(sed 's/.*) //' /proc/${pidExpr}/stat 2>/dev/null | cut -d' ' -f20)`

export function buildStartScript(s: OwnServerSettings, port: number): string {
  const lines: string[] = []
  const add = (l: string): number => lines.push(l)
  add('set -e')
  add(`cd ${bashPath(s.comfyDir)}`)
  add(`mkdir -p ${bashPath(s.baseDir)}`)
  add(`RULZ_BASE="$(cd ${bashPath(s.baseDir)} && pwd -P)"`)
  add('echo "RULZ_BASE=$RULZ_BASE"')
  const args = [
    'main.py',
    '--base-directory',
    '"$RULZ_BASE"',
    '--port',
    String(Math.trunc(port)),
    '--listen',
    '127.0.0.1',
    '--disable-auto-launch'
  ]
  if (s.device.kind === 'gpu') args.push('--cuda-device', String(Math.trunc(s.device.index)))
  else args.push('--cpu')
  args.push('--models-directory', bashPath(modelsPath(s)))
  if (s.loadCustomNodes) {
    // --base-directory moves ComfyUI's custom_nodes folder too, so add the checkout's own folder as a second path.
    // The path goes into the yaml as a single-quoted scalar with ' doubled, so ", \ and ' in it stay valid.
    add('RULZ_COMFY_ABS="$(pwd -P)"')
    add(`RULZ_YAML_PATH="$(printf '%s' "$RULZ_COMFY_ABS" | sed "s/'/''/g")"`)
    add(
      `printf '%s\\n' 'rulz_custom_nodes:' "  base_path: '$RULZ_YAML_PATH'" '  custom_nodes: custom_nodes' > "$RULZ_BASE/extra_model_paths.yaml"`
    )
    args.push('--extra-model-paths-config', '"$RULZ_BASE/extra_model_paths.yaml"')
  }
  for (const a of s.extraArgs.split(/\s+/).filter(Boolean)) args.push(shellQuote(a))
  add(`RULZ_START="${START_TIME_OF('$$')}"`)
  add('printf \'%s %s\\n\' "$$" "$RULZ_START" > "$RULZ_BASE/server.pid"')
  add('echo "RULZ_PID=$$"')
  add('echo "RULZ_START=$RULZ_START"')
  add('export CUDA_DEVICE_ORDER=PCI_BUS_ID')
  add('export PYTHONUNBUFFERED=1')
  add(`exec ${bashPath(s.python)} ${args.join(' ')}`)
  return lines.join('\n') + '\n'
}

/** Signals one process, but only when its identity (see IDENTITY_FUNCTION) still matches right now. */
export function buildStopScript(pid: number, signal: 'INT' | 'KILL', id: ProcessIdentity): string {
  const n = Math.trunc(pid)
  if (!Number.isInteger(n) || n <= 1) throw new Error(`Invalid pid ${pid}`)
  return [
    IDENTITY_FUNCTION,
    `if RULZ_ID_OK ${n} ${shellQuote(id.baseDir)} ${shellQuote(id.startTime ?? '')}; then`,
    `  kill -s ${signal} ${n} || true`,
    'else',
    '  echo RULZ_GONE',
    'fi',
    'true',
    ''
  ].join('\n')
}

/** Prints RULZ_ALIVE while the identified process is still running. */
export function buildAliveScript(pid: number, id: ProcessIdentity): string {
  const n = Math.trunc(pid)
  return [
    IDENTITY_FUNCTION,
    `if RULZ_ID_OK ${n} ${shellQuote(id.baseDir)} ${shellQuote(id.startTime ?? '')}; then echo RULZ_ALIVE; fi`,
    'true',
    ''
  ].join('\n')
}

/**
 * Reads the pid file of a base directory. Prints RULZ_BASE=<absolute base>, then, when the file names a process that
 * passes the identity check, RULZ_STALE=<pid> <start time>.
 */
export function buildPidFileScript(baseDirSetting: string): string {
  return [
    IDENTITY_FUNCTION,
    `RULZ_BASE="$(cd ${bashPath(baseDirSetting)} 2>/dev/null && pwd -P)" || RULZ_BASE=""`,
    'if [ -n "$RULZ_BASE" ]; then',
    '  echo "RULZ_BASE=$RULZ_BASE"',
    '  if [ -r "$RULZ_BASE/server.pid" ]; then',
    '    read -r PID START < "$RULZ_BASE/server.pid" || true',
    `    case "$PID" in ''|*[!0-9]*) PID="";; esac`,
    `    case "$START" in *[!0-9]*) START="";; esac`,
    '    if [ -n "$PID" ] && RULZ_ID_OK "$PID" "$RULZ_BASE" "$START"; then echo "RULZ_STALE=$PID $START"; fi',
    '  fi',
    'fi',
    'true',
    ''
  ].join('\n')
}

/** Removes the pid file, but only when it still names this pid. */
export function buildRemovePidScript(baseDir: string, pid: number): string {
  const f = shellQuote(`${baseDir.replace(/\/+$/, '')}/server.pid`)
  return [`F=${f}`, `if [ -r "$F" ]; then read -r P _ < "$F" || true; [ "$P" = "${Math.trunc(pid)}" ] && rm -f "$F"; fi`, 'true', ''].join('\n')
}

/** One script for the pre-start checks: prints RULZ_OK, RULZ_COMFY, RULZ_PYTHON and the `free -b` table. */
export function buildProbeScript(s: OwnServerSettings): string {
  const py = s.python.trim()
  const pyCheck = py.includes('/') ? `[ -x ${bashPath(py)} ]` : `command -v ${shellQuote(py)} >/dev/null 2>&1`
  return [
    'echo RULZ_OK',
    `if cd ${bashPath(s.comfyDir)} 2>/dev/null && [ -f main.py ]; then`,
    '  echo RULZ_COMFY',
    `  if ${pyCheck}; then echo RULZ_PYTHON; fi`,
    'fi',
    'free -b 2>/dev/null',
    'true',
    ''
  ].join('\n')
}
