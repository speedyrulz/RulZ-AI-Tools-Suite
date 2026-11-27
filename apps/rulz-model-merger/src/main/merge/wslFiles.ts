// Small bash helpers that run inside WSL for merges: where the models folder is, which output files exist, free RAM,
// and moving a finished file into the models folder without ever overwriting or deleting anything.
// Every script is built by a pure function (testable), quotes every path with shellQuote and uses LF only.
import type { OwnServerSettings } from '@shared/types'
import { bashPath, modelsPath } from '../server/scripts'
import { shellQuote, type WslRunner } from '../server/wsl'

export type ModelKind = 'checkpoint' | 'diffusion'
export type ModelSubfolder = 'checkpoints' | 'diffusion_models'

export function subfolderOf(kind: ModelKind): ModelSubfolder {
  return kind === 'checkpoint' ? 'checkpoints' : 'diffusion_models'
}

/** `\\wsl.localhost\<distro>\home\x\file` for an absolute WSL path, so Windows code can read it with fs. */
export function toUncPath(distro: string, absWslPath: string): string {
  const d = distro.trim()
  const rest = absWslPath.replace(/\/+/g, '/').replace(/\//g, '\\')
  return `\\\\wsl.localhost\\${d}${rest.startsWith('\\') ? rest : `\\${rest}`}`
}

/** `<models>/<checkpoints|diffusion_models>/<name>`. Names may hold sub-folders; ".." and absolute names are refused. */
export function sourcePath(modelsAbs: string, kind: ModelKind, name: string): string {
  const n = name.trim()
  if (!n) throw new Error('No model file name was given.')
  if (n.startsWith('/') || n.startsWith('\\') || /^[A-Za-z]:/.test(n)) throw new Error(`"${name}" is not a name inside the models folder.`)
  const parts = n.split(/[\\/]+/)
  if (parts.some((p) => p === '..' || p === '.' || p === '')) throw new Error(`"${name}" is not a name inside the models folder.`)
  if (parts.some((p) => p.includes('\0'))) throw new Error(`"${name}" is not a valid file name.`)
  return `${modelsAbs.replace(/\/+$/, '')}/${subfolderOf(kind)}/${parts.join('/')}`
}

export function buildResolveModelsScript(s: OwnServerSettings): string {
  return [`cd ${bashPath(modelsPath(s))} 2>/dev/null && pwd -P`, 'true', ''].join('\n')
}

export async function resolveModelsDir(wsl: WslRunner, distro: string, s: OwnServerSettings): Promise<string> {
  const r = await wsl.run(distro, buildResolveModelsScript(s), { timeoutMs: 15000 })
  const line = r.stdout
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter((l) => l.startsWith('/'))
    .pop()
  if (r.code !== 0 || !line) throw new Error(`The models folder "${modelsPath(s)}" was not found in WSL.`)
  return line
}

export interface OutputCandidate {
  /** Absolute WSL path. */
  path: string
  name: string
  sizeBytes: number
  /** Modification time, seconds since the epoch. */
  mtime: number
}

/** Lists `<dir>/<subfolder>/<stem>_*_.safetensors`; one line per file: `RULZ_FILE|<size>|<mtime>|<name>`. */
export function buildListOutputScript(outputDirAbs: string, subfolder: string, stem: string): string {
  const dir = `${outputDirAbs.replace(/\/+$/, '')}/${subfolder.replace(/^\/+|\/+$/g, '')}`
  return [
    `STEM=${shellQuote(stem)}`,
    `cd ${shellQuote(dir)} 2>/dev/null || exit 0`,
    'for f in *; do',
    '  case "$f" in',
    '    "$STEM"_*_.safetensors)',
    '      [ -f "$f" ] || continue',
    `      set -- $(stat -c '%s %Y' -- "$f" 2>/dev/null) || continue`,
    '      echo "RULZ_FILE|$1|$2|$f";;',
    '  esac',
    'done',
    'true',
    ''
  ].join('\n')
}

export function parseOutputCandidates(stdout: string, outputDirAbs: string, subfolder: string): OutputCandidate[] {
  const dir = `${outputDirAbs.replace(/\/+$/, '')}/${subfolder.replace(/^\/+|\/+$/g, '')}`
  const out: OutputCandidate[] = []
  for (const raw of stdout.split(/\r?\n/)) {
    const line = raw.replace(/\r$/, '')
    if (!line.startsWith('RULZ_FILE|')) continue
    const first = line.indexOf('|', 10)
    const second = first < 0 ? -1 : line.indexOf('|', first + 1)
    if (first < 0 || second < 0) continue
    const sizeBytes = Number(line.slice(10, first))
    const mtime = Number(line.slice(first + 1, second))
    const name = line.slice(second + 1)
    if (!name || !Number.isFinite(sizeBytes) || !Number.isFinite(mtime)) continue
    out.push({ path: `${dir}/${name}`, name, sizeBytes, mtime })
  }
  return out
}

export async function listOutputCandidates(
  wsl: WslRunner,
  distro: string,
  outputDirAbs: string,
  subfolder: string,
  stem: string
): Promise<OutputCandidate[]> {
  const r = await wsl.run(distro, buildListOutputScript(outputDirAbs, subfolder, stem), { timeoutMs: 15000 })
  if (r.code !== 0) throw new Error(`Could not list the output folder: ${r.stderr.trim() || 'WSL did not answer'}`)
  return parseOutputCandidates(r.stdout, outputDirAbs, subfolder)
}

export function buildFreeMemoryScript(): string {
  return ["awk '/^MemAvailable:/ { print \"RULZ_MEM=\" $2 }' /proc/meminfo 2>/dev/null", 'true', ''].join('\n')
}

/** MemAvailable in bytes, or null when it cannot be read. */
export async function freeMemoryBytes(wsl: WslRunner, distro: string): Promise<number | null> {
  try {
    const r = await wsl.run(distro, buildFreeMemoryScript(), { timeoutMs: 10000 })
    const m = /RULZ_MEM=(\d+)/.exec(r.stdout)
    if (r.code !== 0 || !m) return null
    return Number(m[1]) * 1024
  } catch {
    return null
  }
}

/**
 * Moves `from` into `toDir` under a free name (`name.safetensors`, then `name-2.safetensors`, ...) with `mv -n`, checks
 * that the source is gone and the target exists, and moves `<from>.recipe.json` along under the matching name.
 * Prints RULZ_MOVED=<final path> and, when a sidecar moved, RULZ_SIDECAR=<path>. Never overwrites, never deletes.
 */
export function buildMoveScript(from: string, toDir: string): string {
  return [
    `SRC=${shellQuote(from)}`,
    `DIR=${shellQuote(toDir.replace(/\/+$/, ''))}`,
    '[ -f "$SRC" ] || { echo "RULZ_ERR=The file is no longer there."; exit 3; }',
    'mkdir -p -- "$DIR" || { echo "RULZ_ERR=Could not create the target folder."; exit 3; }',
    'DIRABS="$(cd -- "$DIR" && pwd -P)"',
    'SRCDIR="$(cd -- "$(dirname -- "$SRC")" && pwd -P)"',
    'BASE="$(basename -- "$SRC")"',
    'if [ "$SRCDIR" = "$DIRABS" ]; then',
    '  echo "RULZ_MOVED=$SRC"',
    '  [ -f "$SRC.recipe.json" ] && echo "RULZ_SIDECAR=$SRC.recipe.json"',
    '  exit 0',
    'fi',
    'STEM="${BASE%.safetensors}"',
    'N=1',
    'TARGET="$DIRABS/$BASE"',
    'while [ -e "$TARGET" ] || [ -L "$TARGET" ] || [ -e "$TARGET.recipe.json" ]; do',
    '  N=$((N+1))',
    '  TARGET="$DIRABS/$STEM-$N.safetensors"',
    'done',
    'mv -n -- "$SRC" "$TARGET" || { echo "RULZ_ERR=The move failed."; exit 4; }',
    'if [ -e "$SRC" ] || [ ! -f "$TARGET" ]; then echo "RULZ_ERR=The move could not be verified."; exit 4; fi',
    'if [ -f "$SRC.recipe.json" ] && [ ! -e "$TARGET.recipe.json" ]; then',
    '  mv -n -- "$SRC.recipe.json" "$TARGET.recipe.json" && echo "RULZ_SIDECAR=$TARGET.recipe.json"',
    'fi',
    'echo "RULZ_MOVED=$TARGET"',
    'true',
    ''
  ].join('\n')
}

export async function moveNoOverwrite(
  wsl: WslRunner,
  distro: string,
  from: string,
  toDir: string
): Promise<{ path: string; sidecar: string | null }> {
  const r = await wsl.run(distro, buildMoveScript(from, toDir), { timeoutMs: 600000 })
  const err = /RULZ_ERR=([^\r\n]*)/.exec(r.stdout)
  if (err) throw new Error(err[1])
  const moved = /RULZ_MOVED=([^\r\n]+)/.exec(r.stdout)
  if (r.code !== 0 || !moved) throw new Error(`The file was not moved: ${r.stderr.trim() || 'WSL did not answer'}`)
  const side = /RULZ_SIDECAR=([^\r\n]+)/.exec(r.stdout)
  return { path: moved[1], sidecar: side ? side[1] : null }
}
