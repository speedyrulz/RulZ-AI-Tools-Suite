// WSL helpers for LoRA merges: list the LoRA files of the models folder and make a staging folder.
// Scripts are built by pure functions (testable), quote every path with shellQuote and use LF only.
import { shellQuote, type WslRunner } from '../server/wsl'

/** Lists `<models>/loras/**.safetensors` (symlinks followed); one line per file: `RULZ_LORA|<relative path>`. */
export function buildListLorasScript(modelsAbs: string): string {
  const dir = `${modelsAbs.replace(/\/+$/, '')}/loras`
  return [
    `cd ${shellQuote(dir)} 2>/dev/null || exit 0`,
    'find -L . -type f -name "*.safetensors" -print0 2>/dev/null | while IFS= read -r -d "" f; do',
    '  case "$f" in *[[:cntrl:]]*) continue;; esac',
    '  echo "RULZ_LORA|${f#./}"',
    'done',
    'true',
    ''
  ].join('\n')
}

/** Relative names with "/" separators, sorted; names with a line break or a "." / ".." part are skipped. */
export function parseListLoras(stdout: string): string[] {
  const out = new Set<string>()
  for (const raw of stdout.split('\n')) {
    const line = raw.replace(/\r$/, '')
    if (!line.startsWith('RULZ_LORA|')) continue
    const name = line.slice('RULZ_LORA|'.length)
    if (!name || name.includes('\r') || !name.endsWith('.safetensors')) continue
    if (name.split('/').some((p) => p === '' || p === '.' || p === '..')) continue
    out.add(name)
  }
  return [...out].sort((a, b) => a.localeCompare(b, 'en', { sensitivity: 'base', numeric: true }) || (a < b ? -1 : a > b ? 1 : 0))
}

export async function listLoras(wsl: WslRunner, distro: string, modelsAbs: string): Promise<string[]> {
  const r = await wsl.run(distro, buildListLorasScript(modelsAbs), { timeoutMs: 30000 })
  if (r.code !== 0) throw new Error(`Could not list the LoRA folder: ${r.stderr.trim() || 'WSL did not answer'}`)
  return parseListLoras(r.stdout)
}

/** A leading `~` is left to bash; the rest is quoted. Makes `<baseDir>/lora-merge` and prints its absolute path. */
export function buildStagingScript(baseDir: string): string {
  const b = baseDir.trim().replace(/\/+$/, '')
  let target: string
  if (b === '~') target = '"$HOME"/lora-merge'
  else if (b.startsWith('~/')) target = `"$HOME"/${shellQuote(b.slice(2))}/lora-merge`
  else target = `${shellQuote(b)}/lora-merge`
  return [
    `DIR=${target}`,
    'mkdir -p -- "$DIR" || { echo "RULZ_ERR=Could not create the staging folder."; exit 3; }',
    'cd -- "$DIR" && echo "RULZ_STAGING=$(pwd -P)"',
    'true',
    ''
  ].join('\n')
}

export async function resolveStagingDir(wsl: WslRunner, distro: string, baseDir: string): Promise<string> {
  const r = await wsl.run(distro, buildStagingScript(baseDir), { timeoutMs: 15000 })
  const err = /RULZ_ERR=([^\r\n]*)/.exec(r.stdout)
  if (err) throw new Error(err[1])
  const m = /RULZ_STAGING=(\/[^\r\n]*)/.exec(r.stdout)
  if (r.code !== 0 || !m) throw new Error(`The staging folder was not made: ${r.stderr.trim() || 'WSL did not answer'}`)
  return m[1]
}
