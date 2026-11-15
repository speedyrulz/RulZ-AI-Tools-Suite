// A quick syntax check of a file the agent has just written, so a typo comes back with the write
// itself instead of a whole model turn later, when the app is run.
//
// Nothing from the project is ever run: JSON and plain scripts are parsed inside this app, and the
// rest is handed to the computer's own Node.js or Python (never one found inside the project) to be
// parsed only. Anything that can't be checked for sure is passed without a word: a wrong "this
// file is broken" sends the model off to break a working file.
import { execFile } from 'node:child_process'
import { promises as fs } from 'node:fs'
import { tmpdir } from 'node:os'
import { delimiter, dirname, isAbsolute, join, resolve, sep } from 'node:path'
import { compileFunction } from 'node:vm'
import { commandEnv } from './commands'

/** Bigger files are generated or data, and not worth the time. */
const MAX_CHECK_BYTES = 1024 * 1024
const CHECK_TIMEOUT_MS = 10_000
const MAX_DETAIL_CHARS = 700

/** JSON files whose tools take comments and trailing commas, which JSON.parse refuses. */
const JSON_WITH_COMMENTS =
  /(^|\/)(tsconfig[^/]*\.json|jsconfig[^/]*\.json|\.eslintrc(\.json)?|\.babelrc(\.json)?|babel\.config\.json|\.swcrc|\.jshintrc|\.mocharc\.json|devcontainer\.json|angular\.json|turbo\.json|deno\.json|tslint\.json|biome\.json|cspell\.json|wrangler\.json|renovate\.json|typedoc\.json|pyrightconfig\.json|api-extractor\.json|rush\.json|\.luarc\.json|language-configuration\.json|[^/]*\.code-workspace|[^/]*\.jsonc)$|(^|\/)\.vscode\//i

/** JSON files where a comment is certainly an error (npm refuses them). */
const JSON_STRICT = /(^|\/)(package|package-lock)\.json$/i

type Find = (name: string) => string | null | Promise<string | null>

export interface SyntaxCheckOptions {
  /** The project folder: programs inside it are never used to do the checking. */
  root: string
  signal?: AbortSignal
  /** Finds a program by name; for tests. */
  find?: Find
}

/**
 * Checks a file's syntax by its extension. Returns the problem as text for the model, or null when
 * the file is fine, isn't a kind that is checked, or couldn't be checked.
 */
export async function checkSyntax(abs: string, rel: string, opts: SyntaxCheckOptions): Promise<string | null> {
  const ext = /\.([a-z0-9]+)$/i.exec(rel)?.[1].toLowerCase() ?? ''
  if (!['json', 'js', 'mjs', 'cjs', 'py', 'pyw'].includes(ext)) return null
  if (ext === 'json' && JSON_WITH_COMMENTS.test(rel.replace(/\\/g, '/'))) return null
  let source: string
  try {
    const st = await fs.stat(abs)
    if (!st.isFile() || st.size > MAX_CHECK_BYTES) return null
    const buf = await fs.readFile(abs)
    if (buf.subarray(0, 8000).includes(0)) return null
    const text = buf.toString('utf8')
    source = text.charCodeAt(0) === 0xfeff ? text.slice(1) : text
  } catch {
    return null
  }
  const find: Find = opts.find ?? ((name) => findProgram(name, opts.root))
  let detail: string | null
  try {
    if (ext === 'json') detail = checkJson(source, !JSON_STRICT.test(rel.replace(/\\/g, '/')))
    else if (ext === 'py' || ext === 'pyw') detail = await checkPython(abs, source, opts.root, find, opts.signal)
    else detail = await checkJavaScript(source, ext, find, opts.signal)
  } catch {
    return null
  }
  if (!detail || opts.signal?.aborted) return null
  const shown = detail.length > MAX_DETAIL_CHARS ? `${detail.slice(0, MAX_DETAIL_CHARS)}…` : detail
  return (
    `Syntax check: ${rel} has an error and will not work as it is.\n${shown}\n` +
    'Fix it now, before anything else (read the lines around it with read_file if you need to).'
  )
}

// ---------- JSON ----------

/**
 * `commentsMayBeMeant`: a file with comments in it is passed over. Many tools read their JSON
 * settings with comments allowed, and the list of them above can't be complete.
 */
export function checkJson(source: string, commentsMayBeMeant = false): string | null {
  try {
    JSON.parse(source)
    return null
  } catch (e) {
    if (commentsMayBeMeant && hasJsonComment(source)) return null
    const message = (e as Error).message
    // Older engines give only a position.
    const pos = /position (\d+)/.exec(message)
    if (pos && !/line \d+/.test(message)) return `${message} (line ${lineAt(source, Number(pos[1]))})`
    return message
  }
}

const lineAt = (text: string, index: number): number => text.slice(0, Math.max(0, index)).split('\n').length

/** Whether the text has a // or a slash-star comment outside its strings. */
function hasJsonComment(source: string): boolean {
  let inString = false
  for (let i = 0; i < source.length; i++) {
    const c = source[i]
    if (inString) {
      if (c === '\\') i++
      else if (c === '"') inString = false
    } else if (c === '"') inString = true
    else if (c === '/' && (source[i + 1] === '/' || source[i + 1] === '*')) return true
  }
  return false
}

// ---------- JavaScript ----------

/** Lines that only a module may have. */
const MODULE_SYNTAX = /^\s*(import\s*(["'{*]|[\w$]+\s*(,|from\b))|export\s+(default\b|const\b|let\b|var\b|function\b|async\b|class\b|\{|\*))|\bimport\.meta\b|^\s*(const|let|var)\s[^=\n]*=\s*await\b|^\s*await\b/m

/** Whether the text reads as a module (import / export lines, import.meta, await outside a function). */
export const looksLikeModule = (source: string): boolean => MODULE_SYNTAX.test(source)

/**
 * What the parse here says about text that may be fine as a module or as a web page's script:
 * module-only syntax, and a page script's own `module` or `require` (taken names here, because the
 * text is parsed the way Node.js reads a CommonJS file). Such an error isn't reported as it is.
 */
const MODULE_ONLY_ERROR =
  /outside a module|Unexpected token 'export'|import\.meta|await is only valid|Unexpected reserved word|Identifier '(exports|require|module|__filename|__dirname)' has already been declared/

/** Syntax an older Node.js doesn't know yet (import attributes), while the project's own tools may. */
const NEWER_SYNTAX_ERROR = /Unexpected (token|identifier) '(with|assert)'/

/** The names Node.js gives a CommonJS file: it runs as the body of a function, so a `return` at the top is fine. */
const COMMONJS_PARAMS = ['exports', 'require', 'module', '__filename', '__dirname']

/** "name:LINE", the code line and the caret from a syntax error's stack, then its message. */
function describeScriptError(e: Error): string {
  const head = (e.stack ?? '').split('\n')
  const at = head.findIndex((l) => /^\s*SyntaxError\b/.test(l))
  const before = (at > 0 ? head.slice(0, at) : []).filter((l) => l.trim())
  const line = /:(\d+)$/.exec(before[0] ?? '')?.[1]
  const code = before.slice(1).join('\n')
  return `${line ? `Line ${line}: ` : ''}${e.name}: ${e.message}${code ? `\n${code}` : ''}`
}

/** The same from what `node --check` prints. */
export function describeNodeOutput(stderr: string): string | null {
  const lines = stderr.replace(/\r/g, '').split('\n')
  const at = lines.findIndex((l) => /^SyntaxError\b/.test(l))
  if (at < 0) return null
  const before = lines.slice(0, at).filter((l) => l.trim())
  const line = /:(\d+)$/.exec(before[0] ?? '')?.[1]
  const code = before.slice(1).join('\n')
  return `${line ? `Line ${line}: ` : ''}${lines[at].trim()}${code ? `\n${code}` : ''}`
}

/** A "<" where code was expected is how JSX in a .js file reads, which is fine in projects set up for it. */
const maybeJsx = (detail: string): boolean => /Unexpected token '<'/.test(detail)

/**
 * A script that parses here is fine. One that doesn't is either broken or a module (import /
 * export), which can't be parsed in this process; Node.js then checks it as a module, and only a
 * file that fails both ways is reported.
 */
async function checkJavaScript(source: string, ext: string, find: Find, signal?: AbortSignal): Promise<string | null> {
  let asScript: string | null = null
  if (ext !== 'mjs') {
    try {
      compileFunction(source, COMMONJS_PARAMS, { filename: 'file.js' })
      return null
    } catch (e) {
      if ((e as Error)?.name !== 'SyntaxError') return null
      asScript = describeScriptError(e as Error)
    }
    if (ext === 'cjs') return maybeJsx(asScript) ? null : asScript
  }
  // The script error is the one to show unless the file is written as a module.
  const scriptError = asScript && !looksLikeModule(source) && !MODULE_ONLY_ERROR.test(asScript) && !maybeJsx(asScript) ? asScript : null
  const node = await find('node')
  // Without Node.js a module can't be checked; a plain script's error stands.
  if (!node) return scriptError
  const run = await runChecker(node, ['--input-type=module', '--check', '-'], source, signal)
  // Stopped, timed out or not started: nothing is known about the file as a module.
  if (!run) return scriptError
  if (run.code === 0) return null
  const asModule = describeNodeOutput(run.stderr)
  // No syntax error in what it printed: Node.js itself failed (too old for this, say).
  if (!asModule) return scriptError
  if (scriptError) return scriptError
  return maybeJsx(asModule) || NEWER_SYNTAX_ERROR.test(asModule) ? null : asModule
}

// ---------- Python ----------

const PYTHON_MARK = 'RULZ_SYNTAX '
/** Parses the file (never runs it) and prints a syntax error as one marked line. */
const PYTHON_SCRIPT = [
  'import ast, json, sys',
  'try:',
  "    with open(sys.argv[1], 'rb') as f:",
  '        ast.parse(f.read(), sys.argv[1])',
  'except SyntaxError as e:',
  `    print('${PYTHON_MARK}' + json.dumps([e.lineno, e.msg, (e.text or '').rstrip(), sys.version_info[0], sys.version_info[1]]))`,
  'except Exception:',
  '    pass'
].join('\n')

/**
 * Reads the marked line; anything else (no Python, the Microsoft Store placeholder, a crash) is no
 * finding. `project` is the Python version the project itself runs on, when known: if that is newer
 * than the Python that did the checking, what is an error here may be fine there, so nothing is said.
 */
export function describePythonOutput(stdout: string, project: [number, number] | null = null): string | null {
  const line = stdout.split(/\r?\n/).find((l) => l.startsWith(PYTHON_MARK))
  if (!line) return null
  try {
    const [lineNo, message, text, major, minor] = JSON.parse(line.slice(PYTHON_MARK.length)) as [number | null, string, string, number?, number?]
    if (project && typeof major === 'number' && typeof minor === 'number' && (project[0] > major || (project[0] === major && project[1] > minor))) return null
    return `${lineNo ? `Line ${lineNo}: ` : ''}SyntaxError: ${message}${text?.trim() ? `\n${text}` : ''}`
  } catch {
    return null
  }
}

const VENV_FOLDERS = ['.venv', 'venv', 'env']

/**
 * The Python version of the project's own virtual environment, read from its pyvenv.cfg (a text
 * file; nothing in the environment is run). Looked for next to the file and in each folder above
 * it, up to the project folder.
 */
export async function projectPythonVersion(root: string, abs: string): Promise<[number, number] | null> {
  const top = resolve(root)
  let dir = dirname(resolve(abs))
  for (let depth = 0; depth < 12; depth++) {
    if (dir !== top && !dir.toLowerCase().startsWith(`${top.toLowerCase()}${sep}`)) break
    for (const name of VENV_FOLDERS) {
      const cfg = await fs.readFile(join(dir, name, 'pyvenv.cfg'), 'utf8').catch(() => null)
      const m = cfg ? /^\s*version(?:_info)?\s*=\s*(\d+)\.(\d+)/m.exec(cfg) : null
      if (m) return [Number(m[1]), Number(m[2])]
    }
    if (dir === top) break
    dir = dirname(dir)
  }
  return null
}

async function checkPython(abs: string, source: string, root: string, find: Find, signal?: AbortSignal): Promise<string | null> {
  if (!source.trim()) return null
  const python = await find('python')
  if (!python) return null
  // -I: isolated, so nothing in the project (or a PYTHON* setting) can be loaded in place of the standard library.
  const run = await runChecker(python, ['-I', '-X', 'utf8', '-c', PYTHON_SCRIPT, abs], null, signal)
  if (!run || !describePythonOutput(run.stdout)) return null
  return describePythonOutput(run.stdout, await projectPythonVersion(root, abs))
}

// ---------- running the checkers ----------

function runChecker(program: string, args: string[], stdin: string | null, signal?: AbortSignal): Promise<{ code: number; stdout: string; stderr: string } | null> {
  return new Promise((done) => {
    try {
      const child = execFile(
        program,
        args,
        // Not the project folder: Windows looks for programs (and their DLLs) in the current folder first.
        { cwd: tmpdir(), env: commandEnv(), timeout: CHECK_TIMEOUT_MS, windowsHide: true, maxBuffer: 1024 * 1024, signal },
        (err, stdout, stderr) => {
          const failure = err as (Error & { code?: unknown; killed?: boolean }) | null
          // Timed out, stopped or couldn't start: no finding. A number is the program's own exit code.
          if (failure && (failure.killed || typeof failure.code !== 'number')) return done(null)
          done({ code: failure ? (failure.code as number) : 0, stdout: String(stdout), stderr: String(stderr) })
        }
      )
      child.stdin?.on('error', () => undefined)
      child.stdin?.end(stdin ?? '')
    } catch {
      done(null)
    }
  })
}

const found = new Map<string, string | null>()

/** Asked without blocking: the window must not freeze while a slow disk answers. */
const isThere = (file: string): Promise<boolean> => fs.access(file).then(() => true, () => false)

/**
 * The full path of a program on the computer's PATH. Folders inside the project are skipped, and
 * so are relative PATH entries: the checker must be the computer's own Node.js or Python, not a
 * file the project brought along.
 */
export async function findProgram(name: string, root: string, env: NodeJS.ProcessEnv = commandEnv()): Promise<string | null> {
  const pathKey = Object.keys(env).find((k) => k.toUpperCase() === 'PATH') ?? 'PATH'
  const key = `${name}\n${root}\n${env[pathKey] ?? ''}`
  const known = found.get(key)
  if (known !== undefined && (known === null || (await isThere(known)))) return known
  const inside = `${resolve(root).toLowerCase()}${sep}`
  const names = process.platform === 'win32' ? [`${name}.exe`] : [name, `${name}3`]
  let hit: string | null = null
  search: for (const dir of (env[pathKey] ?? '').split(delimiter)) {
    const clean = dir.trim().replace(/^"|"$/g, '')
    // A network share can take minutes to answer when it is away, and is no place to take a checker from.
    if (!clean || !isAbsolute(clean) || /^[\\/]{2}/.test(clean)) continue
    const full = resolve(clean)
    if (`${full.toLowerCase()}${sep}`.startsWith(inside)) continue
    for (const n of names) {
      const candidate = join(full, n)
      if (await isThere(candidate)) {
        hit = candidate
        break search
      }
    }
  }
  found.set(key, hit)
  return hit
}
