import { execFileSync } from 'node:child_process'
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join } from 'node:path'
import { afterAll, describe, expect, it } from 'vitest'
import { checkJson, checkSyntax, describeNodeOutput, describePythonOutput, findProgram, looksLikeModule, projectPythonVersion } from '../src/main/agent/syntaxCheck'

const root = mkdtempSync(join(tmpdir(), 'rulz-agent-syntax-'))
afterAll(() => rmSync(root, { recursive: true, force: true }))

/** The Node.js running the tests stands in for the computer's own. */
const withNode = (name: string): string | null => (name === 'node' ? process.execPath : null)
const nothing = (): string | null => null

const python = await findProgram('python', root)
const pythonWorks = (() => {
  if (!python) return false
  try {
    return /Python 3/.test(execFileSync(python, ['--version'], { encoding: 'utf8', timeout: 20_000 }))
  } catch {
    return false
  }
})()

async function check(rel: string, content: string, find: (name: string) => string | null = withNode): Promise<string | null> {
  const abs = join(root, rel)
  mkdirSync(dirname(abs), { recursive: true })
  writeFileSync(abs, content)
  return checkSyntax(abs, rel, { root, find })
}

describe('the syntax check after a file is written', () => {
  it('passes files it does not check, and files that are fine', async () => {
    expect(await check('index.html', '<h1>Hi<h1')).toBeNull()
    expect(await check('style.css', 'body {')).toBeNull()
    expect(await check('app.ts', 'const a: number = ;')).toBeNull()
    expect(await check('package.json', '{ "name": "app", "scripts": { "start": "electron ." } }\n')).toBeNull()
    expect(await check('main.js', "const { app } = require('electron')\napp.whenReady().then(() => console.log('hi'))\n")).toBeNull()
  }, 30_000)

  it('reports a broken JSON file with the line', async () => {
    const problem = await check('package.json', '{\n  "name": "app",\n  "scripts": { "start": "electron ." },\n}\n')
    expect(problem).toMatch(/^Syntax check: package\.json has an error/)
    expect(problem).toMatch(/line 4/)
    expect(problem).toMatch(/Fix it now/)
  }, 30_000)

  it('leaves JSON files that may hold comments alone', async () => {
    const withComments = '{\n  // comment\n  "compilerOptions": { "strict": true, },\n}\n'
    expect(await check('tsconfig.json', withComments)).toBeNull()
    expect(await check('web/tsconfig.node.json', withComments)).toBeNull()
    expect(await check('.vscode/settings.json', withComments)).toBeNull()
    expect(await check('settings.jsonc', withComments)).toBeNull()
    expect(await check('angular.json', '{ "version": 1, }')).toBeNull()
    expect(await check('apps/web/biome.json', '{ "files": {}, }')).toBeNull()
    // Any other JSON file with a comment in it may be read by a tool that allows them.
    expect(await check('data/config.json', withComments)).toBeNull()
    expect(await check('data/block.json', '{ /* note */ "a": 1 }')).toBeNull()
    // Without a comment it is an error, and "//" inside a text is no comment.
    expect(await check('data/list.json', '{ "a": 1, }')).toMatch(/Syntax check/)
    expect(await check('data/link.json', '{ "url": "https://example.com/a", }')).toMatch(/Syntax check/)
    // npm takes no comments: there it is always an error.
    expect(await check('package.json', '{\n  // the name\n  "name": "app"\n}\n')).toMatch(/Syntax check/)
    expect(await check('web/package.json', '{\n  // the name\n  "name": "app"\n}\n')).toMatch(/Syntax check/)
  }, 30_000)

  it('accepts a JSON file that starts with a byte order mark', async () => {
    expect(await check('bom.json', `${String.fromCharCode(0xfeff)}{"a": 1}`)).toBeNull()
  }, 30_000)

  it('reports a broken script with its line and the code, without needing Node.js', async () => {
    const problem = await check('main.js', 'const a = 1;\nfoo(;\n', nothing)
    expect(problem).toMatch(/Syntax check: main\.js has an error/)
    expect(problem).toMatch(/Line 2: SyntaxError: Unexpected token ';'/)
    expect(problem).toContain('foo(;')
  }, 30_000)

  it('accepts what Node.js accepts in a CommonJS file: a return at the top, a #! line', async () => {
    expect(await check('cli.js', "#!/usr/bin/env node\nif (!process.argv[2]) return\nconsole.log('ok')\n", nothing)).toBeNull()
    expect(await check('tool.cjs', 'if (!process.argv[2]) return\n', nothing)).toBeNull()
    expect(await check('tool2.cjs', 'if (!process.argv[2] return\n', nothing)).toMatch(/Line 1: SyntaxError/)
  }, 30_000)

  it('checks modules with Node.js: a good one passes, a broken one is reported', async () => {
    expect(await check('src/app.js', "import { a } from './a.js'\nexport const b = a + 1\n")).toBeNull()
    expect(await check('src/top.js', "const data = await fetch('/api')\nconsole.log(data)\n")).toBeNull()
    expect(await check('src/lib.mjs', 'export default function f() {}\n')).toBeNull()
    const problem = await check('src/bad.js', "import { a } from './a.js'\nexport const b = ;\n")
    expect(problem).toMatch(/Syntax check: src\/bad\.js has an error/)
    expect(problem).toMatch(/Line 2: SyntaxError: Unexpected token ';'/)
    expect(await check('src/bad.mjs', 'export const a = ;\n')).toMatch(/Line 1: SyntaxError/)
  }, 30_000)

  it('says nothing about a module when there is no Node.js to check it', async () => {
    expect(await check('src/app2.js', "import { a } from './a.js'\nexport const b = ;\n", nothing)).toBeNull()
    expect(await check('src/lib2.mjs', 'export const a = ;\n', nothing)).toBeNull()
  }, 30_000)

  it('says nothing about JSX in a .js file, which some projects are set up for', async () => {
    expect(await check('src/App.js', 'export default function App() {\n  return <div>Hi</div>\n}\n')).toBeNull()
    expect(await check('src/view.js', 'const el = <div>Hi</div>\n', nothing)).toBeNull()
  }, 30_000)

  it("says nothing for sure about a web page's script that uses names Node.js keeps for itself", async () => {
    // Fine in a page, an error only the way Node.js reads a CommonJS file.
    const page = 'const module = { ready: false }\nlet exports = []\nconsole.log(module, exports)\n'
    expect(await check('web/page.js', page, nothing)).toBeNull()
    expect(await check('web/page.js', page)).toBeNull()
    expect(await check('web/loop.js', 'for await (const x of stream) console.log(x)\n', nothing)).toBeNull()
    // With Node.js to read it as a module, a real mistake in such a file is still found, and it is the one shown.
    const problem = await check('web/broken.js', `${page}foo(;\n`)
    expect(problem).toMatch(/Line 4: SyntaxError: Unexpected token ';'/)
    expect(problem).not.toContain('already been declared')
    // In a .cjs file the name really is taken.
    expect(await check('web/taken.cjs', page, nothing)).toMatch(/already been declared/)
  }, 30_000)

  it('says nothing about import attributes, which an older Node.js does not know', async () => {
    // This Node.js dropped the old "assert" spelling, the way an older one doesn't know "with".
    expect(await check('src/data.js', "import data from './data.json' assert { type: 'json' }\nexport default data\n")).toBeNull()
  }, 30_000)

  it('never uses a program inside the project to do the checking', async () => {
    const bin = join(root, 'tools')
    mkdirSync(bin, { recursive: true })
    const exe = process.platform === 'win32' ? 'node.exe' : 'node'
    writeFileSync(join(bin, exe), '')
    const outside = mkdtempSync(join(tmpdir(), 'rulz-agent-syntax-path-'))
    try {
      writeFileSync(join(outside, exe), '')
      const sep = process.platform === 'win32' ? ';' : ':'
      // A relative entry, the project's folder and the project itself are passed over.
      expect(await findProgram('node', root, { PATH: ['.', 'tools', bin, root, outside].join(sep) })).toBe(join(outside, exe))
      expect(await findProgram('node', root, { PATH: ['.', bin].join(sep) })).toBeNull()
      // A network share isn't looked at: it can hang for minutes when it is away.
      expect(await findProgram('node', root, { PATH: ['\\\\127.0.0.1\\nothing-here\\bin', '//127.0.0.1/nothing-here/bin'].join(sep) })).toBeNull()
    } finally {
      rmSync(outside, { recursive: true, force: true })
    }
  }, 30_000)

  it.runIf(pythonWorks)('reports a broken Python file, and passes a good one', async () => {
    const find = (name: string): string | null => (name === 'python' ? python : null)
    expect(await check('main.py', 'import tkinter as tk\n\ndef main():\n    root = tk.Tk()\n    root.mainloop()\n\nmain()\n', find)).toBeNull()
    const problem = await check('app.py', 'def main(:\n    pass\n', find)
    expect(problem).toMatch(/Syntax check: app\.py has an error/)
    expect(problem).toMatch(/Line 1: SyntaxError/)
    expect(problem).toContain('def main(:')
    expect(await check('indent.py', 'def main():\npass\n', find)).toMatch(/Line 2: SyntaxError/)
    // The project runs on a newer Python (its own virtual environment) than the one that checked:
    // what looks broken here may be new syntax there, so nothing is said.
    mkdirSync(join(root, 'newer', '.venv'), { recursive: true })
    writeFileSync(join(root, 'newer', '.venv', 'pyvenv.cfg'), 'home = C:\\Python\nversion = 99.1.0\n')
    expect(await check('newer/src/app.py', 'def main(:\n    pass\n', find)).toBeNull()
    // An environment on an older Python changes nothing.
    mkdirSync(join(root, 'older', 'venv'), { recursive: true })
    writeFileSync(join(root, 'older', 'venv', 'pyvenv.cfg'), 'version_info = 3.0.1.final.0\n')
    expect(await check('older/app.py', 'def main(:\n    pass\n', find)).toMatch(/Line 1: SyntaxError/)
  }, 30_000)

  it("reads the Python version of the project's virtual environment, from the file's folder upwards", async () => {
    mkdirSync(join(root, 'svc', '.venv'), { recursive: true })
    writeFileSync(join(root, 'svc', '.venv', 'pyvenv.cfg'), 'home = C:\\Python312\r\ninclude-system-site-packages = false\r\nversion = 3.12.10\r\n')
    expect(await projectPythonVersion(root, join(root, 'svc', 'pkg', 'deep', 'mod.py'))).toEqual([3, 12])
    expect(await projectPythonVersion(root, join(root, 'elsewhere', 'mod.py'))).toBeNull()
    // Never looks above the project folder.
    expect(await projectPythonVersion(join(root, 'svc', 'pkg'), join(root, 'svc', 'pkg', 'mod.py'))).toBeNull()
    expect(describePythonOutput('RULZ_SYNTAX [3, "invalid syntax", "x = = 1", 3, 11]\n', [3, 12])).toBeNull()
    expect(describePythonOutput('RULZ_SYNTAX [3, "invalid syntax", "x = = 1", 3, 12]\n', [3, 12])).toBe('Line 3: SyntaxError: invalid syntax\nx = = 1')
    expect(describePythonOutput('RULZ_SYNTAX [3, "invalid syntax", "x = = 1", 3, 13]\n', [3, 12])).toMatch(/^Line 3/)
  }, 30_000)

  it('says nothing about a Python file when there is no Python', async () => {
    expect(await check('nopython.py', 'def main(:\n    pass\n', nothing)).toBeNull()
  }, 30_000)

  it('reads the checkers output, and takes anything unexpected as no finding', () => {
    expect(describePythonOutput('RULZ_SYNTAX [3, "invalid syntax", "x = = 1"]\n')).toBe('Line 3: SyntaxError: invalid syntax\nx = = 1')
    expect(describePythonOutput('Python was not found; run without arguments to install from the Microsoft Store\n')).toBeNull()
    expect(describePythonOutput('RULZ_SYNTAX not json')).toBeNull()
    expect(describeNodeOutput("[stdin]:2\nexport const b = ;\n                 ^\n\nSyntaxError: Unexpected token ';'\n    at checkSyntax (node:internal/main/check_syntax:1:1)\n\nNode.js v24.0.0\n")).toBe(
      "Line 2: SyntaxError: Unexpected token ';'\nexport const b = ;\n                 ^"
    )
    expect(describeNodeOutput('node: bad option: --input-type=module\n')).toBeNull()
  }, 30_000)

  it('finds the line of a JSON error that only gives a position', () => {
    expect(checkJson('{"a": 1}')).toBeNull()
    expect(checkJson('{\n"a": 1,\n}')).toMatch(/line 3/)
    expect(checkJson('{\n"a": 1, // one\n}')).toMatch(/line 2/)
    expect(checkJson('{\n"a": 1, // one\n}', true)).toBeNull()
    expect(checkJson('{"a": "a \\" // not a comment", }', true)).not.toBeNull()
  }, 30_000)

  it('tells a module from a plain script', () => {
    expect(looksLikeModule("import x from './x.js'\n")).toBe(true)
    expect(looksLikeModule("import './side-effect.js'\n")).toBe(true)
    expect(looksLikeModule('export default 1\n')).toBe(true)
    expect(looksLikeModule('const u = import.meta.url\n')).toBe(true)
    expect(looksLikeModule("const m = await import('./x.js')\n")).toBe(true)
    expect(looksLikeModule("const x = require('x')\nmodule.exports = { x }\n")).toBe(false)
    expect(looksLikeModule("function load() { return import('./x.js') }\n")).toBe(false)
  }, 30_000)
})
