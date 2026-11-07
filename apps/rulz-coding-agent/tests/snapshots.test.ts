import { execFileSync } from 'node:child_process'
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterAll, describe, expect, it } from 'vitest'
import { GitSnapshots, gitInstalled } from '../src/main/project/snapshots'

const base = mkdtempSync(join(tmpdir(), 'rulz-agent-snap-'))
afterAll(() => rmSync(base, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 }))
let n = 0

const git = (cwd: string, ...args: string[]): string =>
  execFileSync('git', ['-c', 'user.name=t', '-c', 'user.email=t@t', '-c', 'commit.gpgsign=false', '-c', 'core.autocrlf=false', ...args], { cwd, encoding: 'utf8' })

function project(): { root: string; snaps: GitSnapshots } {
  const root = join(base, `p${++n}`)
  mkdirSync(root, { recursive: true })
  return { root, snaps: new GitSnapshots(join(base, `snap${n}`), root) }
}

const write = (root: string, rel: string, text: string): void => {
  mkdirSync(join(root, rel, '..'), { recursive: true })
  writeFileSync(join(root, rel), text)
}
const read = (root: string, rel: string): string | null => (existsSync(join(root, rel)) ? readFileSync(join(root, rel), 'utf8') : null)

describe.runIf(await gitInstalled())('git restore points', { timeout: 60_000 }, () => {
  it("puts back files the project's own .gitignore lists, byte for byte", async () => {
    const { root, snaps } = project()
    write(root, '.gitignore', '.env\ndata/\nbuild-output/\n')
    write(root, '.gitattributes', '* text=auto\n')
    write(root, '.env', 'SECRET=1\n')
    write(root, 'data/app.db', 'rows')
    write(root, 'run.sh', 'line1\nline2\n')
    write(root, 'node_modules/pkg/index.js', 'x')
    const snap = await snaps.begin('t1', 'Task one')

    write(root, '.env', 'SECRET=changed\n')
    rmSync(join(root, 'data'), { recursive: true })
    write(root, 'run.sh', 'line1\r\nline2\r\n')
    write(root, '.env.local', 'new')
    const changed = await snaps.changedSince(snap)
    expect(changed).toEqual(
      expect.arrayContaining([
        { path: '.env', change: 'changed' },
        { path: '.env.local', change: 'created' },
        { path: 'data/app.db', change: 'deleted' },
        { path: 'run.sh', change: 'changed' }
      ])
    )
    expect(changed!.some((c) => c.path.startsWith('node_modules/'))).toBe(false)
    expect((await snaps.fileBefore(snap, '.env'))?.toString()).toBe('SECRET=1\n')

    await snaps.restore(snap, 'Task one')
    expect(read(root, '.env')).toBe('SECRET=1\n')
    expect(read(root, 'data/app.db')).toBe('rows')
    expect(read(root, 'run.sh')).toBe('line1\nline2\n')
    expect(read(root, '.env.local')).toBeNull()
    expect(read(root, 'node_modules/pkg/index.js')).toBe('x')
  })

  it('leaves a big folder the project ignores (build output) out, as before', async () => {
    const { root, snaps } = project()
    write(root, '.gitignore', 'bin/\n')
    for (let i = 0; i < 1100; i++) write(root, `bin/f${i}.o`, 'x')
    const snap = await snaps.begin('t1', 'Task one')
    write(root, 'main.c', 'int main(){}')
    expect(await snaps.changedSince(snap)).toEqual([{ path: 'main.c', change: 'created' }])
  })

  it('handles folders with a git repository of their own, with or without commits', async () => {
    const { root, snaps } = project()
    write(root, 'a.txt', 'a')
    write(root, 'web/index.html', 'web before')
    git(join(root, 'web'), 'init', '-q')
    write(root, 'app/main.rs', 'app before')
    git(join(root, 'app'), 'init', '-q')
    git(join(root, 'app'), 'add', '.')
    git(join(root, 'app'), 'commit', '-qm', 'init')
    const snap = await snaps.begin('t1', 'Task one')

    write(root, 'web/index.html', 'web after')
    write(root, 'web/new.js', 'new')
    write(root, 'app/main.rs', 'app after')
    write(root, 'site/index.html', 'site')
    git(join(root, 'site'), 'init', '-q')
    expect(await snaps.changedSince(snap)).toEqual(
      expect.arrayContaining([
        { path: 'app/main.rs', change: 'changed' },
        { path: 'site/index.html', change: 'created' },
        { path: 'web/index.html', change: 'changed' },
        { path: 'web/new.js', change: 'created' }
      ])
    )

    await snaps.restore(snap, 'Task one')
    expect(read(root, 'web/index.html')).toBe('web before')
    expect(read(root, 'web/new.js')).toBeNull()
    expect(read(root, 'app/main.rs')).toBe('app before')
    expect(read(root, 'site/index.html')).toBeNull()
    // Their own repositories are untouched.
    expect(git(join(root, 'app'), 'log', '--oneline')).toMatch(/init/)
    expect(existsSync(join(root, 'web', '.git'))).toBe(true)
  })
})
