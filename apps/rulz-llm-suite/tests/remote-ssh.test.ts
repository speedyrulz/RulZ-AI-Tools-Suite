// src/main/remote/ssh.ts: command lines, quoting, error sentences and the runner with a fake ssh process.
import { spawnSync, type ChildProcess } from 'node:child_process'
import { EventEmitter } from 'node:events'
import { existsSync, mkdirSync, mkdtempSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { PassThrough } from 'node:stream'
import { describe, expect, it } from 'vitest'
import type { RentalDirectory, RentalInfo, SshRoute } from '../src/main/remote/contract'
import {
  assertFileName,
  assertInstanceId,
  assertSlug,
  assertTag,
  buildSshArgs,
  describeSshFailure,
  findSsh,
  frameScript,
  parseMarkers,
  REMOTE_COMMAND,
  shq,
  SshError,
  SshRunner,
  sshKeyInfo
} from '../src/main/remote/ssh'

const direct: SshRoute = { route: 'direct', host: '203.0.113.7', port: 41022 }
const proxy: SshRoute = { route: 'proxy', host: 'ssh4.vast.ai', port: 12345 }

describe('shq', () => {
  it('wraps in single quotes and escapes single quotes', () => {
    expect(shq('abc')).toBe("'abc'")
    expect(shq("it's")).toBe("'it'\\''s'")
    expect(shq('')).toBe("''")
    expect(shq(42)).toBe("'42'")
  })

  it('leaves no way out of the quotes', () => {
    for (const v of ["'; rm -rf / #", '$(id)', '`id`', '\n; id', "'\"'\"'", "\\'", "a'b'c'd"]) {
      const q = shq(v)
      // Undo the quoting the way a shell would: the result must be the value again.
      const unquoted = q
        .slice(1, -1)
        .split("'\\''")
        .join("'")
      expect(unquoted).toBe(v)
      expect(q.startsWith("'") && q.endsWith("'")).toBe(true)
      // Every quote inside is part of an escaped group.
      expect(q.slice(1, -1).replace(/'\\''/g, '')).not.toContain("'")
    }
  })

  it('refuses NUL and non-integers', () => {
    expect(() => shq('a\0b')).toThrow()
    expect(() => shq(1.5)).toThrow()
    expect(() => shq(Number.NaN)).toThrow()
  })
})

describe('validation', () => {
  it('accepts slugs, tags, instance ids and file names of the allowed shape only', () => {
    expect(assertSlug('my-model_1.5')).toBe('my-model_1.5')
    for (const bad of ['', 'Upper', '../x', 'a/b', '-lead', 'a b', "a'b", 'x'.repeat(61)]) expect(() => assertSlug(bad), bad).toThrow()
    expect(assertTag('b12407')).toBe('b12407')
    for (const bad of ['', 'b1', 'b12', '12407', 'b12407x', 'b123456789', 'B12407', 'b12407; id']) expect(() => assertTag(bad), bad).toThrow()
    expect(assertInstanceId('3f2a9c10-aaaa-bbbb-cccc-1234567890ab')).toBeTruthy()
    for (const bad of ['', '../x', 'a b', "a'b", '-x', 'a/b']) expect(() => assertInstanceId(bad), bad).toThrow()
    expect(assertFileName('model-00001-of-00002.gguf')).toBeTruthy()
    for (const bad of ['', '.hidden', 'a/b.gguf', '..', 'a..b', 'a.part', 'a b.gguf', "a'b", 'a\nb']) expect(() => assertFileName(bad), bad).toThrow()
  })
})

describe('buildSshArgs', () => {
  const base = { route: direct, knownHostsFile: 'C:\\Users\\u\\AppData\\Roaming\\RulZ LLM\\ssh\\known_hosts_7' }

  it('sets the fixed options, the rental known_hosts file with forward slashes and quotes it when it has a space', () => {
    const a = buildSshArgs({ ...base, command: REMOTE_COMMAND })
    const joined = a.join(' ')
    for (const o of ['ConnectTimeout=15', 'ServerAliveInterval=15', 'ServerAliveCountMax=3', 'LogLevel=ERROR', 'StrictHostKeyChecking=accept-new', 'BatchMode=yes']) expect(a).toContain(o)
    expect(a).toContain('UserKnownHostsFile="C:/Users/u/AppData/Roaming/RulZ LLM/ssh/known_hosts_7"')
    expect(joined).not.toContain('IdentitiesOnly')
    expect(a.slice(a.indexOf('-p'), a.indexOf('-p') + 2)).toEqual(['-p', '41022'])
    expect(a).toContain('root@203.0.113.7')
    expect(a[a.length - 1]).toBe(REMOTE_COMMAND)
    expect(a).toContain('-T')
    expect(a).not.toContain('-t')
    expect(a).not.toContain('-tt')
    expect(a).not.toContain('-N')
    expect(a).not.toContain('ExitOnForwardFailure=yes')
  })

  it('does not quote a path without spaces', () => {
    expect(buildSshArgs({ route: direct, knownHostsFile: 'C:\\ssh\\kh_1' })).toContain('UserKnownHostsFile=C:/ssh/kh_1')
  })

  it('builds a tunnel with ExitOnForwardFailure and no remote command', () => {
    const a = buildSshArgs({ ...base, tunnel: { localPort: 50123, remotePort: 31337 } })
    expect(a).toContain('-N')
    expect(a).toContain('ExitOnForwardFailure=yes')
    expect(a[a.indexOf('-L') + 1]).toBe('127.0.0.1:50123:127.0.0.1:31337')
    expect(a[a.length - 1]).toBe('root@203.0.113.7')
  })

  it('refuses a host or port that could be read as an option', () => {
    for (const host of ['-oProxyCommand=calc', 'a b', 'host;id', '', 'x'.repeat(300), '$(id)']) expect(() => buildSshArgs({ ...base, route: { ...direct, host } }), host).toThrow()
    for (const port of [0, -1, 70000, 1.5, Number.NaN]) expect(() => buildSshArgs({ ...base, route: { ...direct, port } }), String(port)).toThrow()
    expect(() => buildSshArgs({ ...base, knownHostsFile: 'C:\\a"b\\kh' })).toThrow()
    expect(buildSshArgs({ ...base, route: { route: 'direct', host: 'ssh4.vast.ai', port: 22 } })).toContain('root@ssh4.vast.ai')
  })

  it('never carries a script or data in the arguments', () => {
    const args = buildSshArgs({ ...base, command: REMOTE_COMMAND })
    expect(args.join(' ').length).toBeLessThan(500)
    expect(REMOTE_COMMAND).toBe('bash --noprofile --norc -s')
  })
})

describe('findSsh', () => {
  it('prefers ssh on PATH', () => {
    const p = findSsh({ platform: 'win32', env: { Path: 'C:\\x;C:\\Program Files\\Git\\usr\\bin' }, exists: (f) => f === 'C:\\Program Files\\Git\\usr\\bin\\ssh.exe' })
    expect(p).toBe('C:\\Program Files\\Git\\usr\\bin\\ssh.exe')
  })

  it('falls back to the OpenSSH folder of Windows when ssh is not on PATH', () => {
    const p = findSsh({ platform: 'win32', env: { PATH: 'C:\\x', SystemRoot: 'D:\\Win' }, exists: (f) => f === 'D:\\Win\\System32\\OpenSSH\\ssh.exe' })
    expect(p).toBe('D:\\Win\\System32\\OpenSSH\\ssh.exe')
    expect(findSsh({ platform: 'win32', env: { PATH: '' }, exists: (f) => f === 'C:\\Windows\\System32\\OpenSSH\\ssh.exe' })).toBe('C:\\Windows\\System32\\OpenSSH\\ssh.exe')
  })

  it('says how to get ssh when there is none', () => {
    try {
      findSsh({ platform: 'win32', env: { PATH: '' }, exists: () => false })
      expect.unreachable()
    } catch (err) {
      expect(err).toBeInstanceOf(SshError)
      expect((err as SshError).kind).toBe('missing-ssh')
      expect((err as SshError).message).toContain('OpenSSH Client')
    }
  })

  it('works with POSIX paths', () => {
    expect(findSsh({ platform: 'linux', env: { PATH: '/a:/usr/bin' }, exists: (f) => f === '/usr/bin/ssh' })).toBe('/usr/bin/ssh')
  })
})

describe('frameScript', () => {
  it('wraps the script in a function and calls it on one line, so data after it stays unread', () => {
    const f = frameScript('echo hi\r\nread -r x')
    expect(f).toBe('__rz_main() {\necho hi\nread -r x\n}\n__rz_main; exit $?\n')
  })
})

describe('describeSshFailure', () => {
  it('explains a refused key', () => {
    const e = describeSshFailure('root@1.2.3.4: Permission denied (publickey).', 'rulz-llm-ab12')
    expect(e.kind).toBe('auth')
    expect(e.message).toContain('Keys on cloud.vast.ai')
    expect(e.message).toContain('ssh-agent')
    expect(e.message).toContain('rulz-llm-ab12')
  })

  it('explains an instance that does not answer yet', () => {
    for (const text of ['ssh: connect to host 1.2.3.4 port 22: Connection refused', 'Connection timed out during banner exchange', 'kex_exchange_identification: read: Connection reset by peer', 'ssh: connect to host x port 1: No route to host']) {
      const e = describeSshFailure(text, 'it')
      expect(e.kind, text).toBe('unreachable')
      expect(e.message).toContain('still be starting')
    }
  })

  it('explains a changed host key and offers forgetting it', () => {
    const e = describeSshFailure('@@@ WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED! @@@\nHost key verification failed.', 'it')
    expect(e.kind).toBe('hostkey')
    expect(e.message).toContain('Forget host key')
    expect(e.message).toContain('new key')
  })

  it('gives the last line for anything else', () => {
    const e = describeSshFailure('weird\nthing went wrong', 'it')
    expect(e.kind).toBe('transport')
    expect(e.message).toContain('thing went wrong')
  })
})

// ---------- The runner with a fake ssh ----------

class FakeChild extends EventEmitter {
  stdin = new PassThrough()
  stdout = new PassThrough()
  stderr = new PassThrough()
  received: Buffer[] = []
  killed = false
  stdinEnded = false
  constructor() {
    super()
    this.stdin.on('data', (c: Buffer) => this.received.push(c))
    this.stdin.on('end', () => (this.stdinEnded = true))
  }
  kill(): boolean {
    this.killed = true
    setImmediate(() => this.emit('close', null))
    return true
  }
  finish(code: number | null, out = '', err = ''): void {
    if (out) this.stdout.write(out)
    if (err) this.stderr.write(err)
    setImmediate(() => this.emit('close', code))
  }
  get input(): string {
    return Buffer.concat(this.received).toString('latin1')
  }
}

interface Call {
  exe: string
  args: string[]
  options: Record<string, unknown>
  child: FakeChild
}

function setup(infoFor: (id: number) => RentalInfo | null, behave: (call: Call, n: number) => void) {
  const dir = mkdtempSync(join(tmpdir(), 'rulz-ssh-'))
  const calls: Call[] = []
  const notes: string[] = []
  const directory: RentalDirectory = { info: infoFor, use: () => () => undefined, note: (_id, line) => void notes.push(line) }
  const runner = new SshRunner({
    directory,
    knownHostsDir: join(dir, 'ssh'),
    sshPath: () => 'ssh-test',
    spawn: (exe, args, options) => {
      const child = new FakeChild()
      const call: Call = { exe, args, options: options as Record<string, unknown>, child }
      calls.push(call)
      setImmediate(() => behave(call, calls.length - 1))
      return child as unknown as ChildProcess
    }
  })
  return { runner, calls, notes, dir }
}

const info = (over: Partial<RentalInfo> = {}): RentalInfo => ({ rentalId: 7, label: 'rulz-llm-ab12', origin: 'app', running: true, routes: [direct, proxy], engineTag: 'b12407', dollarsPerHour: 0.5, ...over })

describe('SshRunner.run', () => {
  it('sends the script on stdin to bash -s, never on the command line, without a shell or a window', async () => {
    const big = `echo start\n${'# padding line to make the script long\n'.repeat(3000)}echo end`
    const { runner, calls } = setup(() => info(), (c) => c.child.finish(0, '@rz DONE=1\n'))
    const r = await runner.run(7, big)
    expect(r.code).toBe(0)
    expect(r.stdout).toContain('@rz DONE=1')
    const c = calls[0]
    expect(c.exe).toBe('ssh-test')
    expect(c.args.join(' ').length).toBeLessThan(500)
    expect(c.args[c.args.length - 1]).toBe('bash --noprofile --norc -s')
    expect(c.args.join(' ')).not.toContain('echo start')
    expect(c.options.windowsHide).toBe(true)
    expect(c.options.shell).toBeUndefined()
    expect(c.child.input).toBe(frameScript(big))
    expect(c.child.input.length).toBeGreaterThan(100_000)
    expect(c.child.stdinEnded).toBe(true)
  })

  it('puts data after the script on stdin', async () => {
    const { runner, calls } = setup(() => info(), (c) => c.child.finish(0))
    await runner.run(7, 'read -r x', { stdin: 'header = "x"\n' })
    expect(calls[0].child.input).toBe(`${frameScript('read -r x')}header = "x"\n`)
    await runner.run(7, 'cat', { stdin: Buffer.from([0, 1, 255]) })
    expect(calls[1].child.input.endsWith('\0\u0001\u00ff')).toBe(true)
  })

  it('streams a Readable after the script', async () => {
    const { runner, calls } = setup(() => info(), (c) => setTimeout(() => c.child.finish(0), 30))
    const src = new PassThrough()
    const p = runner.run(7, 'cat >>f', { stdin: src, timeoutMs: 0 })
    src.write('chunk1')
    src.end('chunk2')
    await p
    expect(calls[0].child.input).toBe(`${frameScript('cat >>f')}chunk1chunk2`)
  })

  it('returns a script failure as an exit code, not as a transport error', async () => {
    const { runner } = setup(() => info(), (c) => c.child.finish(3, '@rz ERR=no space\n', 'bash: oops'))
    const r = await runner.run(7, 'exit 3')
    expect(r.code).toBe(3)
    expect(r.stderr).toContain('oops')
  })

  it('treats exit 255 as a transport failure with a sentence', async () => {
    const { runner } = setup(() => info({ routes: [direct] }), (c) => c.child.finish(255, '', 'ssh: connect to host 203.0.113.7 port 41022: Connection refused'))
    await expect(runner.run(7, 'true')).rejects.toMatchObject({ kind: 'unreachable' })
  })

  it('does not try another route after a refused key', async () => {
    const { runner, calls } = setup(() => info(), (c) => c.child.finish(255, '', 'root@x: Permission denied (publickey).'))
    await expect(runner.run(7, 'true')).rejects.toMatchObject({ kind: 'auth' })
    expect(calls).toHaveLength(1)
  })

  it('reports a changed host key and forgets it on request', async () => {
    const { runner, dir } = setup(() => info({ routes: [direct] }), (c) => c.child.finish(255, '', 'Host key verification failed.'))
    await expect(runner.run(7, 'true')).rejects.toMatchObject({ kind: 'hostkey' })
    const file = runner.knownHostsFile(7)
    expect(file).toBe(join(dir, 'ssh', 'known_hosts_7'))
    mkdirSync(join(dir, 'ssh'), { recursive: true })
    writeFileSync(file, 'x')
    runner.forgetHostKey(7)
    expect(existsSync(file)).toBe(false)
    runner.forgetHostKey(7)
  })

  it('tries the routes in order, remembers the one that works and starts with it next time', async () => {
    const { runner, calls } = setup(
      () => info(),
      (c) => {
        if (c.args.includes('root@203.0.113.7')) c.child.finish(255, '', 'Connection timed out')
        else c.child.finish(0, 'ok')
      }
    )
    expect((await runner.run(7, 'true')).stdout).toBe('ok')
    expect(calls.map((c) => c.args[c.args.length - 2])).toEqual(['root@203.0.113.7', 'root@ssh4.vast.ai'])
    expect(runner.currentRoute(7)).toContain('proxy')
    await runner.run(7, 'true')
    expect(calls).toHaveLength(3)
    expect(calls[2].args[calls[2].args.length - 2]).toBe('root@ssh4.vast.ai')
  })

  it('asks the directory again when every route failed, and uses new routes', async () => {
    let asked = 0
    const fresh: SshRoute = { route: 'direct', host: '198.51.100.9', port: 50022 }
    const { runner, calls } = setup(
      () => (++asked <= 1 ? info() : info({ routes: [fresh] })),
      (c) => {
        if (c.args.includes('root@198.51.100.9')) c.child.finish(0, 'new route')
        else c.child.finish(255, '', 'Connection refused')
      }
    )
    expect((await runner.run(7, 'true')).stdout).toBe('new route')
    expect(calls).toHaveLength(3)
    expect(asked).toBeGreaterThanOrEqual(2)
  })

  it('gives up when the new routes are the same', async () => {
    const { runner, calls } = setup(() => info(), (c) => c.child.finish(255, '', 'Connection refused'))
    await expect(runner.run(7, 'true')).rejects.toMatchObject({ kind: 'unreachable' })
    expect(calls).toHaveLength(2)
  })

  it('tries a stream on one route only', async () => {
    const { runner, calls } = setup(() => info(), (c) => c.child.finish(255, '', 'Connection refused'))
    const src = new PassThrough()
    src.end('x')
    await expect(runner.run(7, 'cat', { stdin: src, timeoutMs: 0 })).rejects.toMatchObject({ kind: 'unreachable' })
    expect(calls).toHaveLength(1)
  })

  it('says so when the rental is not running or unknown, and when it has no address', async () => {
    await expect(setup(() => null, () => undefined).runner.run(7, 'true')).rejects.toMatchObject({ kind: 'not-running' })
    await expect(setup(() => info({ running: false }), () => undefined).runner.run(7, 'true')).rejects.toThrow(/not running/)
    await expect(setup(() => info({ routes: [] }), () => undefined).runner.run(7, 'true')).rejects.toMatchObject({ kind: 'unreachable' })
  })

  it('times out and kills ssh', async () => {
    const { runner, calls } = setup(() => info(), () => undefined)
    await expect(runner.run(7, 'sleep 100', { timeoutMs: 20 })).rejects.toMatchObject({ kind: 'timeout' })
    expect(calls[0].child.killed).toBe(true)
  })

  it('aborts on the signal, also one that is already aborted', async () => {
    const { runner, calls } = setup(() => info(), () => undefined)
    const ctl = new AbortController()
    const p = runner.run(7, 'sleep 100', { signal: ctl.signal, timeoutMs: 0 })
    setTimeout(() => ctl.abort(), 10)
    await expect(p).rejects.toMatchObject({ kind: 'aborted' })
    expect(calls[0].child.killed).toBe(true)
    await expect(runner.run(7, 'x', { signal: ctl.signal })).rejects.toMatchObject({ kind: 'aborted' })
  })

  it('delivers output line by line, also the last one without a newline', async () => {
    const lines: string[] = []
    const { runner } = setup(
      () => info(),
      (c) => {
        c.child.stdout.write('@rz A=1\r\n@rz B')
        setImmediate(() => {
          c.child.stdout.write('=2\nlast')
          c.child.finish(0)
        })
      }
    )
    await runner.run(7, 'x', { onLine: (l) => lines.push(l) })
    expect(lines).toEqual(['@rz A=1', '@rz B=2', 'last'])
  })

  it('cuts an endless line instead of filling the memory, and reads the next line again', async () => {
    const lines: string[] = []
    const { runner } = setup(
      () => info(),
      (c) => {
        for (let i = 0; i < 40; i++) c.child.stdout.write('x'.repeat(8 * 1024))
        c.child.stdout.write('\n@rz A=1\n')
        setImmediate(() => c.child.finish(0))
      }
    )
    await runner.run(7, 'x', { onLine: (l) => lines.push(l) })
    expect(lines.map((l) => l.length)).toEqual([64 * 1024, 7])
    expect(lines[1]).toBe('@rz A=1')
  })

  it('reports a missing ssh program', async () => {
    const { runner } = setup(
      () => info(),
      (c) => c.child.emit('error', Object.assign(new Error('spawn ssh ENOENT'), { code: 'ENOENT' }))
    )
    await expect(runner.run(7, 'x')).rejects.toMatchObject({ kind: 'missing-ssh' })
  })

  it('moves a failed route to the end with routeFailed', async () => {
    const { runner, calls } = setup(() => info(), (c) => c.child.finish(0))
    await runner.run(7, 'x')
    expect(calls[0].args).toContain('root@203.0.113.7')
    runner.routeFailed(7)
    await runner.run(7, 'x')
    expect(calls[1].args).toContain('root@ssh4.vast.ai')
  })
})

describe('SshRunner.open and tunnel', () => {
  it('holds stdin open, writes lines and ends it (EOF) on request', async () => {
    const { runner, calls } = setup(() => info(), () => undefined)
    const p = runner.open(7, 'while read -r _; do touch k; done')
    p.write('\n')
    p.write('\n')
    await new Promise((r) => setImmediate(r))
    expect(calls[0].child.input).toBe(`${frameScript('while read -r _; do touch k; done')}\n\n`)
    expect(calls[0].child.stdinEnded).toBe(false)
    p.end()
    await new Promise((r) => setImmediate(r))
    expect(calls[0].child.stdinEnded).toBe(true)
    calls[0].child.finish(0)
    expect((await p.done).code).toBe(0)
  })

  it('reports a dropped connection in done without rejecting', async () => {
    const { runner, calls } = setup(() => info(), () => undefined)
    const p = runner.open(7, 'true')
    calls[0].child.finish(255, '', 'Connection reset by peer')
    const r = await p.done
    expect(r.error?.kind).toBe('unreachable')
  })

  it('builds a tunnel command with -N -L and reports its exit with stderr', async () => {
    const { runner, calls } = setup(() => info(), () => undefined)
    const t = runner.tunnel(7, 50111, 30222)
    const a = calls[0].args
    expect(a).toContain('-N')
    expect(a).toContain('ExitOnForwardFailure=yes')
    expect(a[a.indexOf('-L') + 1]).toBe('127.0.0.1:50111:127.0.0.1:30222')
    calls[0].child.finish(255, '', 'Connection reset')
    expect(await t.exited).toEqual({ code: 255, stderr: 'Connection reset' })
    const t2 = runner.tunnel(7, 50112, 30222)
    t2.kill()
    expect((await t2.exited).code).toBeNull()
  })
})

describe('markers', () => {
  it('parses marker lines only', () => {
    expect(parseMarkers('banner\n@rz A=1\n@rz b=2\n@rz C=x=y\n @rz D=1\n')).toEqual([
      ['A', '1'],
      ['C', 'x=y']
    ])
  })
})

describe('sshKeyInfo', () => {
  const home = (): string => mkdtempSync(join(tmpdir(), 'rulz-home-'))

  it('finds a public key, preferring ed25519', () => {
    const h = home()
    mkdirSync(join(h, '.ssh'))
    writeFileSync(join(h, '.ssh', 'id_rsa.pub'), 'ssh-rsa AAAAB3 rsa@pc\n')
    expect(sshKeyInfo(h).publicKey).toBe('ssh-rsa AAAAB3 rsa@pc')
    writeFileSync(join(h, '.ssh', 'id_ed25519.pub'), 'ssh-ed25519 AAAAC3 ed@pc\n')
    const k = sshKeyInfo(h)
    expect(k.available).toBe(true)
    expect(k.publicKey).toBe('ssh-ed25519 AAAAC3 ed@pc')
    expect(k.publicKeyPath).toBe(join(h, '.ssh', 'id_ed25519.pub').replace(/\\/g, '/'))
    expect(k.error).toBeUndefined()
  })

  it('says how to make a key when there is none, and rejects a file that is not a key', () => {
    const h = home()
    const none = sshKeyInfo(h)
    expect(none.available).toBe(false)
    expect(none.publicKeyPath).toBeNull()
    expect(none.error).toContain('ssh-keygen')
    mkdirSync(join(h, '.ssh'))
    writeFileSync(join(h, '.ssh', 'id_ed25519.pub'), 'garbage')
    const bad = sshKeyInfo(h)
    expect(bad.available).toBe(false)
    expect(bad.error).toContain('does not look like a public key')
  })
})

describe('the known_hosts path with spaces, read by the real OpenSSH', () => {
  it('reaches ssh as one value (ssh -G only prints its configuration; nothing connects)', () => {
    let exe: string
    try {
      exe = findSsh()
    } catch {
      return
    }
    const dir = mkdtempSync(join(tmpdir(), 'rulz ssh dir-'))
    const file = join(dir, 'known_hosts_7')
    const args = buildSshArgs({ route: direct, knownHostsFile: file, command: REMOTE_COMMAND })
    const r = spawnSync(exe, ['-G', ...args], { encoding: 'utf8', timeout: 20_000, windowsHide: true })
    expect(r.status, r.stderr).toBe(0)
    const line = r.stdout.split(/\r?\n/).find((l) => l.startsWith('userknownhostsfile ')) ?? ''
    expect(line.replace(/"/g, '').replace(/\\/g, '/')).toBe('userknownhostsfile ' + file.replace(/\\/g, '/'))
    expect(r.stdout).toContain('stricthostkeychecking accept-new')
    expect(r.stdout).toContain('batchmode yes')
  })
})
