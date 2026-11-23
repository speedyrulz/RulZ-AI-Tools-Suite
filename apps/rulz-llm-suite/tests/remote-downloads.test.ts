// The model downloads of src/main/remote/scripts.ts: one script at a time per model folder, and several connections for curl.
// Real bash (WSL on Windows, bash elsewhere) with a fake curl that knows ranges and signed addresses. Skips when there is no bash.
import { spawnSync } from 'node:child_process'
import { describe, expect, it, vi } from 'vitest'
import { hfConfigLine, hfDownloadScript, hfStopScript, modelStateScript, removeModelScript } from '../src/main/remote/scripts'
import { frameScript, marker, markers, shq } from '../src/main/remote/ssh'

vi.setConfig({ testTimeout: 240_000 })

/** Same convention as scripts.ts: raw text, `@{` for `${`. */
function bash(strings: TemplateStringsArray, ...vals: Array<string | number>): string {
  return String.raw(strings, ...vals)
    .split('@{')
    .join('${')
}

const distro = process.env.RULZ_WSL_DISTRO ?? 'Ubuntu-24.04'
function bashCmd(): { exe: string; args: string[] } {
  return process.platform === 'win32'
    ? { exe: 'wsl.exe', args: ['-d', distro, '--', 'bash', '--noprofile', '--norc', '-s'] }
    : { exe: 'bash', args: ['--noprofile', '--norc', '-s'] }
}
interface Out {
  code: number | null
  stdout: string
  stderr: string
}
function rawRun(script: string, data = ''): Out {
  const { exe, args } = bashCmd()
  const r = spawnSync(exe, args, { input: frameScript(script) + data, encoding: 'utf8', timeout: 150_000, maxBuffer: 64 << 20 })
  return { code: r.status, stdout: (r.stdout ?? '').replace(/\0/g, ''), stderr: r.stderr ?? '' }
}
const hasBash = ((): boolean => {
  try {
    return rawRun('echo rz-ok').stdout.includes('rz-ok')
  } catch {
    return false
  }
})()
const bashIt = it.skipIf(!hasBash)

class Sandbox {
  readonly T: string
  private n = 0
  constructor() {
    const t = rawRun('mktemp -d').stdout.trim()
    if (!/^\/tmp\/tmp\.[A-Za-z0-9]+$/.test(t)) throw new Error(`unexpected temp folder ${t}`)
    this.T = t
    rawRun(`mkdir -p ${t}/bin ${t}/root ${t}/srv`)
  }
  put(path: string, content: string, exec = false): void {
    // The template literals of the fakes start with a line break; a shebang must be on the first line to be one.
    const r = rawRun(`cat >${this.T}/${path} <<'RZEOF'\n${content.replace(/^\n/, '')}\nRZEOF\n${exec ? `chmod +x ${this.T}/${path}` : ':'}`)
    if (r.code !== 0) throw new Error(`put ${path} failed: ${r.stderr}`)
  }
  private prefix(env: Record<string, string>): string {
    const all = { T: this.T, RZ_ROOT: `${this.T}/root`, ...env }
    const exports = Object.entries(all)
      .map(([k, v]) => `${k}=${shq(v)}`)
      .join(' ')
    return `export ${exports}\nexport PATH=${this.T}/bin:$PATH\n`
  }
  run(script: string, opts: { env?: Record<string, string>; data?: string } = {}): Out {
    return rawRun(this.prefix(opts.env ?? {}) + script, opts.data)
  }
  sh(script: string): string {
    return this.run(script).stdout.trim()
  }
  /** Starts a script the way ssh would, detached from this call; returns the pid of its bash. `tail` is a shell pipeline end, e.g. `| head -n 1 >/dev/null`. */
  bg(script: string, opts: { env?: Record<string, string>; data?: string; tail?: string } = {}): number {
    const id = ++this.n
    this.put(`bg${id}.in`, frameScript(this.prefix(opts.env ?? {}) + script) + (opts.data ?? ''))
    const tail = opts.tail ?? `>${this.T}/bg${id}.out 2>&1`
    // The half second: a session that ends at once takes a process that has just been started with it.
    return Number(this.sh(`setsid nohup bash --noprofile --norc -s <${this.T}/bg${id}.in ${tail} &\np=$!\nsleep 0.5\necho $p`).split('\n').pop())
  }
  isAlive(pid: number): boolean {
    return this.sh(`[ -d /proc/${pid} ] && ! grep -q '^State:[[:space:]]*Z' /proc/${pid}/status && echo alive`) === 'alive'
  }
  waitFor(cond: string, tries = 150): void {
    const r = this.sh(`for i in $(seq 1 ${tries}); do if ${cond}; then echo yes; exit 0; fi; sleep 0.2; done; echo no`)
    if (r !== 'yes') throw new Error(`condition never held: ${cond}
${this.sh('tail -n 20 $T/bg*.out 2>&1; ls -la $T $RZ_ROOT/models/*/ 2>&1; cat $T/curl.log 2>&1')}`)
  }
  cleanup(): void {
    rawRun(`pkill -f '${this.T}/' 2>/dev/null; rm -rf ${this.T}`)
  }
}

/**
 * Plays Hugging Face and its storage: serves $T/srv/<name>. A resolve (a ranged request for byte 0 that prints the final
 * address) answers 206 and a numbered signed address; a ranged request with the newest number gets its bytes. Switches:
 * FAKE_NORANGE (200 to a range), FAKE_403_AT=n (the nth ranged request revokes every address so far), FAKE_SHORT (the
 * first chunk is one byte short), FAKE_FLAKY (the first three requests of every chunk fail), FAKE_SLEEP (each request takes that long), FAKE_CODE (every request fails with it).
 */
const FAKE_CURL = bash`
#!/bin/bash
echo "$*" >>"$T/curl.log"
printf '%s\n' "$@" >>"$T/curl.argv"
out=; url=; range=; wfmt=; cfg=0; head=0
args=("$@")
for ((i=0;i<@{#args[@]};i++)); do a=@{args[i]}; case "$a" in -o) out=@{args[i+1]};; -r) range=@{args[i+1]};; -w) wfmt=@{args[i+1]};; --config) cfg=1;; -sIL) head=1;; https://*) url=$a;; esac; done
if [ $cfg = 1 ]; then c=$(cat); printf '%s\n' "$c" >>"$T/curl.stdin"; u=$(printf '%s\n' "$c" | sed -n 's/^url = "\(.*\)"$/\1/p'); [ -n "$u" ] && url=$u; fi
[ -n "$FAKE_SLEEP" ] && sleep "$FAKE_SLEEP"
name=$(basename "@{url%%\?*}")
src="$T/srv/$name"
code(){ case "$wfmt" in *url_effective*) ;; *http_code*) printf '%s' "$1";; esac; }
if [ -n "$FAKE_CODE" ]; then code "$FAKE_CODE"; exit 22; fi
if [ ! -f "$src" ]; then code 404; exit 22; fi
if [ $head = 1 ]; then printf 'HTTP/2 200\r\ncontent-length: %s\r\n\r\n' "$(stat -c%s "$src")"; exit 0; fi
[ -f "$T/sig" ] || echo 1 >"$T/sig"
sig=$(cat "$T/sig")
if [ "$range" = 0-0 ] && [[ "$wfmt" == *url_effective* ]]; then
  if [ -n "$FAKE_NORANGE" ]; then printf '200 https://cdn.example/blob/%s?sig=%s' "$name" "$sig"; else printf '206 https://cdn.example/blob/%s?sig=%s' "$name" "$sig"; fi
  exit 0
fi
if [ -n "$range" ]; then
  n=1; while ! mkdir "$T/n.$n" 2>/dev/null; do n=$((n+1)); done
  if [ -n "$FAKE_403_AT" ] && [ "$n" = "$FAKE_403_AT" ]; then echo $((sig+1)) >"$T/sig"; touch "$T/revoked"; fi
  sig=$(cat "$T/sig")
  mine=$(printf '%s' "$url" | sed -n 's/.*sig=\([^&]*\).*/\1/p')
  if [ -f "$T/revoked" ] && [ "$mine" != "$sig" ]; then code 403; exit 22; fi
  if [ -n "$FAKE_NORANGE" ]; then code 200; exit 63; fi
  if [ -n "$FAKE_RANGE_FAIL" ]; then code 503; exit 22; fi
  a=@{range%-*}; b=@{range#*-}
  if [ -n "$FAKE_FLAKY" ]; then echo x >>"$T/try.$a"; [ "$(wc -l <"$T/try.$a")" -le 3 ] && { code 503; exit 22; }; fi
  mkdir -p "$T/act.$$"; ls -d "$T"/act.* | wc -l >>"$T/par.log"
  echo "$a" >>"$T/offsets.log"
  len=$((b-a+1)); [ -n "$FAKE_SHORT" ] && [ ! -f "$T/short" ] && { touch "$T/short"; len=$((len-1)); }
  tail -c +$((a+1)) "$src" | head -c "$len" >"$out"
  rm -rf "$T/act.$$"
  code 206; exit 0
fi
off=0; [ -f "$out" ] && off=$(stat -c%s "$out")
tail -c +$((off+1)) "$src" >>"$out"
code 200
exit 0
`

const URL = (n: string): string => `https://huggingface.co/o/r/resolve/main/${n}`

describe.skipIf(!hasBash)('bash: several connections', () => {
  const withBox = (fn: (box: Sandbox) => void): void => {
    const box = new Sandbox()
    try {
      box.put('bin/curl', FAKE_CURL, true)
      fn(box)
    } finally {
      box.cleanup()
    }
  }
  /** A file of `bytes` random bytes in the fake server. */
  const serve = (box: Sandbox, bytes = 700_000, name = 'model.gguf'): { size: number; sha: string } => {
    box.sh(`head -c ${bytes} /dev/urandom > "$T/srv/${name}"`)
    return { size: bytes, sha: box.sh(`sha256sum "$T/srv/${name}" | cut -d" " -f1`) }
  }
  const small = { RZ_CHUNK: '65536', RZ_CONNS: '4', RZ_MULTI_MIN: '1000', RZ_RETRY_SLEEP: '0', RZ_POLL: '0.5' }
  const file = (f: { size: number; sha: string }, name = 'model.gguf') => [{ url: URL(name), name, size: f.size, sha256: f.sha }]
  const part = '"$RZ_ROOT/models/m1/model.gguf.part"'

  bashIt('fetches ranged chunks side by side, appends them in order, and keeps every address off argv', () =>
    withBox((box) => {
      const f = serve(box)
      const r = box.run(hfDownloadScript('m1', file(f)), { data: hfConfigLine(null), env: { ...small, FAKE_SLEEP: '0.3' } })
      expect(r.code, r.stdout + r.stderr).toBe(0)
      expect(box.sh('sha256sum "$RZ_ROOT/models/m1/model.gguf" | cut -d" " -f1')).toBe(f.sha)
      expect(markers(r.stdout, 'FILE_DONE')).toEqual(['model.gguf'])
      expect(markers(r.stdout, 'STAGE')).toContain('Downloading model.gguf, 4 connections')
      // 700000 bytes in chunks of 65536: 11 ranged requests, up to 4 at once and more than 1.
      expect(box.sh('grep -c "^-r$" "$T/curl.argv"')).toBe('12')
      const par = box.sh('sort -n "$T/par.log" | tail -n 1')
      expect(Number(par)).toBeGreaterThan(1)
      expect(Number(par)).toBeLessThanOrEqual(4)
      expect(box.sh('cat "$T/curl.argv"')).not.toContain('cdn.example')
      expect(box.sh('cat "$T/curl.argv"')).not.toContain('sig=')
      expect(box.sh('cat "$T/curl.stdin"')).toContain('sig=1')
      expect(box.sh('ls -a "$RZ_ROOT/models/m1" | tr "\\n" " "')).not.toContain('.dl ')
    })
  )

  bashIt('resumes from a part file of an odd size: the first chunk starts at that offset', () =>
    withBox((box) => {
      const f = serve(box)
      box.sh(`mkdir -p "$RZ_ROOT/models/m1"; head -c 100003 "$T/srv/model.gguf" > ${part}`)
      const r = box.run(hfDownloadScript('m1', file(f)), { data: hfConfigLine(null), env: small })
      expect(r.code, r.stdout + r.stderr).toBe(0)
      expect(box.sh('sort -n "$T/offsets.log" | head -n 1')).toBe('100003')
      expect(box.sh('sha256sum "$RZ_ROOT/models/m1/model.gguf" | cut -d" " -f1')).toBe(f.sha)
    })
  )

  bashIt('uses the single curl when the server answers 200 to a range', () =>
    withBox((box) => {
      const f = serve(box)
      const r = box.run(hfDownloadScript('m1', file(f)), { data: hfConfigLine(null), env: { ...small, FAKE_NORANGE: '1' } })
      expect(r.code, r.stdout + r.stderr).toBe(0)
      expect(box.sh('grep -c "^-C$" "$T/curl.argv"')).toBe('1')
      expect(box.sh('sha256sum "$RZ_ROOT/models/m1/model.gguf" | cut -d" " -f1')).toBe(f.sha)
    })
  )

  bashIt('resolves the address again when a chunk gets 403, once for all chunks that were started with the old one', () =>
    withBox((box) => {
      const f = serve(box)
      const r = box.run(hfDownloadScript('m1', file(f)), { data: hfConfigLine(null), env: { ...small, FAKE_403_AT: '3', FAKE_SLEEP: '0.2' } })
      expect(r.code, r.stdout + r.stderr).toBe(0)
      expect(box.sh('sha256sum "$RZ_ROOT/models/m1/model.gguf" | cut -d" " -f1')).toBe(f.sha)
      expect(box.sh('grep -c "url_effective" "$T/curl.log"'), box.sh('cat "$T/curl.log" "$T/curl.stdin"')).toBe('2')
      expect(box.sh('cat "$T/curl.stdin"')).toContain('sig=2')
    })
  )

  bashIt('fetches the chunks that were refused with an old address again side by side, not one after the other', () =>
    withBox((box) => {
      const f = serve(box)
      // Four chunks for four connections: all of them are refused, and none is left to start.
      const env = { ...small, RZ_CHUNK: '175000', FAKE_403_AT: '1', FAKE_SLEEP: '0.5' }
      const r = box.run(hfDownloadScript('m1', file(f)), { data: hfConfigLine(null), env })
      expect(r.code, r.stdout + r.stderr).toBe(0)
      expect(box.sh('sha256sum "$RZ_ROOT/models/m1/model.gguf" | cut -d" " -f1')).toBe(f.sha)
            const par = Number(box.sh('sort -n "$T/par.log" | tail -n 1'))
      expect(par).toBeGreaterThan(1)
      expect(par).toBeLessThanOrEqual(4)
    })
  )

  bashIt('stops with 36 when a signed address is refused mid-way, and what was appended is a prefix of the file', () =>
    withBox((box) => {
      const f = serve(box)
      const signed = 'https://cdn.example/blob/model.gguf?sig=s3cret_sig'
      const r = box.run(hfDownloadScript('m1', file(f)), { data: hfConfigLine(signed), env: { ...small, FAKE_403_AT: '4', FAKE_SLEEP: '0.2' } })
      expect(r.code).toBe(36)
      const have = Number(box.sh(`stat -c%s ${part}`))
      expect(have).toBeLessThan(f.size)
      expect(box.sh(`cmp -n ${have} ${part} "$T/srv/model.gguf" && echo same`)).toBe('same')
      expect(box.sh('cat "$T/curl.argv"')).not.toContain('s3cret_sig')
      // A fresh address finishes it from there.
      box.sh('rm -f "$T/revoked"')
      expect(box.run(hfDownloadScript('m1', file(f)), { data: hfConfigLine(signed), env: small }).code).toBe(0)
      expect(box.sh('sha256sum "$RZ_ROOT/models/m1/model.gguf" | cut -d" " -f1')).toBe(f.sha)
    })
  )

  bashIt('a long download may fail more than six times as long as it moves on', () =>
    withBox((box) => {
      const f = serve(box, 9 * 1048576)
      const env = { ...small, RZ_CHUNK: '1048576', RZ_CONNS: '1', FAKE_FLAKY: '1' }
      const r = box.run(hfDownloadScript('m1', file(f)), { data: hfConfigLine(null), env })
      expect(r.code, r.stdout + r.stderr).toBe(0)
      expect(box.sh('sha256sum "$RZ_ROOT/models/m1/model.gguf" | cut -d" " -f1')).toBe(f.sha)
      expect(markers(r.stdout, 'STAGE').length).toBeGreaterThan(7)
    })
  )

  bashIt('gives up after six tries that do not move on, and keeps the part file', () =>
    withBox((box) => {
      const f = serve(box)
      const r = box.run(hfDownloadScript('m1', file(f)), { data: hfConfigLine(null), env: { ...small, RZ_RETRY_SLEEP: '0', FAKE_RANGE_FAIL: '1' } })
      expect(r.code, r.stdout + r.stderr).toBe(35)
      expect(markers(r.stdout, 'STAGE')).toHaveLength(6)
      expect(box.sh(`test -f ${part} && echo yes`)).toBe('yes')
    })
  )

  bashIt('fetches a chunk again when it comes back short', () =>
    withBox((box) => {
      const f = serve(box)
      const r = box.run(hfDownloadScript('m1', file(f)), { data: hfConfigLine(null), env: { ...small, FAKE_SHORT: '1' } })
      expect(r.code, r.stdout + r.stderr).toBe(0)
      expect(box.sh('sha256sum "$RZ_ROOT/models/m1/model.gguf" | cut -d" " -f1')).toBe(f.sha)
    })
  )

  bashIt('gives every curl a speed limit, and never lets progress go backwards', () =>
    withBox((box) => {
      const f = serve(box)
      const r = box.run(hfDownloadScript('m1', file(f)), { data: hfConfigLine(null), env: { ...small, FAKE_SLEEP: '0.4' } })
      expect(r.code).toBe(0)
      // 11 chunks; the resolving request has its own time limit.
      expect(box.sh('grep -c "^--speed-limit$" "$T/curl.argv"')).toBe('11')
      let top = -1
      for (const v of markers(r.stdout, 'PROG')) {
        const d = Number(v.split('|')[1])
        expect(d).toBeGreaterThanOrEqual(top)
        top = d
      }
    })
  )

  bashIt('tries again when a worker is gone without an answer, instead of waiting for it', () =>
    withBox((box) => {
      const f = serve(box)
      // The first ranged curl ends its own worker (its parent's parent: the subshell of chunk) and itself.
      box.put(
        'bin/curl',
        FAKE_CURL.replace('echo "$*" >>"$T/curl.log"', 'echo "$*" >>"$T/curl.log"\ncase "$*" in *--max-filesize*) if mkdir "$T/killed" 2>/dev/null; then pp=$(awk "{print \\$4}" /proc/$PPID/stat); kill -9 $pp $PPID; exit 1; fi;; esac'),
        true
      )
      const r = box.run(hfDownloadScript('m1', file(f)), { data: hfConfigLine(null), env: small })
      expect(box.sh('test -d "$T/killed" && echo yes')).toBe('yes')
      expect(r.code, r.stdout + r.stderr).toBe(0)
      expect(box.sh('sha256sum "$RZ_ROOT/models/m1/model.gguf" | cut -d" " -f1')).toBe(f.sha)
    })
  )

  bashIt('ends itself and its workers when its stdout is gone', () =>
    withBox((box) => {
      serve(box)
      // head reads one line and goes away; the next progress line cannot be printed. RZ_MARK finds every process of the script.
      box.bg(hfDownloadScript('m1', [{ url: URL('model.gguf'), name: 'model.gguf', size: 700_000, sha256: null }]), {
        env: { ...small, FAKE_SLEEP: '60', RZ_MARK: 'gone1' },
        data: hfConfigLine(null),
        tail: `2>/dev/null | head -n 1 >/dev/null`
      })
      const marked = 'grep -la "RZ_MARK=gone1" /proc/[0-9]*/environ 2>/dev/null | wc -l'
      box.waitFor('[ -s "$T/curl.log" ]')
      box.waitFor(`[ "$(${marked})" = 0 ]`, 100)
    })
  )
})

describe.skipIf(!hasBash)('bash: owner of a model folder', () => {
  const withBox = (fn: (box: Sandbox) => void): void => {
    const box = new Sandbox()
    try {
      box.put('bin/curl', FAKE_CURL, true)
      fn(box)
    } finally {
      box.cleanup()
    }
  }
  const serve = (box: Sandbox, bytes = 300_000, name = 'model.gguf'): { size: number; sha: string } => {
    box.sh(`head -c ${bytes} /dev/urandom > "$T/srv/${name}"`)
    return { size: bytes, sha: box.sh(`sha256sum "$T/srv/${name}" | cut -d" " -f1`) }
  }
  const dlFile = (f: { size: number; sha: string }) => [{ url: URL('model.gguf'), name: 'model.gguf', size: f.size, sha256: f.sha }]
  const curlProcs = 'grep -la "$T/bin/[c]url" /proc/[0-9]*/cmdline 2>/dev/null | wc -l'

  bashIt('a new download script ends a live leftover of the folder and its curl, then downloads', () =>
    withBox((box) => {
      const f = serve(box)
      const old = box.bg(hfDownloadScript('m1', dlFile(f)), { env: { FAKE_SLEEP: '120' }, data: hfConfigLine(null) })
      box.waitFor('[ "$(' + curlProcs + ')" -gt 0 ]')
      expect(box.isAlive(old)).toBe(true)
      const r = box.run(hfDownloadScript('m1', dlFile(f)), { data: hfConfigLine(null) })
      expect(r.code, r.stdout + r.stderr).toBe(0)
      expect(box.isAlive(old)).toBe(false)
      expect(box.sh(curlProcs)).toBe('0')
      expect(box.sh('sha256sum "$RZ_ROOT/models/m1/model.gguf" | cut -d" " -f1')).toBe(f.sha)
    })
  )

  bashIt('ends a curl whose script is already dead before it writes', () =>
    withBox((box) => {
      const f = serve(box)
      const old = box.bg(hfDownloadScript('m1', dlFile(f)), { env: { FAKE_SLEEP: '120' }, data: hfConfigLine(null) })
      box.waitFor('[ "$(' + curlProcs + ')" -gt 0 ]')
      box.sh(`kill -9 ${old}`)
      expect(Number(box.sh(curlProcs))).toBeGreaterThan(0)
      const r = box.run(hfDownloadScript('m1', dlFile(f)), { data: hfConfigLine(null) })
      expect(r.code, r.stdout + r.stderr).toBe(0)
      expect(box.sh(curlProcs)).toBe('0')
      expect(box.sh('sha256sum "$RZ_ROOT/models/m1/model.gguf" | cut -d" " -f1')).toBe(f.sha)
    })
  )

  bashIt('leaves a download into another model folder alone', () =>
    withBox((box) => {
      const f = serve(box)
      const other = box.bg(hfDownloadScript('m2', dlFile(f)), { env: { FAKE_SLEEP: '120' }, data: hfConfigLine(null) })
      box.waitFor('[ "$(' + curlProcs + ')" -gt 0 ]')
      expect(box.run(hfDownloadScript('m1', dlFile(f)), { data: hfConfigLine(null) }).code).toBe(0)
      expect(box.isAlive(other)).toBe(true)
      expect(box.run(hfStopScript('m2')).code).toBe(0)
      expect(box.isAlive(other)).toBe(false)
    })
  )

  bashIt('stopping a download ends its curl, clears the chunks and keeps the part file', () =>
    withBox((box) => {
      const f = serve(box)
      box.sh('mkdir -p "$RZ_ROOT/models/m1/.dl"; head -c 1000 "$T/srv/model.gguf" > "$RZ_ROOT/models/m1/model.gguf.part"')
      const old = box.bg(hfDownloadScript('m1', dlFile(f)), { env: { FAKE_SLEEP: '120', RZ_CHUNK: '65536', RZ_MULTI_MIN: '1000' }, data: hfConfigLine(null) })
      box.waitFor('[ "$(' + curlProcs + ')" -gt 0 ]')
      const r = box.run(hfStopScript('m1'))
      expect(r.code, r.stdout).toBe(0)
      expect(marker(r.stdout, 'STOPPED')).toBe('1')
      expect(box.isAlive(old)).toBe(false)
      expect(box.sh(curlProcs)).toBe('0')
      expect(box.sh('ls -a "$RZ_ROOT/models/m1" | grep -c "\\.dl$"')).toBe('0')
      expect(box.sh('stat -c%s "$RZ_ROOT/models/m1/model.gguf.part"')).toBe('1000')
      // Nothing to stop, and no folder made for a model that is not there.
      expect(box.run(hfStopScript('m9')).code).toBe(0)
      expect(box.sh('ls "$RZ_ROOT/models"')).toBe('m1')
    })
  )

  bashIt('removing a model ends a live download first', () =>
    withBox((box) => {
      const f = serve(box)
      const old = box.bg(hfDownloadScript('m1', dlFile(f)), { env: { FAKE_SLEEP: '120' }, data: hfConfigLine(null) })
      box.waitFor('[ "$(' + curlProcs + ')" -gt 0 ]')
      const r = box.run(removeModelScript('m1'))
      expect(r.code, r.stdout).toBe(0)
      expect(marker(r.stdout, 'REMOVED')).toBe('1')
      expect(box.isAlive(old)).toBe(false)
      expect(box.sh(curlProcs)).toBe('0')
      expect(box.sh('ls "$RZ_ROOT/models" | wc -l')).toBe('0')
    })
  )

  bashIt('the state script clears the chunks a dead download left, and leaves those of a live one', () =>
    withBox((box) => {
      const f = serve(box)
      const chunks = 'test -d "$RZ_ROOT/models/m1/.dl" && echo yes'
      box.sh('mkdir -p "$RZ_ROOT/models/m1/.dl"; echo junk > "$RZ_ROOT/models/m1/.dl/x.0"; : > "$RZ_ROOT/models/.own-m1"')
      box.run(modelStateScript('m1'))
      expect(box.sh(chunks)).toBe('')
      box.bg(hfDownloadScript('m1', dlFile(f)), { env: { FAKE_SLEEP: '8', RZ_CHUNK: '65536', RZ_MULTI_MIN: '1000' }, data: hfConfigLine(null) })
      box.waitFor('[ -d "$RZ_ROOT/models/m1/.dl" ]')
      box.run(modelStateScript('m1'))
      expect(box.sh(chunks)).toBe('yes')
      box.run(hfStopScript('m1'))
    })
  )
})

