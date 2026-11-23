// The bash scripts of src/main/remote/scripts.ts: their text always, and their behaviour in real bash (WSL on Windows,
// bash elsewhere) with fake curl and nvidia-smi on PATH. The bash tests skip when no bash is available.
import { spawnSync } from 'node:child_process'
import { describe, expect, it, vi } from 'vitest'
import {
  armScript,
  disarmScript,
  engineInstallScript,
  freeSpaceScript,
  hfConfigLine,
  hfDownloadScript,
  keepaliveLoopScript,
  launchCheckScript,
  markCompleteScript,
  modelStateScript,
  onstartScript,
  parseComplete,
  parseProbe,
  PORT_PLACEHOLDER,
  probeScript,
  removeModelScript,
  serverLogScript,
  serverStartScript,
  serverStatusScript,
  serverStopScript,
  touchScript,
  uploadAppendScript,
  uploadDiscardScript,
  uploadFinishScript,
  uploadStateScript,
  verifyScript,
  WATCHDOG_PROGRAM
} from '../src/main/remote/scripts'
import { frameScript, marker, markers, shq } from '../src/main/remote/ssh'

// Each shell call takes a moment and spawnSync blocks the timer, so the default 5 s is far too short.
vi.setConfig({ testTimeout: 240_000 })

const SHA = 'a'.repeat(64)

/** Same convention as scripts.ts: raw text, `@{` for `${`. */
function bash(strings: TemplateStringsArray, ...vals: Array<string | number>): string {
  return String.raw(strings, ...vals)
    .split('@{')
    .join('${')
}

// ---------- Real bash, when there is one ----------

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
/** Runs a script the way ssh does: framed, on stdin of `bash -s`. */
function rawRun(script: string, data = ''): Out {
  const { exe, args } = bashCmd()
  const r = spawnSync(exe, args, { input: frameScript(script) + data, encoding: 'utf8', timeout: 120_000, maxBuffer: 64 << 20 })
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

/** A scratch folder in the shell's own file system with fake commands in bin/. */
class Sandbox {
  readonly T: string
  constructor() {
    const t = rawRun('mktemp -d').stdout.trim()
    if (!/^\/tmp\/tmp\.[A-Za-z0-9]+$/.test(t)) throw new Error(`unexpected temp folder ${t}`)
    this.T = t
    rawRun(`mkdir -p ${t}/bin ${t}/root ${t}/srv`)
  }
  put(path: string, content: string, exec = false): void {
    const r = rawRun(`cat >${this.T}/${path} <<'RZEOF'\n${content}\nRZEOF\n${exec ? `chmod +x ${this.T}/${path}` : ':'}`)
    if (r.code !== 0) throw new Error(`put ${path} failed: ${r.stderr}`)
  }
  run(script: string, opts: { env?: Record<string, string>; data?: string } = {}): Out {
    const env = { T: this.T, RZ_ROOT: `${this.T}/root`, ...opts.env }
    const exports = Object.entries(env)
      .map(([k, v]) => `${k}=${shq(v)}`)
      .join(' ')
    return rawRun(`export ${exports}\nexport PATH=${this.T}/bin:$PATH\n${script}`, opts.data)
  }
  sh(script: string): string {
    return this.run(script).stdout.trim()
  }
  cleanup(): void {
    // Only the folder made by mktemp above.
    rawRun(`pkill -f '${this.T}/' 2>/dev/null; rm -rf ${this.T}`)
  }
}

const FAKE_NVIDIA = bash`
#!/bin/bash
[ -n "$NV_FAIL" ] && exit 9
case "$*" in
*index,name*) printf '0, NVIDIA Fake GPU, 24000, %s, %s\n' "@{NV_USED:-100}" "@{NV_UTIL:-0}";;
*utilization.gpu*) printf '%s\n' "@{NV_UTIL:-0}";;
*compute-apps*) [ -n "$NV_APPS" ] && printf '%s\n' $NV_APPS;;
*memory.used*) printf '%s\n' "@{NV_USED:-0}";;
esac
exit 0
`

/** Plays the Hugging Face / GitHub side: serves files from $T/srv, resumes with -C -, reads a config from stdin. */
const FAKE_CURL = bash`
#!/bin/bash
echo "$*" >>"$T/curl.log"
printf '%s\n' "$@" >>"$T/curl.argv"
out=; head=0; url=; wcode=0; cfg=0
args=("$@")
for ((i=0;i<@{#args[@]};i++)); do a=@{args[i]}; case "$a" in -o) out=@{args[i+1]};; -w) wcode=1;; --config) cfg=1;; -sIL|-I|-sI) head=1;; https://*) url=$a;; esac; done
if [ $cfg = 1 ]; then c=$(cat); printf '%s\n' "$c" >>"$T/curl.stdin"; u=$(printf '%s\n' "$c" | sed -n 's/^url = "\(.*\)"$/\1/p'); [ -n "$u" ] && url=$u; fi
name=$(basename "@{url%%\?*}")
src="$T/srv/$name"
if [ -n "$FAKE_CODE" ]; then [ $wcode = 1 ] && printf '%s' "$FAKE_CODE"; exit 22; fi
if [ ! -f "$src" ]; then [ $wcode = 1 ] && printf 404; exit 22; fi
if [ $head = 1 ]; then printf 'HTTP/2 200\r\ncontent-length: %s\r\n\r\n' "$(stat -c%s "$src")"; exit 0; fi
off=0; [ -f "$out" ] && off=$(stat -c%s "$out")
tail -c +$((off+1)) "$src" >>"$out"
[ $wcode = 1 ] && printf 200
exit 0
`

/** A fake curl for the watchdog: records argv and stdin, answers 200 (or $CURL_CODE). */
const FAKE_VAST_CURL = bash`
#!/bin/bash
printf '%s\n' "$@" >>"$T/vast.argv"
cat >>"$T/vast.stdin"
printf '%s' "@{CURL_CODE:-200}"
`

// ---------- Text properties ----------

const builders: Array<[string, string]> = [
  ['probe', probeScript()],
  ['arm', armScript({ idleMinutes: 30, capAt: 1900000000, origin: 'app', capHours: 12 })],
  ['disarm', disarmScript()],
  ['verify', verifyScript()],
  ['onstart', onstartScript({ idleMinutes: 30, capHours: 12 })],
  ['onstart without cap', onstartScript({ idleMinutes: 0, capHours: null })],
  ['keepalive', keepaliveLoopScript()],
  ['touch', touchScript()],
  ['free space', freeSpaceScript()],
  [
    'engine',
    engineInstallScript({
      tag: 'b12407',
      llama: { url: 'https://github.com/ggml-org/llama.cpp/releases/download/b12407/llama-b12407-bin-ubuntu-cuda-12.8-x64.tar.gz', sha256: SHA, size: 1000 },
      cudart: { url: 'https://github.com/ggml-org/llama.cpp/releases/download/b12407/cudart-llama-b12407-bin-ubuntu-cuda-12.8-x64.tar.gz', sha256: SHA, size: 2000 }
    })
  ],
  [
    'hf download',
    hfDownloadScript('my-model', [
      { url: 'https://huggingface.co/o/r/resolve/main/m-00001-of-00002.gguf', name: 'm-00001-of-00002.gguf', size: 123, sha256: SHA },
      { url: 'https://huggingface.co/o/r/resolve/main/mmproj.gguf', name: 'mmproj.gguf', size: null, sha256: null }
    ])
  ],
  ['mark complete', markCompleteScript('my-model', [{ name: 'a.gguf', size: 5, sha256: SHA, source: '-' }, { name: 'b.gguf', size: 6, sha256: '-', source: '-' }])],
  ['model state', modelStateScript('my-model')],
  ['remove model', removeModelScript('my-model')],
  ['upload state', uploadStateScript('my-model', 'a.gguf')],
  ['upload append', uploadAppendScript('my-model', 'a.gguf')],
  ['upload finish', uploadFinishScript('my-model', 'a.gguf', 5, SHA)],
  ['upload finish final', uploadFinishScript('my-model', 'a.gguf', 5, SHA, 'final')],
  ['upload discard', uploadDiscardScript('my-model', 'a.gguf')],
  ['server start', serverStartScript({ id: 'abc-123', tag: 'b12407', args: ['--model', "/w/it's a model.gguf", '--port', PORT_PLACEHOLDER], hash: SHA, expectMiB: 4000 })],
  ['server status', serverStatusScript('abc-123')],
  ['server log', serverLogScript('abc-123', 0)],
  ['server log tail', serverLogScript('abc-123', -4000)],
  ['server stop', serverStopScript('abc-123')],
  ['launch check', launchCheckScript({ id: 'abc-123', tag: 'b12407', slug: 'my-model' })],
  ['watchdog', WATCHDOG_PROGRAM]
]

describe('script text', () => {
  it('has LF endings only and no NUL', () => {
    for (const [name, text] of builders) {
      expect(text, name).not.toContain('\r')
      expect(text, name).not.toContain('\0')
    }
  })

  it('keeps the onstart script under 4000 characters, watchdog included', () => {
    for (const s of [onstartScript({ idleMinutes: 30, capHours: 12 }), onstartScript({ idleMinutes: 10080, capHours: 2160 })]) {
      expect(s.length).toBeLessThan(4000)
      expect(s).toContain('watchdog.sh')
      expect(s).toContain('keepalive')
    }
    expect(WATCHDOG_PROGRAM.length).toBeLessThan(3000)
  })

  it('writes onstart so that a missing /workspace works and the image start is not broken', () => {
    const s = onstartScript({ idleMinutes: 30, capHours: 12 })
    expect(s).toContain('[ -d /workspace ] || R=/root/rulz-llm')
    // The watchdog text in the middle may exit (it is its own program); the parts around it must not.
    const open = s.indexOf('<<')
    const close = s.indexOf('\nRZW\n')
    expect(s.slice(0, open)).not.toMatch(/\bexit\b/)
    expect(s.slice(close + 5)).not.toMatch(/\bexit\b/)
    expect(s).not.toContain('set -e')
    expect(s).toContain('setsid nohup')
  })

  it('never puts a secret on a command line: the key goes through a curl config on stdin', () => {
    expect(WATCHDOG_PROGRAM).toContain('--config -')
    expect(WATCHDOG_PROGRAM).toContain('printf \'header = "Authorization: Bearer %s"')
    expect(WATCHDOG_PROGRAM).not.toMatch(/curl[^\n]*Bearer/)
    const dl = hfDownloadScript('m', [{ url: 'https://huggingface.co/o/r/resolve/main/a.gguf', name: 'a.gguf', size: 1, sha256: null }])
    expect(dl).toContain('IFS= read -r CFG')
    expect(dl).toContain("printf '%s\\n' \"$CFG\" | curl --globoff --config -")
    expect(dl).not.toContain('Authorization')
    expect(hfConfigLine('https://cdn.example/x?sig=1')).toBe('url = "https://cdn.example/x?sig=1"\n')
    expect(hfConfigLine(null)).toBe('\n')
    for (const bad of ['hf_abc', 'http://cdn.example/x', 'https://cdn.example/"x', 'https://cdn.example/x\nurl = "file:///etc/passwd"', 'https://cdn.example/\\x', 'https://cdn.example/a b']) {
      expect(() => hfConfigLine(bad), bad).toThrow()
    }
  })

  it('installs only the missing packages, checks digests and resumes both downloads', () => {
    const s = builders.find(([n]) => n === 'engine')?.[1] ?? ''
    expect(s).toContain('apt-get install -y libgomp1 curl ca-certificates')
    expect(s).toContain('-C -')
    expect(s).toContain('sha256sum -c')
    expect(s).toContain('--version')
    expect(s).toContain('--list-devices')
    expect(s.indexOf('touch "$E/.ok"')).toBeGreaterThan(s.indexOf('--list-devices'))
  })

  it('refuses unvalidated values', () => {
    expect(() => serverStopScript("a'; rm -rf /; '")).toThrow()
    expect(() => serverStartScript({ id: 'ok', tag: 'v1', args: [], hash: SHA, expectMiB: null })).toThrow()
    expect(() => serverStartScript({ id: 'ok', tag: 'b12407', args: ['a\nb'], hash: SHA, expectMiB: null })).toThrow()
    expect(() => serverStartScript({ id: 'ok', tag: 'b12407', args: [], hash: 'nothex', expectMiB: null })).toThrow()
    expect(() => modelStateScript('Bad Slug')).toThrow()
    expect(() => modelStateScript('../x')).toThrow()
    expect(() => uploadStateScript('ok', '../etc')).toThrow()
    expect(() => uploadStateScript('ok', 'a.gguf.part')).toThrow()
    expect(() => uploadFinishScript('ok', 'a.gguf', 5, 'zz')).toThrow()
    expect(() => hfDownloadScript('ok', [{ url: 'https://evil.example/x', name: 'a.gguf', size: 1, sha256: null }])).toThrow()
    expect(() => hfDownloadScript('ok', [{ url: "https://huggingface.co/a'b", name: 'a.gguf', size: 1, sha256: null }])).toThrow()
    expect(() => engineInstallScript({ tag: 'b12407', llama: { url: 'https://evil.example/a', sha256: SHA, size: 1 }, cudart: { url: 'https://github.com/ggml-org/llama.cpp/releases/download/b12407/x', sha256: SHA, size: 1 } })).toThrow()
    expect(() => armScript({ idleMinutes: -1, capAt: null, origin: 'app', capHours: 12 })).toThrow()
    expect(() => armScript({ idleMinutes: 1.5, capAt: null, origin: 'app', capHours: 12 })).toThrow()
    expect(() => serverLogScript('ok', 1.5)).toThrow()
  })

  it('quotes arguments of the start script as single words', () => {
    const s = serverStartScript({ id: 'abc', tag: 'b12407', args: ['--alias', "x'; touch /tmp/pwned #", '--port', PORT_PLACEHOLDER], hash: SHA, expectMiB: 100 })
    expect(s).toContain(`'x'\\''; touch /tmp/pwned #'`)
  })

  it('parses probe output and ignores noise', () => {
    const out = ['Welcome banner', '@rz ROOT=/workspace/rulz-llm', '@rz GPU=0|NVIDIA GeForce RTX 4090|24564|312|7', '@rz NVSMI=1', '@rz DISK_FREE=1234', '@rz ENGINE=b12407', '@rz MODEL=m1', '@rz WATCHDOG_RUNNING=1', '@rz WATCHDOG_STATE=ok', '@rz FOREIGN=0', 'tmux: attached'].join('\n')
    const p = parseProbe(out)
    expect(p.gpus).toEqual([{ index: 0, name: 'NVIDIA GeForce RTX 4090', totalMiB: 24564, usedMiB: 312, utilization: 7 }])
    expect(p.diskFreeBytes).toBe(1234)
    expect(p.engines).toEqual(['b12407'])
    expect(p.modelSlugs).toEqual(['m1'])
    expect(p.watchdogRunning && p.watchdogState === 'ok' && !p.foreign).toBe(true)
    expect(marker('@rz A=1\n@rz A=2', 'A')).toBe('2')
    expect(markers('@rz A=1\nnoise\n@rz A=2', 'A')).toEqual(['1', '2'])
    expect(parseComplete('a.gguf\t10\tabc\nbad line\nb.gguf\t5\t-\n')).toEqual([
      { name: 'a.gguf', size: 10, sha256: 'abc', source: '-' },
      { name: 'b.gguf', size: 5, sha256: '-', source: '-' }
    ])
  })
})

// ---------- Real bash ----------

describe.skipIf(!hasBash)('bash: syntax and framing', () => {
  it('passes bash -n for every script', () => {
    const { exe, args } = bashCmd()
    for (const [name, text] of builders) {
      const r = spawnSync(exe, [...args.slice(0, -1), '-n'], { input: text, encoding: 'utf8' })
      expect(r.status, `${name}: ${r.stderr}`).toBe(0)
    }
  })

  it('quotes hostile values so bash sees them unchanged', () => {
    const values = ["it's", '$(echo pwned)', '`echo pwned`', 'a\\b\\n', '"; echo pwned; "', 'line1\nline2', '*', '-rf', '$HOME', ' leading and trailing ', "'", "''"]
    for (const v of values) {
      const r = rawRun(`printf '%s' ${shq(v)} | od -An -tx1 | tr -d ' \\n'`)
      expect(r.stdout, JSON.stringify(v)).toBe(Buffer.from(v).toString('hex'))
    }
  })

  it('lets a script read the data that follows it on stdin, byte for byte', () => {
    const payload = Buffer.from([0, 1, 2, 255, 10, 13, 10, 200, 7]).toString('latin1')
    const { exe, args } = bashCmd()
    const input = Buffer.concat([Buffer.from(frameScript('cat | od -An -tx1 | tr -d " \\n"; echo; echo done')), Buffer.from(payload, 'latin1')])
    const r = spawnSync(exe, args, { input, encoding: 'utf8' })
    expect(r.stdout.split('\n')[0]).toBe('000102ff0a0d0ac807')
    expect(r.stdout).toContain('done')
  })

  it('does not read more of the script than it runs', () => {
    const r = rawRun('read -r a; read -r b; echo "[$a][$b]"', 'first\nsecond\n')
    expect(r.stdout.trim()).toBe('[first][second]')
  })
})

describe.skipIf(!hasBash)('bash: watchdog', () => {
  const wd = (box: Sandbox, env: Record<string, string>, limits: string, opts: { keepaliveAge?: string; pre?: string; post?: string } = {}): { log: string; vast: string; stdin: string } => {
    box.put('bin/nvidia-smi', FAKE_NVIDIA, true)
    box.put('bin/curl', FAKE_VAST_CURL, true)
    box.put('watchdog.sh', WATCHDOG_PROGRAM)
    box.sh(`mkdir -p "$RZ_ROOT"; printf '${limits}' > "$RZ_ROOT/limits"; touch -d '${opts.keepaliveAge ?? '2 hours ago'}' "$RZ_ROOT/keepalive"`)
    box.run(`${opts.pre ?? ''}\nbash $T/watchdog.sh\n${opts.post ?? ''}`, {
      env: { RZ_TICK: '0', RZ_RETRY: '0', RZ_N: '14', CONTAINER_ID: 'C.4242', CONTAINER_API_KEY: 'sekret-key-123', ...env }
    })
    return { log: box.sh('cat "$RZ_ROOT/watchdog.log" 2>/dev/null'), vast: box.sh('cat "$T/vast.argv" 2>/dev/null'), stdin: box.sh('cat "$T/vast.stdin" 2>/dev/null') }
  }
  const run = (fn: (box: Sandbox) => void): void => {
    const box = new Sandbox()
    try {
      fn(box)
    } finally {
      box.cleanup()
    }
  }

  bashIt('stops an idle instance: old keepalive, ten quiet samples, no foreign process', () =>
    run((box) => {
      const r = wd(box, { NV_UTIL: '3', NV_USED: '200' }, 'idle=30\\ncap=0\\norg=app\\n')
      expect(r.log).toContain('stopping the instance (idle limit)')
      expect(r.log).toContain('stop request: http 200')
      expect(r.vast).toContain('https://console.vast.ai/api/v0/instances/4242/')
      expect(r.vast).toContain('PUT')
      expect(r.vast).toContain('{"state":"stopped"}')
    })
  )

  bashIt('keeps the key out of argv, the log and every file of the sandbox; only the curl config on stdin has it', () =>
    run((box) => {
      const r = wd(box, { NV_UTIL: '0' }, 'idle=30\\ncap=0\\norg=app\\n')
      expect(r.vast).not.toContain('sekret-key-123')
      expect(r.log).not.toContain('sekret-key-123')
      expect(r.stdin).toContain('header = "Authorization: Bearer sekret-key-123"')
      expect(box.sh('grep -rl sekret-key-123 "$T" | grep -v vast.stdin | wc -l')).toBe('0')
    })
  )

  bashIt('does not stop while the GPU is busy, keepalive is fresh, fewer than ten samples exist, or the limits are 0', () => {
    run((box) => expect(wd(box, { NV_UTIL: '60' }, 'idle=30\\ncap=0\\norg=app\\n').log).not.toContain('stopping'))
    run((box) => expect(wd(box, { NV_UTIL: '0' }, 'idle=30\\ncap=0\\norg=app\\n', { keepaliveAge: '1 minute ago' }).log).not.toContain('stopping'))
    run((box) => expect(wd(box, { NV_UTIL: '0', RZ_N: '6' }, 'idle=30\\ncap=0\\norg=app\\n').log).not.toContain('stopping'))
    run((box) => expect(wd(box, { NV_UTIL: '0' }, 'idle=0\\ncap=0\\norg=app\\n').log).not.toContain('stopping'))
    run((box) => expect(wd(box, { NV_UTIL: '0' }, 'idle=0\\ncap=0\\norg=app\\n').vast).toBe(''))
  })

  bashIt('a busy sample in the middle restarts the count', () =>
    run((box) => {
      // Utilisation alternates, so there are never ten quiet samples in a row.
      box.put('bin/nvidia-smi', FAKE_NVIDIA.replace('*utilization.gpu*) printf \'%s\\n\' "${NV_UTIL:-0}";;', '*utilization.gpu*) n=$(cat "$T/n" 2>/dev/null || echo 0); echo $((n+1)) >"$T/n"; [ $((n%5)) = 4 ] && echo 90 || echo 0;;'), true)
      box.put('bin/curl', FAKE_VAST_CURL, true)
      box.put('watchdog.sh', WATCHDOG_PROGRAM)
      box.sh(`printf 'idle=30\\ncap=0\\norg=app\\n' > "$RZ_ROOT/limits"; touch -d '2 hours ago' "$RZ_ROOT/keepalive"`)
      box.run('bash $T/watchdog.sh', { env: { RZ_TICK: '0', RZ_RETRY: '0', RZ_N: '30', CONTAINER_ID: '77', CONTAINER_API_KEY: 'k' } })
      expect(box.sh('cat "$RZ_ROOT/watchdog.log"')).not.toContain('stopping')
    })
  )

  bashIt('does not stop an idle instance while another process uses the GPU, but a cap on an app rental stops it anyway', () => {
    const foreign = 'sleep 300 >/dev/null 2>&1 & export NV_APPS=$!'
    run((box) => {
      const r = wd(box, { NV_UTIL: '0' }, 'idle=30\\ncap=0\\norg=app\\n', { pre: foreign, post: 'kill $NV_APPS' })
      expect(r.log).toContain('another GPU process may be running: not stopping')
      expect(r.vast).toBe('')
    })
    run((box) => {
      const r = wd(box, { NV_UTIL: '90' }, 'idle=0\\ncap=1\\norg=app\\n', { pre: foreign, post: 'kill $NV_APPS' })
      expect(r.log).toContain('stopping the instance (cap limit)')
      expect(r.vast).toContain('PUT')
    })
    run((box) => {
      const r = wd(box, { NV_UTIL: '90' }, 'idle=0\\ncap=1\\norg=attached\\n', { pre: foreign, post: 'kill $NV_APPS' })
      expect(r.log).toContain('not stopping')
      expect(r.vast).toBe('')
    })
    run((box) => {
      const r = wd(box, { NV_UTIL: '90' }, 'idle=0\\ncap=1\\norg=attached\\n')
      expect(r.log).toContain('stopping the instance (cap limit)')
    })
  })

  bashIt('does not stop before the cap and logs a refusal of the API without the key, retrying', () => {
    run((box) => expect(wd(box, {}, 'idle=0\\ncap=4102444800\\norg=app\\n').log).not.toContain('stopping'))
    run((box) => {
      const r = wd(box, { CURL_CODE: '401', NV_UTIL: '90', RZ_N: '3' }, 'idle=0\\ncap=1\\norg=app\\n')
      expect(r.log.match(/stop request: http 401/g)?.length).toBe(3)
      expect(r.log).not.toContain('sekret')
    })
  })

  bashIt('says so when it has no id or key, and reads the instance id from CONTAINER_ID or the label', () => {
    run((box) => {
      const r = wd(box, { CONTAINER_API_KEY: '', NV_UTIL: '90', RZ_N: '1' }, 'idle=0\\ncap=1\\norg=app\\n')
      expect(r.log).toContain('http nokey')
    })
    run((box) => {
      const r = wd(box, { CONTAINER_ID: '', VAST_CONTAINERLABEL: 'C.999', NV_UTIL: '90', RZ_N: '1' }, 'idle=0\\ncap=1\\norg=app\\n')
      expect(r.vast).toContain('/instances/999/')
    })
  })

  bashIt('runs once: a second start exits at once while the first is alive', () =>
    run((box) => {
      box.put('bin/nvidia-smi', FAKE_NVIDIA, true)
      box.put('bin/curl', FAKE_VAST_CURL, true)
      box.put('watchdog.sh', WATCHDOG_PROGRAM)
      const r = box.run(
        'mkdir -p "$RZ_ROOT"; printf "idle=0\\ncap=0\\n" >"$RZ_ROOT/limits"; (RZ_TICK=1 bash $T/watchdog.sh &) ; sleep 1.5; a=$(cat "$RZ_ROOT/watchdog.pid"); bash $T/watchdog.sh; echo "second exited $?"; b=$(cat "$RZ_ROOT/watchdog.pid"); [ "$a" = "$b" ] && echo same-pid; kill "$a"'
      )
      expect(r.stdout).toContain('second exited 0')
      expect(r.stdout).toContain('same-pid')
    })
  )

  bashIt('onstart writes the limits, touches keepalive, installs the watchdog and starts it', () =>
    run((box) => {
      box.put('bin/nvidia-smi', FAKE_NVIDIA, true)
      box.put('bin/curl', FAKE_VAST_CURL, true)
      // The root is the sandbox, not the machine's /workspace or /root.
      const script = onstartScript({ idleMinutes: 45, capHours: 6 }).replace('R=/workspace/rulz-llm; [ -d /workspace ] || R=/root/rulz-llm', 'R=$RZ_ROOT')
      expect(script).toContain('R=$RZ_ROOT')
      const r = box.run(`${script}\nsleep 2\ncat "$R/limits"; echo ---; ls "$R"; echo ---; kill $(cat "$R/watchdog.pid")`, { env: { CONTAINER_ID: '5', CONTAINER_API_KEY: 'k' } })
      expect(r.stdout).toContain('idle=45')
      expect(r.stdout).toMatch(/cap=\d{10}/)
      expect(r.stdout).toContain('org=app')
      expect(r.stdout).toContain('keepalive')
      expect(r.stdout).toContain('watchdog.pid')
      expect(r.stdout).toContain('watchdog.sh')
      const cap = Number(/cap=(\d+)/.exec(r.stdout)?.[1])
      const now = Math.floor(Date.now() / 1000)
      expect(Math.abs(cap - (now + 6 * 3600))).toBeLessThan(600)
    })
  )

  bashIt('arm writes limits, starts the watchdog and verifies with a GET; reports why it cannot', () => {
    run((box) => {
      box.put('bin/nvidia-smi', FAKE_NVIDIA, true)
      box.put('bin/curl', FAKE_VAST_CURL, true)
      const r = box.run(armScript({ idleMinutes: 20, capAt: 1900000000, origin: 'app', capHours: 12 }) + '\n', { env: { CONTAINER_ID: '12', CONTAINER_API_KEY: 'k', RZ_TICK: '1' } })
      expect(marker(r.stdout, 'ARMED')).toBe('1')
      expect(marker(r.stdout, 'VERIFY')).toBe('200')
      expect(box.sh('cat "$RZ_ROOT/limits"')).toBe('idle=20\ncap=1900000000\norg=app')
      expect(box.sh('cat "$T/vast.argv"')).not.toContain('PUT')
      expect(box.sh('cat "$RZ_ROOT/watchdog.state"')).toBe('ok')
      box.sh('kill $(cat "$RZ_ROOT/watchdog.pid")')
    })
    run((box) => {
      box.put('bin/nvidia-smi', FAKE_NVIDIA, true)
      box.put('bin/curl', FAKE_VAST_CURL, true)
      const r = box.run(armScript({ idleMinutes: 20, capAt: null, origin: 'attached', capHours: 12 }) + '\n', { env: { CONTAINER_ID: '12', RZ_TICK: '1' } })
      expect(marker(r.stdout, 'ARMED')).toBe('0')
      expect(marker(r.stdout, 'ERR')).toContain('CONTAINER_API_KEY')
      expect(box.sh('cat "$RZ_ROOT/watchdog.state"')).toContain('error:')
      box.sh('kill $(cat "$RZ_ROOT/watchdog.pid")')
    })
    run((box) => {
      box.put('bin/nvidia-smi', FAKE_NVIDIA, true)
      box.put('bin/curl', FAKE_VAST_CURL, true)
      const r = box.run(armScript({ idleMinutes: 20, capAt: null, origin: 'app', capHours: 12 }) + '\n', { env: { CONTAINER_ID: '12', CONTAINER_API_KEY: 'k', CURL_CODE: '401', RZ_TICK: '1' } })
      expect(marker(r.stdout, 'ERR')).toContain('401')
      box.sh('kill $(cat "$RZ_ROOT/watchdog.pid")')
    })
  })

  bashIt('arming again with the same program keeps the running watchdog; disarm zeroes the limits', () =>
    run((box) => {
      box.put('bin/nvidia-smi', FAKE_NVIDIA, true)
      box.put('bin/curl', FAKE_VAST_CURL, true)
      const env = { CONTAINER_ID: '12', CONTAINER_API_KEY: 'k', RZ_TICK: '1' }
      box.run(armScript({ idleMinutes: 20, capAt: null, origin: 'app', capHours: 12 }), { env })
      const a = box.sh('cat "$RZ_ROOT/watchdog.pid"')
      box.run(armScript({ idleMinutes: 5, capAt: null, origin: 'app', capHours: 12 }), { env })
      expect(box.sh('cat "$RZ_ROOT/watchdog.pid"')).toBe(a)
      expect(box.sh('cat "$RZ_ROOT/limits"')).toContain('idle=5')
      const d = box.run(disarmScript())
      expect(marker(d.stdout, 'ARMED')).toBe('0')
      expect(box.sh('cat "$RZ_ROOT/limits"')).toContain('idle=0')
      box.sh('kill $(cat "$RZ_ROOT/watchdog.pid")')
    })
  )
})

describe.skipIf(!hasBash)('bash: foreign process check (probe)', () => {
  const probe = (box: Sandbox, env: Record<string, string>, pre = ''): ReturnType<typeof parseProbe> => {
    box.put('bin/nvidia-smi', FAKE_NVIDIA, true)
    return parseProbe(box.run(`${pre}\n${probeScript()}`, { env }).stdout)
  }
  const withBox = (fn: (box: Sandbox) => void): void => {
    const box = new Sandbox()
    try {
      fn(box)
    } finally {
      box.cleanup()
    }
  }
  /** A server of ours: a copy of bash in our engine folder, with a run folder and an expectation. */
  const ourServer = 'mkdir -p "$RZ_ROOT/engine/b100" "$RZ_ROOT/run/s1"; cp /bin/bash "$RZ_ROOT/engine/b100/llama-server"; "$RZ_ROOT/engine/b100/llama-server" -c "sleep 300; :" >/dev/null 2>&1 </dev/null & echo $! >"$RZ_ROOT/run/s1/pid"; echo 4000 >"$RZ_ROOT/run/s1/expect"; OURS=$!; sleep 0.3'

  bashIt('reads GPUs, disk, engines, models and the watchdog; a quiet GPU with little memory used is not foreign', () =>
    withBox((box) => {
      const p = probe(box, { NV_USED: '150' }, 'mkdir -p "$RZ_ROOT/engine/b900" "$RZ_ROOT/models/m1" "$RZ_ROOT/models/m2"; touch "$RZ_ROOT/engine/b900/.ok" "$RZ_ROOT/models/m1/.complete"; echo hello > "$RZ_ROOT/models/m1/.complete"')
      expect(p.gpus).toEqual([{ index: 0, name: 'NVIDIA Fake GPU', totalMiB: 24000, usedMiB: 150, utilization: 0 }])
      expect(p.diskFreeBytes).toBeGreaterThan(0)
      expect(p.engines).toEqual(['b900'])
      expect(p.modelSlugs).toEqual(['m1'])
      expect(p.watchdogRunning).toBe(false)
      expect(p.foreign).toBe(false)
      expect(p.root).toMatch(/\/root$/)
    })
  )

  bashIt('counts memory used far above our servers share as foreign', () => {
    withBox((box) => expect(probe(box, { NV_USED: '5000' }).foreign).toBe(true))
    withBox((box) => expect(probe(box, { NV_USED: '4500' }, `${ourServer}`).foreign).toBe(false))
    withBox((box) => expect(probe(box, { NV_USED: '9000' }, `${ourServer}`).foreign).toBe(true))
  })

  bashIt('counts a visible process that is not ours as foreign, but not our own loading server listed by pid', () => {
    withBox((box) => expect(probe(box, { NV_USED: '100' }, 'sleep 300 >/dev/null 2>&1 & export NV_APPS=$!').foreign).toBe(true))
    withBox((box) => expect(probe(box, { NV_USED: '100' }, `${ourServer}; export NV_APPS=$OURS`).foreign).toBe(false))
  })

  bashIt('treats pids that are not visible here as ours only as far as we run servers, and a failing nvidia-smi as foreign', () => {
    withBox((box) => expect(probe(box, { NV_USED: '100', NV_APPS: '999999999' }).foreign).toBe(true))
    withBox((box) => expect(probe(box, { NV_USED: '3000', NV_APPS: '999999999' }, `${ourServer}`).foreign).toBe(false))
    withBox((box) => expect(probe(box, { NV_USED: '3000', NV_APPS: '999999999 999999998' }, `${ourServer}`).foreign).toBe(true))
    withBox((box) => {
      box.put('bin/nvidia-smi', FAKE_NVIDIA, true)
      const out = box.run(probeScript(), { env: { NV_FAIL: '1' } }).stdout
      expect(marker(out, 'FOREIGN')).toBe('1')
    })
  })
})

describe.skipIf(!hasBash)('bash: model files', () => {
  const withBox = (fn: (box: Sandbox) => void): void => {
    const box = new Sandbox()
    try {
      box.put('bin/curl', FAKE_CURL, true)
      fn(box)
    } finally {
      box.cleanup()
    }
  }
  const URL = 'https://huggingface.co/o/r/resolve/main/model.gguf'
  /** A 300 KB file with a known sha256 in the fake server. */
  const serve = (box: Sandbox): { size: number; sha: string } => {
    box.sh('head -c 300000 /dev/urandom > "$T/srv/model.gguf"')
    return { size: Number(box.sh('stat -c%s "$T/srv/model.gguf"')), sha: box.sh('sha256sum "$T/srv/model.gguf" | cut -d" " -f1') }
  }

  bashIt('downloads, checks size and sha256, moves the file and reports progress markers', () =>
    withBox((box) => {
      const f = serve(box)
      const r = box.run(hfDownloadScript('m1', [{ url: URL, name: 'model.gguf', size: f.size, sha256: f.sha }]), { data: hfConfigLine(null) })
      expect(r.code).toBe(0)
      expect(markers(r.stdout, 'FILE_DONE')).toEqual(['model.gguf'])
      expect(box.sh('sha256sum "$RZ_ROOT/models/m1/model.gguf" | cut -d" " -f1')).toBe(f.sha)
      expect(box.sh('ls "$RZ_ROOT/models/m1"')).toBe('model.gguf')
    })
  )

  const SIGNED = 'https://cdn.example/blob/model.gguf?sig=s3cret_sig&x=[1]'

  bashIt('downloads a gated file from its signed address, which is never in argv or a file; curl gets it on stdin', () =>
    withBox((box) => {
      const f = serve(box)
      const r = box.run(hfDownloadScript('m1', [{ url: URL, name: 'model.gguf', size: f.size, sha256: f.sha }]), { data: hfConfigLine(SIGNED) })
      expect(r.code).toBe(0)
      expect(box.sh('sha256sum "$RZ_ROOT/models/m1/model.gguf" | cut -d" " -f1')).toBe(f.sha)
      expect(box.sh('cat "$T/curl.argv"')).not.toContain('s3cret_sig')
      expect(box.sh('cat "$T/curl.argv"')).not.toContain('huggingface.co')
      expect(box.sh('cat "$T/curl.argv"')).toContain('--globoff')
      expect(box.sh('cat "$T/curl.stdin"')).toContain('s3cret_sig')
      expect(box.sh('grep -rl s3cret_sig "$RZ_ROOT" | wc -l')).toBe('0')
    })
  )

  bashIt('says with exit 36 that a signed address was refused, and keeps the part file for the next one', () =>
    withBox((box) => {
      const f = serve(box)
      box.sh('mkdir -p "$RZ_ROOT/models/m1"; head -c 100000 "$T/srv/model.gguf" > "$RZ_ROOT/models/m1/model.gguf.part"')
      const file = [{ url: URL, name: 'model.gguf', size: f.size, sha256: f.sha }]
      const r = box.run(hfDownloadScript('m1', file), { data: hfConfigLine(SIGNED), env: { FAKE_CODE: '403' } })
      expect(r.code).toBe(36)
      expect(box.sh('stat -c%s "$RZ_ROOT/models/m1/model.gguf.part"')).toBe('100000')
      expect(box.run(hfDownloadScript('m1', file), { data: hfConfigLine(SIGNED) }).code).toBe(0)
      expect(box.sh('sha256sum "$RZ_ROOT/models/m1/model.gguf" | cut -d" " -f1')).toBe(f.sha)
    })
  )

  bashIt('resumes a .part file with -C - instead of starting over', () =>
    withBox((box) => {
      const f = serve(box)
      box.sh('mkdir -p "$RZ_ROOT/models/m1"; head -c 100000 "$T/srv/model.gguf" > "$RZ_ROOT/models/m1/model.gguf.part"')
      const r = box.run(hfDownloadScript('m1', [{ url: URL, name: 'model.gguf', size: f.size, sha256: f.sha }]), { data: hfConfigLine(null) })
      expect(r.code).toBe(0)
      expect(box.sh('cat "$T/curl.argv"')).toContain('-C')
      expect(box.sh('sha256sum "$RZ_ROOT/models/m1/model.gguf" | cut -d" " -f1')).toBe(f.sha)
    })
  )

  bashIt('trusts a resumed prefix only when the final hash matches: a damaged prefix is thrown away and downloaded once more', () =>
    withBox((box) => {
      const f = serve(box)
      box.sh('mkdir -p "$RZ_ROOT/models/m1"; head -c 100000 /dev/zero > "$RZ_ROOT/models/m1/model.gguf.part"')
      const r = box.run(hfDownloadScript('m1', [{ url: URL, name: 'model.gguf', size: f.size, sha256: f.sha }]), { data: hfConfigLine(null) })
      expect(r.code).toBe(0)
      expect(box.sh('sha256sum "$RZ_ROOT/models/m1/model.gguf" | cut -d" " -f1')).toBe(f.sha)
    })
  )

  bashIt('fails with a sentence when the checksum never matches, and keeps no final file', () =>
    withBox((box) => {
      const f = serve(box)
      const r = box.run(hfDownloadScript('m1', [{ url: URL, name: 'model.gguf', size: f.size, sha256: 'b'.repeat(64) }]), { data: hfConfigLine(null) })
      expect(r.code).toBe(34)
      expect(marker(r.stdout, 'ERR')).toContain('checksum')
      expect(box.sh('ls "$RZ_ROOT/models/m1"')).toBe('')
    })
  )

  bashIt('reports gated, missing and short files', () => {
    withBox((box) => {
      const f = serve(box)
      const r = box.run(hfDownloadScript('m1', [{ url: URL, name: 'model.gguf', size: f.size, sha256: null }]), { data: hfConfigLine(null), env: { FAKE_CODE: '401' } })
      expect(r.code).toBe(31)
      expect(marker(r.stdout, 'ERR')).toContain('token')
    })
    withBox((box) => {
      const r = box.run(hfDownloadScript('m1', [{ url: 'https://huggingface.co/o/r/resolve/main/absent.gguf', name: 'absent.gguf', size: 10, sha256: null }]), { data: hfConfigLine(null) })
      expect(r.code).toBe(32)
    })
    withBox((box) => {
      const f = serve(box)
      const r = box.run(hfDownloadScript('m1', [{ url: URL, name: 'model.gguf', size: f.size + 5, sha256: null }]), { data: hfConfigLine(null) })
      expect(r.code).toBe(33)
      expect(marker(r.stdout, 'ERR')).toContain('incomplete')
    })
  })

  bashIt('asks the server for the size when it is not known, and skips a file that is already there', () =>
    withBox((box) => {
      const f = serve(box)
      const first = box.run(hfDownloadScript('m1', [{ url: URL, name: 'model.gguf', size: null, sha256: null }]), { data: hfConfigLine(null) })
      expect(first.code).toBe(0)
      box.sh('rm -f "$T/curl.log"')
      const again = box.run(hfDownloadScript('m1', [{ url: URL, name: 'model.gguf', size: f.size, sha256: f.sha }]), { data: hfConfigLine(null) })
      expect(markers(again.stdout, 'FILE_DONE')).toEqual(['model.gguf'])
      expect(box.sh('cat "$T/curl.log" 2>/dev/null')).toBe('')
    })
  )

  bashIt('marks files complete only when they exist with that size, and lists state', () =>
    withBox((box) => {
      box.sh('mkdir -p "$RZ_ROOT/models/m1"; printf abcde > "$RZ_ROOT/models/m1/a.gguf"; printf xy > "$RZ_ROOT/models/m1/b.gguf.part"')
      const bad = box.run(markCompleteScript('m1', [{ name: 'a.gguf', size: 6, sha256: '-', source: '-' }]))
      expect(bad.code).toBe(61)
      const ok = box.run(markCompleteScript('m1', [{ name: 'a.gguf', size: 5, sha256: SHA, source: '-' }]))
      expect(marker(ok.stdout, 'COMPLETE')).toBe('1')
      const again = box.run(markCompleteScript('m1', [{ name: 'a.gguf', size: 5, sha256: SHA, source: '-' }]))
      expect(again.code).toBe(0)
      const state = box.run(modelStateScript('m1'))
      expect(markers(state.stdout, 'FILE')).toEqual(['a.gguf|5'])
      expect(markers(state.stdout, 'PART')).toEqual(['b.gguf.part|2'])
      const entries = parseComplete(Buffer.from(marker(state.stdout, 'COMPLETE_B64') ?? '', 'base64').toString())
      expect(entries).toEqual([{ name: 'a.gguf', size: 5, sha256: SHA, source: '-' }])
    })
  )

  bashIt('uploads: append, resume, finish with the right and the wrong hash, discard', () =>
    withBox((box) => {
      const sha = (s: string): string => box.sh(`printf '%s' '${s}' | sha256sum | cut -d" " -f1`)
      expect(markers(box.run(uploadStateScript('m1', 'a.gguf')).stdout, 'PART')).toEqual(['-1'])
      expect(marker(box.run(uploadAppendScript('m1', 'a.gguf'), { data: 'hello ' }).stdout, 'PART')).toBe('6')
      expect(marker(box.run(uploadAppendScript('m1', 'a.gguf'), { data: 'world' }).stdout, 'PART')).toBe('11')
      expect(marker(box.run(uploadStateScript('m1', 'a.gguf')).stdout, 'PART')).toBe('11')
      expect(marker(box.run(uploadFinishScript('m1', 'a.gguf', 12, sha('hello world'))).stdout, 'RESULT')).toBe('short')
      expect(marker(box.run(uploadFinishScript('m1', 'a.gguf', 11, sha('hello WORLD'))).stdout, 'RESULT')).toBe('mismatch')
      expect(marker(box.run(uploadFinishScript('m1', 'a.gguf', 11, sha('hello world'))).stdout, 'RESULT')).toBe('ok')
      const st = box.run(uploadStateScript('m1', 'a.gguf')).stdout
      expect(marker(st, 'FINAL')).toBe('11')
      expect(marker(st, 'PART')).toBe('-1')
      expect(marker(box.run(uploadFinishScript('m1', 'a.gguf', 11, sha('hello world'), 'final')).stdout, 'RESULT')).toBe('ok')
      expect(marker(box.run(uploadFinishScript('m1', 'a.gguf', 11, sha('other'), 'final')).stdout, 'RESULT')).toBe('mismatch')
      box.run(uploadDiscardScript('m1', 'a.gguf'))
      expect(marker(box.run(uploadStateScript('m1', 'a.gguf')).stdout, 'FINAL')).toBe('-1')
    })
  )

  bashIt('uploads bytes that look like a script without running them', () =>
    withBox((box) => {
      const r = box.run(uploadAppendScript('m1', 'a.gguf'), { data: 'echo pwned > "$T/pwned"\nexit 3\n' })
      expect(r.code).toBe(0)
      expect(box.sh('ls "$T/pwned" 2>&1 | grep -c "No such"')).toBe('1')
    })
  )
})

describe.skipIf(!hasBash)('bash: engine install', () => {
  bashIt('downloads both packages, checks them, finds llama-server, writes run-env and the ok marker', () => {
    const box = new Sandbox()
    try {
      box.put('bin/curl', FAKE_CURL, true)
      box.put('bin/apt-get', '#!/bin/bash\necho apt-get "$@" >>"$T/apt.log"\nexit 0', true)
      box.sh(
        'mkdir -p "$T/pk/llama-b12345" "$T/pk2/cudart-llama-b12345-bin-ubuntu-cuda-12.8-x64"; printf \'#!/bin/bash\\necho "version: 1 (b12345)"\\n\' > "$T/pk/llama-b12345/llama-server"; chmod +x "$T/pk/llama-b12345/llama-server"; touch "$T/pk/llama-b12345/libggml.so" "$T/pk2/cudart-llama-b12345-bin-ubuntu-cuda-12.8-x64/libcudart.so.12" "$T/pk2/cudart-llama-b12345-bin-ubuntu-cuda-12.8-x64/libcublas.so.12" "$T/pk2/cudart-llama-b12345-bin-ubuntu-cuda-12.8-x64/libcublasLt.so.12"; tar czf "$T/srv/llama.tgz" -C "$T/pk" llama-b12345; tar czf "$T/srv/cudart.tgz" -C "$T/pk2" .'
      )
      const shaA = box.sh('sha256sum "$T/srv/llama.tgz" | cut -d" " -f1')
      const shaB = box.sh('sha256sum "$T/srv/cudart.tgz" | cut -d" " -f1')
      const script = engineInstallScript({
        tag: 'b12345',
        llama: { url: 'https://github.com/ggml-org/llama.cpp/releases/download/b12345/llama.tgz', sha256: shaA, size: 500 },
        cudart: { url: 'https://github.com/ggml-org/llama.cpp/releases/download/b12345/cudart.tgz', sha256: shaB, size: 500 }
      })
      const r = box.run(script)
      expect(r.code, r.stdout + r.stderr).toBe(0)
      expect(marker(r.stdout, 'ENGINE_OK')).toBe('b12345')
      expect(marker(r.stdout, 'VERSION')).toContain('b12345')
      expect(box.sh('ls -a "$RZ_ROOT/engine/b12345" | tr "\\n" " "')).toContain('.ok')
      const env = box.sh('cat "$RZ_ROOT/engine/b12345/run-env"')
      expect(env).toContain('RZ_SERVER=')
      expect(env).toContain('LD_LIBRARY_PATH=')
      // Both folders are on the library path: llama-server's own and the CUDA runtime's.
      expect(env).toMatch(/engine\/b12345\/llama-b12345:/)
      expect(env).toContain('cudart-llama-b12345-bin-ubuntu-cuda-12.8-x64')
      expect(box.sh('. "$RZ_ROOT/engine/b12345/run-env"; "$RZ_SERVER" --version')).toContain('b12345')
      expect(box.sh('ls -a "$RZ_ROOT/engine" | tr "\\n" " "')).not.toContain('.dl-')
      // Already installed: nothing is downloaded again.
      box.sh('rm -f "$T/curl.log"')
      const again = box.run(script)
      expect(marker(again.stdout, 'ENGINE_OK')).toBe('b12345')
      expect(box.sh('cat "$T/curl.log" 2>/dev/null')).toBe('')
    } finally {
      box.cleanup()
    }
  })

  bashIt('refuses a download whose digest is wrong', () => {
    const box = new Sandbox()
    try {
      box.put('bin/curl', FAKE_CURL, true)
      box.put('bin/apt-get', '#!/bin/bash\nexit 0', true)
      box.sh('echo not-a-tarball > "$T/srv/llama.tgz"; echo x > "$T/srv/cudart.tgz"')
      const r = box.run(
        engineInstallScript({
          tag: 'b12345',
          llama: { url: 'https://github.com/ggml-org/llama.cpp/releases/download/b12345/llama.tgz', sha256: SHA, size: 5 },
          cudart: { url: 'https://github.com/ggml-org/llama.cpp/releases/download/b12345/cudart.tgz', sha256: SHA, size: 5 }
        })
      )
      expect(r.code).toBe(20)
      expect(marker(r.stdout, 'ERR')).toContain('checksum')
      expect(box.sh('ls "$RZ_ROOT/engine/b12345" 2>&1 | grep -c "No such"')).toBe('1')
    } finally {
      box.cleanup()
    }
  })
})

describe.skipIf(!hasBash)('bash: keepalive loop', () => {
  bashIt('touches keepalive for every line and ends at EOF without leaving a process', () => {
    const box = new Sandbox()
    try {
      const r = box.run(keepaliveLoopScript(), { data: 'a\nb\nc\n' })
      expect(r.code).toBe(0)
      expect(box.sh('ls "$RZ_ROOT/keepalive"')).toContain('keepalive')
      expect(box.sh('pgrep -f "read -r _" | wc -l')).toBe('0')
      const t = box.run(touchScript())
      expect(t.code).toBe(0)
    } finally {
      box.cleanup()
    }
  })

  bashIt('refreshes the file when a line arrives', () => {
    const box = new Sandbox()
    try {
      box.sh('mkdir -p "$RZ_ROOT"; touch -d "3 hours ago" "$RZ_ROOT/keepalive"')
      const r = box.run(`touch -d "3 hours ago" "$RZ_ROOT/keepalive"\n${keepaliveLoopScript()}\n`, { data: 'x\n' })
      expect(r.code).toBe(0)
      expect(box.sh('echo $(( $(date +%s) - $(stat -c %Y "$RZ_ROOT/keepalive") ))')).toMatch(/^[0-9]$|^1[0-9]$/)
    } finally {
      box.cleanup()
    }
  })
})

describe.skipIf(!hasBash)('bash: server start, status, log, stop', () => {
  /** An engine folder whose llama-server is a copy of bash, so /proc/<pid>/exe is under it. */
  const engine = (box: Sandbox): void => {
    box.sh('mkdir -p "$RZ_ROOT/engine/b100"; cp /bin/bash "$RZ_ROOT/engine/b100/llama-server"; touch "$RZ_ROOT/engine/b100/.ok"; printf "RZ_SERVER=%s\\nexport LD_LIBRARY_PATH=\\"\\"\\n" "$RZ_ROOT/engine/b100/llama-server" > "$RZ_ROOT/engine/b100/run-env"')
  }
  const start = (box: Sandbox, shellCommand: string, env: Record<string, string> = {}, maxIdle?: number): ReturnType<typeof box.run> =>
    box.run(serverStartScript({ id: 'inst1', tag: 'b100', args: ['-c', shellCommand, '_', PORT_PLACEHOLDER], hash: SHA, expectMiB: 1234, maxIdleSeconds: maxIdle }), { env })
  const withBox = (fn: (box: Sandbox) => void): void => {
    const box = new Sandbox()
    try {
      engine(box)
      fn(box)
    } finally {
      box.cleanup()
    }
  }

  bashIt('starts detached with a run folder, a free port and the port in the arguments', () =>
    withBox((box) => {
      const r = start(box, 'echo booting; sleep 300; :')
      expect(r.code, r.stdout).toBe(0)
      const port = Number(marker(r.stdout, 'PORT'))
      const pid = Number(marker(r.stdout, 'PID'))
      expect(port).toBeGreaterThanOrEqual(20000)
      expect(port).toBeLessThan(40000)
      for (const f of ['args', 'hash', 'log', 'pid', 'port', 'expect', 'sup.sh']) expect(box.sh(`test -f "$RZ_ROOT/run/inst1/${f}" && echo yes`), f).toBe('yes')
      expect(box.sh('cat "$RZ_ROOT/run/inst1/args"')).toContain(String(port))
      expect(box.sh('cat "$RZ_ROOT/run/inst1/args"')).not.toContain(PORT_PLACEHOLDER)
      expect(box.sh('cat "$RZ_ROOT/run/inst1/hash"')).toBe(SHA)
      expect(box.sh('cat "$RZ_ROOT/run/inst1/expect"')).toBe('1234')
      expect(box.sh(`readlink /proc/${pid}/exe`)).toContain('/engine/b100/llama-server')
      const st = box.run(serverStatusScript('inst1'))
      expect(marker(st.stdout, 'ALIVE')).toBe('1')
      expect(marker(st.stdout, 'PORT')).toBe(String(port))
      expect(marker(st.stdout, 'HASH')).toBe(SHA)
      const stop = box.run(serverStopScript('inst1'))
      expect(marker(stop.stdout, 'STOPPED')).toBe('1')
      expect(box.sh(`kill -0 ${pid} 2>&1 | grep -c "No such"`)).toBe('1')
      expect(marker(box.run(serverStatusScript('inst1')).stdout, 'ALIVE')).toBe('0')
      // Stopping again is fine.
      expect(marker(box.run(serverStopScript('inst1')).stdout, 'STOPPED')).toBe('1')
    })
  )

  bashIt('refuses to start a second server for the same instance, and when the engine is missing', () =>
    withBox((box) => {
      expect(start(box, 'sleep 300; :').code).toBe(0)
      const again = start(box, 'sleep 300; :')
      expect(again.code).toBe(42)
      box.run(serverStopScript('inst1'))
      box.sh('rm -f "$RZ_ROOT/engine/b100/.ok"')
      expect(start(box, 'sleep 300; :').code).toBe(40)
    })
  )

  bashIt('reads the log from a byte offset in chunks, as base64, and from the end with a negative offset', () =>
    withBox((box) => {
      expect(start(box, 'echo line-one; echo line-two; sleep 300; :').code).toBe(0)
      box.sh('sleep 0.5')
      const all = box.run(serverLogScript('inst1', 0))
      const text = Buffer.from(marker(all.stdout, 'LOG') ?? '', 'base64').toString()
      expect(text).toContain('line-one')
      expect(text).toContain('line-two')
      const next = Number(marker(all.stdout, 'LOG_NEXT'))
      expect(next).toBe(Buffer.byteLength(text))
      box.sh('echo more-output >> "$RZ_ROOT/run/inst1/log"')
      const rest = box.run(serverLogScript('inst1', next))
      expect(Buffer.from(marker(rest.stdout, 'LOG') ?? '', 'base64').toString()).toBe('more-output\n')
      const tail = box.run(serverLogScript('inst1', -5))
      expect(Buffer.from(marker(tail.stdout, 'LOG') ?? '', 'base64').toString()).toBe('tput\n')
      const beyond = box.run(serverLogScript('inst1', 99999))
      expect(marker(beyond.stdout, 'LOG_RESET')).toBe('1')
      box.run(serverStopScript('inst1'))
    })
  )

  bashIt('tells when the process is gone and still gives the log', () =>
    withBox((box) => {
      expect(start(box, 'echo about-to-die; sleep 1.5').code).toBe(0)
      box.sh('sleep 3')
      const r = box.run(serverLogScript('inst1', 0))
      expect(marker(r.stdout, 'ALIVE')).toBe('0')
      expect(Buffer.from(marker(r.stdout, 'LOG') ?? '', 'base64').toString()).toContain('about-to-die')
    })
  )

  bashIt('reports the log tail when the server dies at once', () =>
    withBox((box) => {
      const r = start(box, 'echo cannot load model; exit 3')
      expect(r.code).toBe(44)
      expect(Buffer.from(marker(r.stdout, 'LOG') ?? '', 'base64').toString()).toContain('cannot load model')
    })
  )

  bashIt('kills a server that ignores TERM', () =>
    withBox((box) => {
      const r = start(box, 'trap "" TERM; while :; do sleep 1; done')
      const pid = Number(marker(r.stdout, 'PID'))
      const stop = box.run(serverStopScript('inst1'), { env: { RZ_STOP_WAIT: '2' } })
      expect(marker(stop.stdout, 'STOPPED')).toBe('1')
      expect(box.sh(`kill -0 ${pid} 2>&1 | grep -c "No such"`)).toBe('1')
    })
  )

  bashIt('never kills a process that is not a binary of our engine', () =>
    withBox((box) => {
      // The stop script ends with exit, so it runs in a subshell and the check follows it.
      const r = box.run('sleep 300 >/dev/null 2>&1 & p=$!; mkdir -p "$RZ_ROOT/run/inst1"; echo $p > "$RZ_ROOT/run/inst1/pid"\n(\n' + serverStopScript('inst1') + '\n)\nkill -0 $p && echo STRANGER-STILL-RUNS; kill $p\n')
      expect(marker(r.stdout, 'NOTOURS')).toBe('1')
      expect(r.stdout).toContain('STRANGER-STILL-RUNS')
    })
  )

  bashIt('its supervisor ends the server when keepalive is older than the limit, and leaves it while keepalive is fresh', () =>
    withBox((box) => {
      box.sh('touch "$RZ_ROOT/keepalive"')
      const r = start(box, 'sleep 300; :', { RZ_SUP_TICK: '1' }, 4)
      const pid = Number(marker(r.stdout, 'PID'))
      box.sh('sleep 2')
      expect(box.sh(`kill -0 ${pid} && echo alive`)).toBe('alive')
      box.sh('touch -d "1 hour ago" "$RZ_ROOT/keepalive"; sleep 4')
      expect(box.sh(`kill -0 ${pid} 2>&1 | grep -c "No such"`)).toBe('1')
      expect(box.sh('cat "$RZ_ROOT/run/inst1/log"')).toContain('keepalive is older')
    })
  )

  bashIt('the launch check reports engine, files, GPUs, foreign processes and the server in one go', () =>
    withBox((box) => {
      box.put('bin/nvidia-smi', FAKE_NVIDIA, true)
      box.sh('mkdir -p "$RZ_ROOT/models/m1"; printf "a.gguf\\t5\\t-\\n" > "$RZ_ROOT/models/m1/.complete"')
      const r = box.run(launchCheckScript({ id: 'inst1', tag: 'b100', slug: 'm1' }), { env: { NV_USED: '200' } })
      expect(marker(r.stdout, 'ENGINE_OK')).toBe('1')
      expect(parseComplete(Buffer.from(marker(r.stdout, 'COMPLETE_B64') ?? '', 'base64').toString())).toEqual([{ name: 'a.gguf', size: 5, sha256: '-', source: '-' }])
      expect(marker(r.stdout, 'FOREIGN')).toBe('0')
      expect(markers(r.stdout, 'RUN')).toEqual([])
      expect(marker(r.stdout, 'GPU')).toContain('NVIDIA Fake GPU')
      const none = box.run(launchCheckScript({ id: 'inst1', tag: 'b900', slug: 'zzz' }))
      expect(marker(none.stdout, 'ENGINE_OK')).toBe('0')
    })
  )

  bashIt('does not remove a model while a server runs on it', () =>
    withBox((box) => {
      box.sh('mkdir -p "$RZ_ROOT/models/m1"; touch "$RZ_ROOT/models/m1/a.gguf"')
      const s = box.run(serverStartScript({ id: 'inst1', tag: 'b100', args: ['-c', 'sleep 300; :', '_', '/x/models/m1/a.gguf', PORT_PLACEHOLDER], hash: SHA, expectMiB: null }))
      expect(s.code).toBe(0)
      box.sh('echo "/models/m1/a.gguf" >> "$RZ_ROOT/run/inst1/args"')
      const refused = box.run(removeModelScript('m1'))
      expect(refused.code).toBe(60)
      expect(box.sh('ls "$RZ_ROOT/models/m1"')).toBe('a.gguf')
      box.run(serverStopScript('inst1'))
      const ok = box.run(removeModelScript('m1'))
      expect(marker(ok.stdout, 'REMOVED')).toBe('1')
      expect(box.sh('ls "$RZ_ROOT/models" | wc -l')).toBe('0')
    })
  )
})

describe.skipIf(!hasBash)('bash: review fixes', () => {
  const engine = (box: Sandbox): void => {
    box.sh('mkdir -p "$RZ_ROOT/engine/b100"; cp /bin/bash "$RZ_ROOT/engine/b100/llama-server"; touch "$RZ_ROOT/engine/b100/.ok"; printf "RZ_SERVER=%s\\nexport LD_LIBRARY_PATH=\\"\\"\\n" "$RZ_ROOT/engine/b100/llama-server" > "$RZ_ROOT/engine/b100/run-env"')
  }
  const withBox = (fn: (box: Sandbox) => void): void => {
    const box = new Sandbox()
    try {
      engine(box)
      box.put('bin/curl', FAKE_CURL, true)
      box.put('bin/nvidia-smi', FAKE_NVIDIA, true)
      fn(box)
    } finally {
      box.cleanup()
    }
  }
  const start = (box: Sandbox, cmd: string, env: Record<string, string> = {}, maxIdle?: number): Out =>
    box.run(serverStartScript({ id: 'inst1', tag: 'b100', args: ['-c', cmd, '_', PORT_PLACEHOLDER], hash: SHA, expectMiB: 1234, maxIdleSeconds: maxIdle }), { env })
  const URL = 'https://huggingface.co/o/r/resolve/main/model.gguf'

  bashIt('a root folder behind a symlink is resolved, so our server is recognised and really stopped', () =>
    withBox((box) => {
      box.sh('ln -s "$T" "$T/link"')
      const env = { RZ_ROOT: `${box.T}/link/root` }
      const r = start(box, 'sleep 300; :', env)
      expect(marker(r.stdout, 'STARTED')).toBe('1')
      const pid = Number(marker(r.stdout, 'PID'))
      expect(marker(box.run(serverStatusScript('inst1'), { env }).stdout, 'ALIVE')).toBe('1')
      const stop = box.run(serverStopScript('inst1'), { env })
      expect(marker(stop.stdout, 'STOPPED')).toBe('1')
      expect(marker(stop.stdout, 'NOTOURS')).toBeUndefined()
      expect(box.sh(`kill -0 ${pid} 2>&1 | grep -c "No such"`)).toBe('1')
    })
  )

  // 1. The limits the user set later survive a restart from vast's site.
  bashIt('onstart takes idle minutes and cap hours from the boot file that arm and disarm keep', () =>
    withBox((box) => {
      box.put('bin/curl', FAKE_VAST_CURL, true)
      const script = onstartScript({ idleMinutes: 45, capHours: 6 }).replace('R=/workspace/rulz-llm; [ -d /workspace ] || R=/root/rulz-llm', 'R=$RZ_ROOT')
      const boot = (content: string): { idle: string; cap: number } => {
        const pre = content === '' ? 'rm -f "$RZ_ROOT/boot"' : "printf '" + content + "' > \"$RZ_ROOT/boot\""
        const r = box.run('mkdir -p "$RZ_ROOT"; ' + pre + '\n' + script + '\nsleep 1\ncat "$R/limits"; kill $(cat "$R/watchdog.pid")', { env: { CONTAINER_ID: '5', CONTAINER_API_KEY: 'k' } })
        return { idle: /idle=(\d+)/.exec(r.stdout)?.[1] ?? '', cap: Number(/cap=(\d+)/.exec(r.stdout)?.[1]) }
      }
      const now = Math.floor(Date.now() / 1000)
      const near = (cap: number, hours: number): boolean => Math.abs(cap - (now + hours * 3600)) < 600
      expect(boot('')).toMatchObject({ idle: '45' })
      expect(near(boot('').cap, 6)).toBe(true)
      const later = boot('bi=7\\nbh=3\\n')
      expect(later.idle).toBe('7')
      expect(near(later.cap, 3)).toBe(true)
      expect(boot('bi=0\\nbh=0\\n')).toEqual({ idle: '0', cap: 0 })
      const bad = boot('bi=abc\\nbh=\\n')
      expect(bad.idle).toBe('45')
      expect(near(bad.cap, 6)).toBe(true)
    })
  )

  bashIt('arm writes the boot file with idle minutes and cap hours, and disarm zeroes it', () =>
    withBox((box) => {
      box.put('bin/curl', FAKE_VAST_CURL, true)
      box.run(armScript({ idleMinutes: 20, capAt: null, origin: 'app', capHours: 12 }), { env: { CONTAINER_ID: '12', CONTAINER_API_KEY: 'k', RZ_TICK: '1' } })
      expect(box.sh('cat "$RZ_ROOT/boot"')).toBe('bi=20\nbh=12')
      box.run(disarmScript())
      expect(box.sh('cat "$RZ_ROOT/boot"')).toBe('bi=0\nbh=0')
      box.sh('kill $(cat "$RZ_ROOT/watchdog.pid")')
    })
  )

  // 2. A reused pid is not our server.
  bashIt('the supervisor leaves a process alone that is not its server any more', () =>
    withBox((box) => {
      expect(start(box, 'sleep 300; :', {}, 100000).code).toBe(0)
      box.run(serverStopScript('inst1'))
      const r = box.run(
        'D=$RZ_ROOT/run/inst1; sleep 300 >/dev/null 2>&1 & p=$!; echo $p > $D/pid; touch -d "1 hour ago" $RZ_ROOT/keepalive; RZ_MAXAGE=1 RZ_SUP_TICK=1 RZ_SUP_KILL=1 timeout 20 bash $D/sup.sh $D $RZ_ROOT/keepalive $RZ_ROOT; echo "sup=$?"; kill -0 $p && echo STRANGER-ALIVE; kill $p'
      )
      expect(r.stdout).toContain('sup=0')
      expect(r.stdout).toContain('STRANGER-ALIVE')
    })
  )

  bashIt('checks the binary and the start time before it kills, in the supervisor and in the stop script', () => {
    const text = serverStartScript({ id: 'abc', tag: 'b100', args: [], hash: SHA, expectMiB: null })
    const sup = text.slice(text.indexOf('RZSUP'), text.lastIndexOf('RZSUP'))
    expect(sup).toContain('/proc/$p/stat')
    expect(sup).toContain('ours && kill "$p"')
    expect(sup).toContain('ours && kill -9 "$p"')
    const stop = serverStopScript('abc')
    expect(stop).toContain('[ "$(t "$p")" = "$st" ]')
    expect(stop.indexOf('kill -KILL')).toBeGreaterThan(stop.indexOf('ours "$p" &&'))
  })

  // 6. A lost pid file does not hide a running server.
  bashIt('the stop script finds a server by RZ_ID in its environment when the pid file is gone, and only then', () =>
    withBox((box) => {
      const r = start(box, 'sleep 300; :')
      const pid = Number(marker(r.stdout, 'PID'))
      expect(box.sh('tr "\\0" "\\n" < /proc/' + pid + '/environ | grep -c "^RZ_ID=inst1$"')).toBe('1')
      box.sh('rm -f "$RZ_ROOT/run/inst1/pid"')
      const stop = box.run(serverStopScript('inst1'))
      expect(marker(stop.stdout, 'STOPPED')).toBe('1')
      expect(box.sh('kill -0 ' + pid + ' 2>&1 | grep -c "No such"')).toBe('1')
      // Another instance id has no such server: nothing else is touched.
      const other = start(box, 'sleep 300; :')
      const pid2 = Number(marker(other.stdout, 'PID'))
      const none = box.run(serverStopScript('nothere1'))
      expect(marker(none.stdout, 'STOPPED')).toBe('1')
      expect(box.sh('kill -0 ' + pid2 + ' && echo alive')).toBe('alive')
      box.run(serverStopScript('inst1'))
    })
  )

  // 4. No transfer, no file.
  bashIt('a download that never succeeded keeps its part file and fails, also with no size and no checksum', () =>
    withBox((box) => {
      box.sh('mkdir -p "$RZ_ROOT/models/m1"; printf partial > "$RZ_ROOT/models/m1/model.gguf.part"')
      const r = box.run(hfDownloadScript('m1', [{ url: URL, name: 'model.gguf', size: null, sha256: null }]), { data: hfConfigLine(null), env: { FAKE_CODE: '500', RZ_RETRY_SLEEP: '0' } })
      expect(r.code).toBe(35)
      expect(marker(r.stdout, 'ERR')).toContain('could not be downloaded')
      expect(markers(r.stdout, 'FILE_DONE')).toEqual([])
      expect(box.sh('ls "$RZ_ROOT/models/m1"')).toBe('model.gguf.part')
      expect(box.sh('cat "$RZ_ROOT/models/m1/model.gguf.part"')).toBe('partial')
    })
  )

  // 5. A file of the same name from another address is not the file.
  bashIt('downloads again when the recorded source of a file differs, and keeps it when the source is the same', () =>
    withBox((box) => {
      const old = 'https://huggingface.co/o/r/resolve/rev1/model.gguf'
      const prep = 'mkdir -p "$RZ_ROOT/models/m1"; head -c 100 /dev/zero | tr "\\0" A > "$RZ_ROOT/models/m1/model.gguf"; head -c 100 /dev/zero | tr "\\0" B > "$T/srv/model.gguf"; printf "model.gguf\\t100\\t-\\t' + old + '\\n" > "$RZ_ROOT/models/m1/.complete"'
      box.sh(prep)
      const same = box.run(hfDownloadScript('m1', [{ url: old, name: 'model.gguf', size: 100, sha256: null }]), { data: hfConfigLine(null) })
      expect(markers(same.stdout, 'FILE_DONE')).toEqual(['model.gguf'])
      expect(box.sh('head -c 3 "$RZ_ROOT/models/m1/model.gguf"')).toBe('AAA')
      const other = box.run(hfDownloadScript('m1', [{ url: URL, name: 'model.gguf', size: 100, sha256: null }]), { data: hfConfigLine(null) })
      expect(other.code).toBe(0)
      expect(box.sh('head -c 3 "$RZ_ROOT/models/m1/model.gguf"')).toBe('BBB')
      expect(box.sh('grep -c . "$RZ_ROOT/models/m1/.complete"')).toBe('0')
    })
  )

  // 5. The launch check reports what is really there.
  bashIt('the launch check emits the real size of each file, not the part files, and every server that runs', () =>
    withBox((box) => {
      box.sh('mkdir -p "$RZ_ROOT/models/m1"; printf abcde > "$RZ_ROOT/models/m1/a.gguf"; printf xy > "$RZ_ROOT/models/m1/b.gguf.part"; printf "a.gguf\\t9\\t-\\t-\\n" > "$RZ_ROOT/models/m1/.complete"')
      const s = start(box, 'sleep 300; :')
      const port = marker(s.stdout, 'PORT')
      const r = box.run(launchCheckScript({ id: 'x', tag: 'b100', slug: 'm1' }))
      expect(markers(r.stdout, 'FILE')).toEqual(['a.gguf|5'])
      expect(markers(r.stdout, 'RUN')).toEqual(['inst1|' + port + '|' + SHA])
      box.run(serverStopScript('inst1'))
      expect(markers(box.run(launchCheckScript({ id: 'x', tag: 'b100', slug: 'm1' })).stdout, 'RUN')).toEqual([])
    })
  )

  bashIt('records the source of a file as the fourth column and reads old three-column lines', () =>
    withBox((box) => {
      box.sh('mkdir -p "$RZ_ROOT/models/m1"; printf abcde > "$RZ_ROOT/models/m1/a.gguf"; printf abc > "$RZ_ROOT/models/m1/b.gguf"; printf "b.gguf\\t3\\t-\\n" > "$RZ_ROOT/models/m1/.complete"')
      const r = box.run(markCompleteScript('m1', [{ name: 'a.gguf', size: 5, sha256: SHA, source: URL }]))
      expect(marker(r.stdout, 'COMPLETE')).toBe('1')
      const entries = parseComplete(box.sh('cat "$RZ_ROOT/models/m1/.complete"'))
      expect(entries).toEqual([
        { name: 'b.gguf', size: 3, sha256: '-', source: '-' },
        { name: 'a.gguf', size: 5, sha256: SHA, source: URL }
      ])
      expect(() => markCompleteScript('m1', [{ name: 'a.gguf', size: 5, sha256: SHA, source: 'https://evil.example/x' }])).toThrow()
    })
  )
})
