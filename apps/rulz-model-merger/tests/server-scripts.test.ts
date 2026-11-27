import { describe, expect, it } from 'vitest'
import { DEFAULT_SETTINGS, type OwnServerSettings } from '@shared/types'
import {
  bashPath,
  buildAliveScript,
  buildPidFileScript,
  buildProbeScript,
  buildRemovePidScript,
  buildStartScript,
  buildStopScript,
  IDENTITY_FUNCTION
} from '../src/main/server/scripts'
import { shellQuote, toWslPath } from '../src/main/server/wsl'

const base: OwnServerSettings = { ...DEFAULT_SETTINGS.ownServer, baseDir: '/home/x/.rulz-model-merger/comfy', comfyDir: '/home/x/ComfyUI' }
const id = { baseDir: '/home/x/.rulz-model-merger/comfy', startTime: '123456' }

describe('wsl path helpers', () => {
  it('converts windows paths', () => {
    expect(toWslPath('C:\\a b\\c')).toBe('/mnt/c/a b/c')
    expect(toWslPath('D:/x/y.sh')).toBe('/mnt/d/x/y.sh')
    expect(toWslPath('\\\\wsl.localhost\\Ubuntu-24.04\\home\\user\\x')).toBe('/home/user/x')
  })
  it('single-quotes for bash', () => {
    expect(shellQuote('a b')).toBe("'a b'")
    expect(shellQuote("it's")).toBe("'it'\\''s'")
    expect(shellQuote('')).toBe("''")
  })
  it('leaves a leading tilde unquoted', () => {
    expect(bashPath('~/ComfyUI')).toBe("~/'ComfyUI'")
    expect(bashPath('~/my dir/x')).toBe("~/'my dir/x'")
    expect(bashPath('/opt/a b')).toBe("'/opt/a b'")
    expect(bashPath('~')).toBe('~')
  })
})

describe('buildStartScript', () => {
  it('starts on the CPU by default, with the absolute base printed, PCI bus order and unbuffered output', () => {
    const s = buildStartScript({ ...base, comfyDir: '~/ComfyUI', baseDir: '~/.rulz-model-merger/comfy' }, 8288)
    expect(s.startsWith('set -e\n')).toBe(true)
    expect(s).toContain("cd ~/'ComfyUI'")
    expect(s).toContain("mkdir -p ~/'.rulz-model-merger/comfy'")
    expect(s).toContain(`RULZ_BASE="$(cd ~/'.rulz-model-merger/comfy' && pwd -P)"`)
    expect(s).toContain('echo "RULZ_BASE=$RULZ_BASE"')
    expect(s).toContain('"$RULZ_BASE/server.pid"')
    expect(s).toContain('echo "RULZ_PID=$$"')
    expect(s).toContain('echo "RULZ_START=$RULZ_START"')
    expect(s).toContain('export CUDA_DEVICE_ORDER=PCI_BUS_ID')
    expect(s).toContain('export PYTHONUNBUFFERED=1')
    expect(s).toContain(
      `exec 'venv/bin/python' main.py --base-directory "$RULZ_BASE" --port 8288 --listen 127.0.0.1 --disable-auto-launch --cpu --models-directory ~/'ComfyUI/models'`
    )
    expect(s).not.toContain('--cuda-device')
    expect(s).not.toContain('extra-model-paths-config')
    expect(s).not.toContain('\r')
  })
  it('records the start time of the process that becomes ComfyUI', () => {
    const s = buildStartScript(base, 1)
    expect(s).toContain("RULZ_START=\"$(sed 's/.*) //' /proc/$$/stat")
    expect(s.indexOf('RULZ_BASE=')).toBeLessThan(s.indexOf('echo "RULZ_PID'))
    expect(s.indexOf('echo "RULZ_PID')).toBeLessThan(s.indexOf('CUDA_DEVICE_ORDER'))
    expect(s.indexOf('CUDA_DEVICE_ORDER')).toBeLessThan(s.indexOf('exec '))
  })
  it('passes --cuda-device for a gpu', () => {
    const s = buildStartScript({ ...base, device: { kind: 'gpu', index: 1, name: 'X' } }, 8290)
    expect(s).toContain('--cuda-device 1')
    expect(s).not.toContain('--cpu')
  })
  it('uses the models folder setting and quotes paths with spaces and apostrophes', () => {
    const s = buildStartScript(
      { ...base, comfyDir: "/home/o'neil/Comfy UI", baseDir: '/data/my base', modelsDir: '/mnt/d/My Models', python: '/usr/bin/python3' },
      9000
    )
    expect(s).toContain("cd '/home/o'\\''neil/Comfy UI'")
    expect(s).toContain("mkdir -p '/data/my base'")
    expect(s).toContain(`--models-directory '/mnt/d/My Models'`)
    expect(s).toContain("exec '/usr/bin/python3' main.py")
  })
  it('adds the custom_nodes yaml with a single-quoted scalar', () => {
    const s = buildStartScript({ ...base, loadCustomNodes: true }, 8288)
    expect(s).toContain('RULZ_COMFY_ABS="$(pwd -P)"')
    expect(s).toContain(`sed "s/'/''/g"`)
    expect(s).toContain(`"  base_path: '$RULZ_YAML_PATH'"`)
    expect(s).toContain('custom_nodes: custom_nodes')
    expect(s).toContain('> "$RULZ_BASE/extra_model_paths.yaml"')
    expect(s).toContain('--extra-model-paths-config "$RULZ_BASE/extra_model_paths.yaml"')
    expect(s.indexOf('RULZ_COMFY_ABS=')).toBeGreaterThan(s.indexOf('cd '))
    expect(s).not.toContain('base_path: "')
  })
  it('splits extra args on whitespace and quotes each', () => {
    const s = buildStartScript({ ...base, extraArgs: "  --fast   --x=a'b  " }, 8288)
    expect(s.split('\n').find((l) => l.startsWith('exec '))!.endsWith("'--fast' '--x=a'\\''b'")).toBe(true)
  })
})

describe('process identity scripts', () => {
  it('reads cmdline as NUL-separated arguments and compares whole values', () => {
    expect(IDENTITY_FUNCTION).toContain("read -r -d '' a")
    expect(IDENTITY_FUNCTION).toContain('*main.py) hasmain=1')
    expect(IDENTITY_FUNCTION).toContain('[ "$prev" = "--base-directory" ] && [ "$a" = "$want" ]')
    expect(IDENTITY_FUNCTION).toContain('"--base-directory=$want"')
    expect(IDENTITY_FUNCTION).toContain('[ "$st" = "$start" ]')
    expect(IDENTITY_FUNCTION).not.toContain('*"$BASE"*')
  })
  it('builds a stop script that checks pid, base dir and start time right before signalling', () => {
    const s = buildStopScript(123, 'INT', id)
    expect(s).toContain("if RULZ_ID_OK 123 '/home/x/.rulz-model-merger/comfy' '123456'; then")
    expect(s).toContain('kill -s INT 123')
    expect(buildStopScript(5, 'KILL', id)).toContain('kill -s KILL 5')
    expect(() => buildStopScript(0, 'INT', id)).toThrow()
  })
  it('builds alive and pid-file scripts', () => {
    expect(buildAliveScript(77, id)).toContain("RULZ_ID_OK 77 '/home/x/.rulz-model-merger/comfy' '123456'")
    const p = buildPidFileScript('~/.rulz-model-merger/comfy')
    expect(p).toContain(`pwd -P`)
    expect(p).toContain('RULZ_STALE=')
    expect(p).toContain('read -r PID START')
  })
  it('removes the pid file only when it still names the pid', () => {
    const s = buildRemovePidScript('/home/x/base/', 42)
    expect(s).toContain("F='/home/x/base/server.pid'")
    expect(s).toContain('[ "$P" = "42" ] && rm -f "$F"')
  })
  it('probes comfy, python and memory', () => {
    const s = buildProbeScript(base)
    expect(s).toContain('RULZ_OK')
    expect(s).toContain("[ -x 'venv/bin/python' ]")
    expect(s).toContain('free -b')
    expect(buildProbeScript({ ...base, python: 'python3' })).toContain("command -v 'python3'")
  })
})
