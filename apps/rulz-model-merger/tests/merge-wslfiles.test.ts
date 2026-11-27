import { execFileSync } from 'node:child_process'
import { existsSync, mkdirSync, mkdtempSync, readFileSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'
import { DEFAULT_SETTINGS } from '@shared/types'
import type { WslResult, WslRunner } from '../src/main/server/wsl'
import {
  buildListOutputScript,
  buildMoveScript,
  buildResolveModelsScript,
  freeMemoryBytes,
  listOutputCandidates,
  moveNoOverwrite,
  parseOutputCandidates,
  resolveModelsDir,
  sourcePath,
  toUncPath
} from '../src/main/merge/wslFiles'

function fakeWsl(reply: (script: string) => Partial<WslResult>): WslRunner & { scripts: string[] } {
  const scripts: string[] = []
  return {
    scripts,
    run: async (_d, script) => {
      scripts.push(script)
      return { code: 0, stdout: '', stderr: '', ...reply(script) }
    },
    spawn: () => {
      throw new Error('not used')
    }
  }
}

describe('toUncPath / sourcePath', () => {
  it('builds a UNC path', () => {
    expect(toUncPath('Ubuntu-24.04', '/home/jeff/models/a.safetensors')).toBe('\\\\wsl.localhost\\Ubuntu-24.04\\home\\jeff\\models\\a.safetensors')
  })
  it('joins kind folder and name, sub-folders allowed', () => {
    expect(sourcePath('/m', 'checkpoint', 'x/y.safetensors')).toBe('/m/checkpoints/x/y.safetensors')
    expect(sourcePath('/m/', 'diffusion', 'a b.safetensors')).toBe('/m/diffusion_models/a b.safetensors')
    expect(sourcePath('/m', 'lora', 'sub/x.safetensors')).toBe('/m/loras/sub/x.safetensors')
  })
  it('rejects .. and absolute names', () => {
    expect(() => sourcePath('/m', 'diffusion', '../x.safetensors')).toThrow()
    expect(() => sourcePath('/m', 'diffusion', 'a/../../x')).toThrow()
    expect(() => sourcePath('/m', 'diffusion', '/etc/passwd')).toThrow()
    expect(() => sourcePath('/m', 'diffusion', 'C:\\x')).toThrow()
    expect(() => sourcePath('/m', 'diffusion', '')).toThrow()
    expect(() => sourcePath('/m', 'lora', '../x.safetensors')).toThrow()
    expect(() => sourcePath('/m', 'lora', '/etc/passwd')).toThrow()
  })
})

describe('scripts', () => {
  it('quote paths with spaces and quotes, LF only', () => {
    const list = buildListOutputScript("/home/a b/it's/output", 'diffusion_models', "my 'm'")
    expect(list).toContain("cd '/home/a b/it'\\''s/output/diffusion_models'")
    expect(list).toContain("STEM='my '\\''m'\\'''")
    expect(list).not.toContain('\r')
    const mv = buildMoveScript("/o/a b/it's.safetensors", '/m/x y/merged')
    expect(mv).toContain("SRC='/o/a b/it'\\''s.safetensors'")
    expect(mv).toContain("DIR='/m/x y/merged'")
    expect(mv).toContain('mv -n --')
    expect(mv).not.toMatch(/\brm\b/)
    expect(mv).not.toContain('\r')
    const res = buildResolveModelsScript({ ...DEFAULT_SETTINGS.ownServer, modelsDir: '/m o/d' })
    expect(res).toContain("cd '/m o/d'")
  })
})

describe('wsl helpers', () => {
  it('resolves the models dir', async () => {
    const w = fakeWsl(() => ({ stdout: 'motd\n/home/jeff/ComfyUI/models\n' }))
    expect(await resolveModelsDir(w, 'U', DEFAULT_SETTINGS.ownServer)).toBe('/home/jeff/ComfyUI/models')
    await expect(resolveModelsDir(fakeWsl(() => ({ code: 1 })), 'U', DEFAULT_SETTINGS.ownServer)).rejects.toThrow(/not found/)
  })
  it('parses candidate lines robustly', () => {
    const out = 'noise\nRULZ_FILE|100|1700000000|merged_00001_.safetensors\nRULZ_FILE|x|1|bad\nRULZ_FILE|5|6|we|ird_00002_.safetensors\r\nRULZ_FILE|5\n'
    expect(parseOutputCandidates(out, '/o/', 'diffusion_models')).toEqual([
      { path: '/o/diffusion_models/merged_00001_.safetensors', name: 'merged_00001_.safetensors', sizeBytes: 100, mtime: 1700000000 },
      { path: '/o/diffusion_models/we|ird_00002_.safetensors', name: 'we|ird_00002_.safetensors', sizeBytes: 5, mtime: 6 }
    ])
  })
  it('lists candidates and fails on a WSL error', async () => {
    const w = fakeWsl(() => ({ stdout: 'RULZ_FILE|1|2|m_00001_.safetensors\n' }))
    expect(await listOutputCandidates(w, 'U', '/o', 'checkpoints', 'm')).toHaveLength(1)
    await expect(listOutputCandidates(fakeWsl(() => ({ code: null, stderr: 'boom' })), 'U', '/o', 'x', 'm')).rejects.toThrow(/boom/)
  })
  it('reads free memory', async () => {
    expect(await freeMemoryBytes(fakeWsl(() => ({ stdout: 'RULZ_MEM=2048\n' })), 'U')).toBe(2048 * 1024)
    expect(await freeMemoryBytes(fakeWsl(() => ({ stdout: '' })), 'U')).toBeNull()
    expect(await freeMemoryBytes(fakeWsl(() => ({ code: 1 })), 'U')).toBeNull()
  })
  it('moves and reports the final path and sidecar', async () => {
    const w = fakeWsl(() => ({ stdout: 'RULZ_SIDECAR=/m/merged/a-2.safetensors.recipe.json\nRULZ_MOVED=/m/merged/a-2.safetensors\n' }))
    expect(await moveNoOverwrite(w, 'U', '/o/a.safetensors', '/m/merged')).toEqual({
      path: '/m/merged/a-2.safetensors',
      sidecar: '/m/merged/a-2.safetensors.recipe.json'
    })
    await expect(moveNoOverwrite(fakeWsl(() => ({ code: 3, stdout: 'RULZ_ERR=The file is no longer there.\n' })), 'U', '/a', '/b')).rejects.toThrow('no longer there')
    await expect(moveNoOverwrite(fakeWsl(() => ({ code: 1 })), 'U', '/a', '/b')).rejects.toThrow(/not moved/)
  })
})

describe('move script in a real bash', () => {
  it.skipIf(process.platform === 'win32')('picks name-2 and never overwrites', () => {
    const d = mkdtempSync(join(tmpdir(), 'mv-'))
    writeFileSync(join(d, 'a.safetensors'), 'new')
    writeFileSync(join(d, 'a.safetensors.recipe.json'), '{}')
    mkdirSync(join(d, 'merged'))
    writeFileSync(join(d, 'merged', 'a.safetensors'), 'old')
    const out = execFileSync('bash', ['-c', buildMoveScript(join(d, 'a.safetensors'), join(d, 'merged'))]).toString()
    expect(out).toContain('a-2.safetensors')
    expect(readFileSync(join(d, 'merged', 'a.safetensors'), 'utf8')).toBe('old')
    expect(readFileSync(join(d, 'merged', 'a-2.safetensors'), 'utf8')).toBe('new')
    expect(existsSync(join(d, 'merged', 'a-2.safetensors.recipe.json'))).toBe(true)
  })
})
