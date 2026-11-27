import { describe, expect, it } from 'vitest'
import { buildListLorasScript, buildStagingScript, listLoras, parseListLoras, resolveStagingDir } from '../src/main/loras/files'
import type { WslRunner } from '../src/main/server/wsl'

describe('buildListLorasScript', () => {
  it('quotes the folder and follows symlinks', () => {
    const s = buildListLorasScript("/home/je ff/Comfy'UI/models/")
    expect(s).toContain("cd '/home/je ff/Comfy'\\''UI/models/loras'")
    expect(s).toContain('find -L .')
    expect(s).not.toContain('\r')
  })
})

describe('parseListLoras', () => {
  it('keeps names with spaces and sub-folders, sorts, drops duplicates and bad names', () => {
    const out = [
      'RULZ_LORA|zeta.safetensors',
      'RULZ_LORA|Alpha Beta.safetensors',
      'RULZ_LORA|sub/dir/b.safetensors',
      'RULZ_LORA|zeta.safetensors',
      'RULZ_LORA|../evil.safetensors',
      'RULZ_LORA|a/./b.safetensors',
      'RULZ_LORA|',
      'noise line',
      'RULZ_LORA|with\rcr.safetensors'
    ].join('\r\n')
    expect(parseListLoras(out)).toEqual(['Alpha Beta.safetensors', 'sub/dir/b.safetensors', 'zeta.safetensors'])
  })

  it('skips lines that are not .safetensors names', () => {
    expect(parseListLoras('RULZ_LORA|a\nRULZ_LORA|b.safetensors')).toEqual(['b.safetensors'])
  })
  it('the script reads NUL-separated names and skips control characters', () => {
    const s = buildListLorasScript('/m')
    expect(s).toContain('-print0')
    expect(s).toContain('[[:cntrl:]]')
  })
})

describe('buildStagingScript', () => {
  it('lets bash expand a leading ~ and quotes the rest', () => {
    const s = buildStagingScript('~/.rulz model merger/comfy')
    expect(s).toContain(`DIR="$HOME"/'.rulz model merger/comfy'/lora-merge`)
    expect(s).toContain('mkdir -p')
  })
  it('handles plain ~, absolute paths with quotes and trailing slashes', () => {
    expect(buildStagingScript('~')).toContain('DIR="$HOME"/lora-merge')
    expect(buildStagingScript("/data/it's/")).toContain("DIR='/data/it'\\''s'/lora-merge")
  })
})

function fakeWsl(reply: { code: number | null; stdout: string; stderr?: string }): WslRunner & { scripts: string[] } {
  const scripts: string[] = []
  return {
    scripts,
    run: async (_d, script) => {
      scripts.push(script)
      return { code: reply.code, stdout: reply.stdout, stderr: reply.stderr ?? '' }
    },
    spawn: () => {
      throw new Error('no spawn')
    }
  }
}

describe('runners', () => {
  it('listLoras parses the output and fails with the WSL message', async () => {
    expect(await listLoras(fakeWsl({ code: 0, stdout: 'RULZ_LORA|b.safetensors\nRULZ_LORA|a.safetensors\n' }), 'U', '/m')).toEqual([
      'a.safetensors',
      'b.safetensors'
    ])
    await expect(listLoras(fakeWsl({ code: 1, stdout: '', stderr: 'boom' }), 'U', '/m')).rejects.toThrow('boom')
  })

  it('resolveStagingDir returns the printed absolute path', async () => {
    const w = fakeWsl({ code: 0, stdout: 'RULZ_STAGING=/home/jeff/.rulz/comfy/lora-merge\n' })
    expect(await resolveStagingDir(w, 'U', '~/.rulz/comfy')).toBe('/home/jeff/.rulz/comfy/lora-merge')
  })

  it('resolveStagingDir surfaces script errors and a missing path', async () => {
    await expect(resolveStagingDir(fakeWsl({ code: 3, stdout: 'RULZ_ERR=Could not create the staging folder.\n' }), 'U', '/x')).rejects.toThrow(
      'Could not create the staging folder.'
    )
    await expect(resolveStagingDir(fakeWsl({ code: 0, stdout: '' }), 'U', '/x')).rejects.toThrow('not made')
  })
})
