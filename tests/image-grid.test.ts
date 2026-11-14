import { existsSync, readFileSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'
import { findApp } from '@shared/catalog'

const ROOT = join(__dirname, '..')
const read = (rel: string): string => readFileSync(join(ROOT, rel), 'utf8')

describe('RulZ Image Grid', () => {
  const app = findApp('image-grid')!

  it('is in the catalog as a portable zip with one .exe', () => {
    expect(app).toBeDefined()
    expect(app.path).toBe('apps/RulZ-Image-Grid')
    expect(app.tagPrefix).toBe('image-grid-v')
    if (app.install.kind !== 'zip') throw new Error('not a zip install')
    // The release workflow names the zip after the app with spaces turned into dashes.
    expect(new RegExp(app.install.asset).test('RulZ-Image-Grid-v1.0.0-win64.zip')).toBe(true)
    expect(app.launch.target).toBe('RulZ Image Grid.exe')
    expect(app.requirements.map((r) => r.id)).toEqual(['dotnet-fx'])
  })

  it('keeps one version in its source, read by the release workflow', () => {
    const version = /AssemblyVersion\("(\d+\.\d+\.\d+)\.\d+"\)/.exec(read('apps/RulZ-Image-Grid/src/Program.cs'))?.[1]
    expect(version).toMatch(/^\d+\.\d+\.\d+$/)
    const workflow = read('.github/workflows/apps.yml')
    expect(workflow).toContain("'image-grid-v*'")
    expect(workflow).toContain("[image-grid]=RulZ-Image-Grid")
    expect(workflow).toMatch(/'image-grid'\s+= @\{ dir = 'RulZ-Image-Grid'; name = 'RulZ Image Grid'; kind = 'dotnet' \}/)
    expect(workflow).toMatch(/\{ \$_ -in 'image-tournament', 'image-grid' \}/)
  })

  it('builds from source without a checked-in .exe', () => {
    expect(existsSync(join(ROOT, 'apps', 'RulZ-Image-Grid', 'build.ps1'))).toBe(true)
    expect(read('apps/RulZ-Image-Grid/.gitignore')).toContain('bin/')
  })

  it('has its catalog README, icon and screenshots', () => {
    for (const f of ['README.md', 'icon.png', 'icon.ico', 'screenshots/01-dark.png'])
      expect(existsSync(join(ROOT, 'catalog', 'image-grid', f)), f).toBe(true)
  })
})
