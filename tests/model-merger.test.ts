import { existsSync, readFileSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'
import { findApp } from '@shared/catalog'

const ROOT = join(__dirname, '..')
const read = (rel: string): string => readFileSync(join(ROOT, rel), 'utf8')

describe('RulZ Model Merger', () => {
  const app = findApp('model-merger')!
  const appDir = join(ROOT, 'apps', 'rulz-model-merger')

  it('is in the catalog as an installer in the ComfyUI category', () => {
    expect(app).toBeDefined()
    expect(app.name).toBe('RulZ Model Merger')
    expect(app.path).toBe('apps/rulz-model-merger')
    expect(app.tagPrefix).toBe('model-merger-v')
    expect(app.category).toBe('ComfyUI')
    if (app.install.kind !== 'installer') throw new Error('not an installer')
    expect(new RegExp(app.install.asset).test('RulZ-Model-Merger-Setup-0.1.0.exe')).toBe(true)
    expect(app.install.uninstallName).toBe('RulZ Model Merger')
    expect(app.launch.target).toBe('RulZ Model Merger.exe')
    expect(app.tagline.length).toBeLessThanOrEqual(72)
  })

  it('names its installer the way electron-builder writes it', () => {
    expect(read('apps/rulz-model-merger/electron-builder.yml')).toContain('artifactName: RulZ-Model-Merger-Setup-${version}.${ext}')
    expect(read('apps/rulz-model-merger/electron-builder.yml')).toContain('productName: RulZ Model Merger')
  })

  it('is built by the release workflow from package.json', () => {
    const version = (JSON.parse(read('apps/rulz-model-merger/package.json')) as { version: string }).version
    expect(version).toMatch(/^\d+\.\d+\.\d+$/)
    const workflow = read('.github/workflows/apps.yml')
    expect(workflow).toContain("'model-merger-v*'")
    expect(workflow).toContain('[model-merger]=rulz-model-merger')
    expect(workflow).toMatch(/'model-merger'\s+= @\{ dir = 'rulz-model-merger'; name = 'RulZ Model Merger'; kind = 'electron' \}/)
    expect(workflow).toMatch(/\$_ -in [^}]*'model-merger'/)
  })

  it('is described in ship.json with versions that agree', () => {
    const ship = JSON.parse(read('.claude/ship.json')) as {
      components: { id: string; version: { files: { path: string; json: string | string[] }[] } }[]
    }
    const c = ship.components.find((x) => x.id === 'model-merger')!
    const pkg = JSON.parse(read('apps/rulz-model-merger/package.json')) as { version: string }
    const lock = JSON.parse(read('apps/rulz-model-merger/package-lock.json')) as { version: string; packages: Record<string, { version: string }> }
    expect(c.version.files.map((f) => f.path)).toContain('apps/rulz-model-merger/package-lock.json')
    expect(lock.version).toBe(pkg.version)
    expect(lock.packages[''].version).toBe(pkg.version)
  })

  it('carries the RulZ icon, a license, a logo notice, its README and the catalog copy', () => {
    for (const f of ['LICENSE', 'LOGO-NOTICE.md', 'README.md', 'build/icon.png', 'build/icon.ico'])
      expect(existsSync(join(appDir, f)), f).toBe(true)
    for (const f of ['README.md', 'icon.png', 'icon.ico'])
      expect(existsSync(join(ROOT, 'catalog', 'model-merger', f)), f).toBe(true)
    expect(read('apps/rulz-model-merger/README.md')).toContain('/wiki/Model-Merger')
  })

  it('is listed in the suite README and the icon and catalog tools', () => {
    expect(read('README.md')).toContain('[RulZ Model Merger](apps/rulz-model-merger)')
    expect(read('README.md')).toContain('`model-merger-v`')
    expect(read('tools/make_app_icons.py')).toContain('"model-merger"')
    expect(read('tools/sync_catalog.py')).toContain('"model-merger"')
  })
})
