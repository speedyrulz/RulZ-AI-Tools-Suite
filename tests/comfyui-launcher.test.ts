import { existsSync, readFileSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'
import { CATALOG, CATEGORIES, findApp } from '@shared/catalog'

const ROOT = join(__dirname, '..')
const read = (rel: string): string => readFileSync(join(ROOT, rel), 'utf8')

describe('RulZ ComfyUI Launcher', () => {
  const app = findApp('comfyui-launcher')!
  const appDir = join(ROOT, 'apps', 'RulZ-ComfyUI-Launcher')

  it('is in the catalog as a portable zip in its own category', () => {
    expect(app).toBeDefined()
    expect(app.name).toBe('RulZ ComfyUI Launcher')
    expect(app.path).toBe('apps/RulZ-ComfyUI-Launcher')
    expect(app.tagPrefix).toBe('comfyui-launcher-v')
    expect(app.branch).toBe('main')
    if (app.install.kind !== 'zip') throw new Error('not a zip install')
    // The release workflow names the zip after the app with spaces turned into dashes.
    expect(new RegExp(app.install.asset).test('RulZ-ComfyUI-Launcher-v1.1.0-win64.zip')).toBe(true)
    expect(app.launch.target).toBe('RulZ ComfyUI Launcher.exe')
    expect(app.folderName).toBe('RulZ ComfyUI Launcher')
    expect(app.tagline.length).toBeLessThanOrEqual(72)
  })

  it('has its category, listed last, shared only with the Model Merger', () => {
    expect(CATEGORIES[CATEGORIES.length - 1].name).toBe('ComfyUI')
    expect(app.category).toBe('ComfyUI')
    expect(CATALOG.filter((a) => a.category === 'ComfyUI').map((a) => a.id)).toEqual(['comfyui-launcher', 'model-merger'])
    // Every app's category is one the home page lists.
    for (const a of CATALOG) expect(CATEGORIES.map((c) => c.name)).toContain(a.category)
  })

  it('needs no requirement but an optional NVIDIA driver', () => {
    expect(app.requirements.map((r) => [r.id, !!r.optional])).toEqual([['nvidia', true]])
  })

  it('keeps one version in its source, read by the release workflow', () => {
    const version = /__version__ = "([^"]+)"/.exec(read('apps/RulZ-ComfyUI-Launcher/rulz_comfy_launcher/__init__.py'))?.[1]
    expect(version).toMatch(/^\d+\.\d+\.\d+$/)
    const workflow = read('.github/workflows/apps.yml')
    expect(workflow).toContain("'comfyui-launcher-v*'")
    expect(workflow).toContain('[comfyui-launcher]=RulZ-ComfyUI-Launcher')
    expect(workflow).toMatch(/'comfyui-launcher'\s+= @\{ dir = 'RulZ-ComfyUI-Launcher'; name = 'RulZ ComfyUI Launcher'; kind = 'python' \}/)
    expect(workflow).toMatch(/'comfyui-launcher' \{ \(Select-String -Path "\$env:DIR\/rulz_comfy_launcher\/__init__\.py"/)
    expect(workflow).toContain('RulZ-ComfyUI-Launcher-v$env:VERSION-win64.zip')
  })

  it('is described in ship.json with a version regex that reads its source', () => {
    const ship = JSON.parse(read('.claude/ship.json')) as {
      components: { id: string; version: { files: { path: string; regex: string }[] }; release: { expectAssets: string[] } }[]
    }
    const c = ship.components.find((x) => x.id === 'comfyui-launcher')!
    expect(c.release.expectAssets).toEqual(['RulZ-ComfyUI-Launcher-v*-win64.zip', 'SHA256SUMS'])
    const file = c.version.files[0]
    const m = new RegExp(file.regex, 'm').exec(read(file.path))
    expect(m?.[1]).toMatch(/^\d+\.\d+\.\d+$/)
  })

  it('builds with its own venv, runs its tests on CI and ignores the venv', () => {
    const build = read('apps/RulZ-ComfyUI-Launcher/build.ps1')
    expect(build).toContain('param([switch]$Test)')
    expect(build).toContain('GITHUB_ACTIONS')
    expect(build).toContain('.venv')
    expect(build).toContain('RulZ ComfyUI Launcher.spec')
    expect(read('apps/RulZ-ComfyUI-Launcher/.gitignore')).toContain('.venv/')
    expect(read('apps/RulZ-ComfyUI-Launcher/RulZ ComfyUI Launcher.spec')).toContain('rulz_comfy_launcher/assets')
  })

  it('carries the RulZ icon, a license and a logo notice naming its files', () => {
    for (const f of ['LICENSE', 'LOGO-NOTICE.md', 'rulz_comfy_launcher/assets/icon.png', 'rulz_comfy_launcher/assets/icon.ico'])
      expect(existsSync(join(appDir, f)), f).toBe(true)
    const notice = read('apps/RulZ-ComfyUI-Launcher/LOGO-NOTICE.md')
    for (const f of ['icon.png', 'icon.ico', 'logo_full_white.png', 'logo_full_black.png'])
      expect(notice).toContain(`rulz_comfy_launcher/assets/${f}`)
  })

  it('has its README with the download and wiki links, and the catalog copy', () => {
    const readme = read('apps/RulZ-ComfyUI-Launcher/README.md')
    expect(readme).toContain('/wiki/ComfyUI-Launcher')
    expect(readme).toContain('under **ComfyUI**')
    expect(readme).toContain("Open the RulZFlow page instead of ComfyUI's")
    for (const f of ['README.md', 'icon.png', 'icon.ico'])
      expect(existsSync(join(ROOT, 'catalog', 'comfyui-launcher', f)), f).toBe(true)
    expect(existsSync(join(ROOT, 'catalog', 'comfyui-launcher', 'screenshots'))).toBe(true)
  })

  it('is listed in the suite README and the icon and catalog tools', () => {
    const readme = read('README.md')
    expect(readme).toContain('[RulZ ComfyUI Launcher](apps/RulZ-ComfyUI-Launcher)')
    expect(readme).toContain('`comfyui-launcher-v`')
    expect(read('tools/make_app_icons.py')).toContain('"comfyui-launcher"')
    expect(read('tools/sync_catalog.py')).toContain('"comfyui-launcher"')
  })

  it('never links to the private RulZFlow repository', () => {
    for (const rel of ['apps/RulZ-ComfyUI-Launcher/README.md', 'src/shared/catalog.ts'])
      expect(read(rel).toLowerCase(), rel).not.toMatch(/github\.com\/[^\s)]*rulzflow/)
  })
})
