import { existsSync, readFileSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'
import { findApp } from '@shared/catalog'

const ROOT = join(__dirname, '..')
const read = (rel: string): string => readFileSync(join(ROOT, rel), 'utf8')

describe('RulZ Huggingface Manager', () => {
  const app = findApp('hf-manager')!
  const appDir = join(ROOT, 'apps', 'RulZ-Huggingface-Manager')

  it('is in the catalog as a source install from its own branch', () => {
    expect(app).toBeDefined()
    expect(app.path).toBe('apps/RulZ-Huggingface-Manager')
    expect(app.tagPrefix).toBe('hf-manager-v')
    expect(app.branch).toBe('app/hf-manager')
    expect(app.install.kind).toBe('source')
    expect(app.requirements.find((r) => r.id === 'python')?.optional).toBeFalsy()
    expect(app.requirements.find((r) => r.id === 'nvidia')).toBeUndefined()
  })

  it('sets up a venv from requirements.txt and launches its start script with the RulZ icon', () => {
    if (app.install.kind !== 'source') throw new Error('not a source install')
    const steps = app.install.setup.map((s) => s.run)
    expect(steps[0]).toEqual(['python', '-m', 'venv', '.venv'])
    expect(steps[1]).toEqual(['.venv\\Scripts\\python.exe', '-m', 'pip', 'install', '-r', 'requirements.txt'])
    expect(read('apps/RulZ-Huggingface-Manager/requirements.txt')).toMatch(/^nicegui/m)
    expect(app.launch.target).toBe('start-windows.bat')
    expect(existsSync(join(appDir, app.launch.target))).toBe(true)
    // The start script uses the .venv the setup steps made instead of creating its own.
    expect(read('apps/RulZ-Huggingface-Manager/start-windows.bat')).toContain('set "VENV=.venv"')
    expect(existsSync(join(appDir, app.launch.icon!.replace(/\\/g, '/')))).toBe(true)
  })

  it('carries the RulZ logo and keeps it out of the MIT license', () => {
    for (const f of ['static/logo-dark.png', 'static/logo-light.png', 'static/icon.png'])
      expect(existsSync(join(appDir, 'rulz_hf_manager', f)), f).toBe(true)
    expect(read('apps/RulZ-Huggingface-Manager/LOGO-NOTICE.md')).toContain('rulz_hf_manager/static/logo-dark.png')
    expect(read('apps/RulZ-Huggingface-Manager/pyproject.toml')).toContain('rulz_hf_manager = ["static/*"]')
    const ignore = read('apps/RulZ-Huggingface-Manager/.gitignore')
    for (const p of ['.venv/', '__pycache__/', '.nicegui/']) expect(ignore).toContain(p)
  })

  it('keeps one version in its source, read by the release workflow', () => {
    const version = /__version__ = '([^']+)'/.exec(read('apps/RulZ-Huggingface-Manager/rulz_hf_manager/__init__.py'))?.[1]
    expect(version).toMatch(/^\d+\.\d+\.\d+$/)
    expect(read('apps/RulZ-Huggingface-Manager/pyproject.toml')).toContain(`version = "${version}"`)
    expect(read('.github/workflows/apps.yml')).toMatch(
      /'hf-manager'\s+\{ \(Select-String -Path "\$env:DIR\/rulz_hf_manager\/__init__\.py"/
    )
  })

  it('has its catalog README, icon and screenshots', () => {
    for (const f of ['README.md', 'icon.png', 'icon.ico', 'screenshots/01-my-hub.png'])
      expect(existsSync(join(ROOT, 'catalog', 'hf-manager', f)), f).toBe(true)
  })
})
