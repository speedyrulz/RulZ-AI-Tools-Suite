import { existsSync, readFileSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'
import { findApp } from '@shared/catalog'

const ROOT = join(__dirname, '..')
const read = (rel: string): string => readFileSync(join(ROOT, rel), 'utf8')

describe('RulZ Image Preparer', () => {
  const app = findApp('image-preparer')!
  const appDir = join(ROOT, 'apps', 'RulZ-Image-Preparer')

  it('is in the catalog as a source install from its own branch', () => {
    expect(app).toBeDefined()
    expect(app.path).toBe('apps/RulZ-Image-Preparer')
    expect(app.tagPrefix).toBe('image-preparer-v')
    expect(app.branch).toBe('app/image-preparer')
    expect(app.install.kind).toBe('source')
    expect(app.requirements.find((r) => r.id === 'python')?.optional).toBeFalsy()
    // The GPU is only for the optional AI mode.
    expect(app.requirements.find((r) => r.id === 'nvidia')?.optional).toBe(true)
  })

  it('sets up a venv from requirements.txt and launches run.py with its icon', () => {
    if (app.install.kind !== 'source') throw new Error('not a source install')
    const steps = app.install.setup.map((s) => s.run)
    expect(steps[0]).toEqual(['python', '-m', 'venv', '.venv'])
    expect(steps[1]).toEqual(['.venv\\Scripts\\python.exe', '-m', 'pip', 'install', '-r', 'requirements.txt'])
    expect(existsSync(join(appDir, 'requirements.txt'))).toBe(true)
    expect(app.launch.target).toBe('.venv\\Scripts\\pythonw.exe')
    expect(existsSync(join(appDir, app.launch.args![0]))).toBe(true)
    expect(existsSync(join(appDir, app.launch.icon!.replace(/\\/g, '/')))).toBe(true)
  })

  it('keeps the optional AI requirements out of the default install', () => {
    expect(read('apps/RulZ-Image-Preparer/requirements.txt')).not.toMatch(/^torch/m)
    expect(read('apps/RulZ-Image-Preparer/requirements-ai.txt')).toMatch(/^torch/m)
    const ignore = read('apps/RulZ-Image-Preparer/.gitignore')
    for (const p of ['.venv/', '__pycache__/']) expect(ignore).toContain(p)
  })

  it('keeps one version in its source, read by the release workflow', () => {
    const version = /APP_VERSION = "([^"]+)"/.exec(read('apps/RulZ-Image-Preparer/rulz/config.py'))?.[1]
    expect(version).toMatch(/^\d+\.\d+\.\d+$/)
    expect(read('.github/workflows/apps.yml')).toMatch(/'image-preparer'\s+\{ \(Select-String -Path "\$env:DIR\/rulz\/config\.py"/)
  })

  it('has its catalog README, icon and screenshots', () => {
    for (const f of ['README.md', 'icon.png', 'icon.ico', 'screenshots/01-grid.png'])
      expect(existsSync(join(ROOT, 'catalog', 'image-preparer', f)), f).toBe(true)
  })
})
