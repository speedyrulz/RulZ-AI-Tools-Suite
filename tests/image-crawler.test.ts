import { existsSync, readFileSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'
import { CATALOG, findApp } from '@shared/catalog'
import { spawnArgs } from '../src/main/process'

const ROOT = join(__dirname, '..')
const read = (rel: string): string => readFileSync(join(ROOT, rel), 'utf8')

describe('workflows', () => {
  it.each(CATALOG.map((a) => [a.id, a] as const))('%s has a build and release tag in apps.yml', (id, app) => {
    const yml = read('.github/workflows/apps.yml')
    const dir = app.path.replace(/^apps\//, '')
    expect(yml).toContain(`- '${app.tagPrefix}*'`)
    expect(yml).toContain(`[${id}]=${dir}`)
    expect(yml).toMatch(new RegExp(`'${id}'\\s*= @\\{ dir = '${dir}'`))
  })

  it.each(CATALOG.filter((a) => a.install.kind === 'source').map((a) => [a.id, a] as const))(
    '%s gets its install branch from app-branches.yml',
    (id, app) => {
      const yml = read('.github/workflows/app-branches.yml')
      const dir = app.path.replace(/^apps\//, '')
      expect(app.branch).toBe(`app/${id}`)
      expect(yml).toContain(`${dir}:${id}`)
      expect(yml).toContain(`'${app.path}/**'`)
    }
  )
})

describe('RulZ Image Crawler', () => {
  const app = findApp('rulz-image-crawler')!
  const appDir = join(ROOT, 'apps', 'rulz-image-crawler')

  it('is in the catalog as a source install from its own branch', () => {
    expect(app).toBeDefined()
    expect(app.path).toBe('apps/rulz-image-crawler')
    expect(app.tagPrefix).toBe('rulz-image-crawler-v')
    expect(app.branch).toBe('app/rulz-image-crawler')
    expect(app.install.kind).toBe('source')
    expect(app.requirements.find((r) => r.id === 'python')?.optional).toBeFalsy()
    expect(app.requirements.find((r) => r.id === 'nvidia')?.optional).toBeFalsy()
  })

  it('sets up a venv, installs both packages and Chromium, then downloads the model', () => {
    if (app.install.kind !== 'source') throw new Error('not a source install')
    const steps = app.install.setup.map((s) => s.run)
    expect(steps[0]).toEqual(['python', '-m', 'venv', '.venv'])
    for (const run of steps.slice(1)) expect(run[0]).toBe('.venv\\Scripts\\python.exe')
    const pip = steps.find((r) => r.includes('pip'))!
    for (const pkg of ['image_detection', 'crawler_app']) {
      expect(pip).toContain(`.\\${pkg}`)
      expect(existsSync(join(appDir, pkg, 'pyproject.toml')), pkg).toBe(true)
    }
    expect(steps.some((r) => r.join(' ').endsWith('playwright install chromium'))).toBe(true)
    // Scripts the steps run are files in the app's folder.
    for (const run of steps)
      for (const arg of run.slice(1).filter((a) => a.endsWith('.py')))
        expect(existsSync(join(appDir, arg.replace(/\\/g, '/'))), arg).toBe(true)
  })

  it('launches the start script with its icon, from folders with spaces', () => {
    expect(existsSync(join(appDir, app.launch.target))).toBe(true)
    expect(existsSync(join(appDir, app.launch.icon!.replace(/\\/g, '/')))).toBe(true)
    const r = spawnArgs(`C:\\Programs\\RulZ Image Crawler\\${app.launch.target}`, [])
    expect(r.verbatim).toBe(true)
    expect(r.args[3]).toBe('""C:\\Programs\\RulZ Image Crawler\\Start Image Crawler.bat""')
  })

  it('finds what the launcher installs: the venv and the llama.cpp component', () => {
    expect(read('apps/rulz-image-crawler/start_app.ps1')).toContain('.venv\\Scripts\\python.exe')
    const llama = app.components?.find((c) => c.id === 'llama-cpp')
    expect(llama?.dest).toBe('llama.cpp')
    expect(read('apps/rulz-image-crawler/image_detection/scripts/start_llama_server.ps1')).toContain(
      `"${llama!.dest}\\llama-server.exe"`
    )
    const ignore = read('apps/rulz-image-crawler/.gitignore')
    for (const p of ['.venv/', 'logs/', 'output/', 'image_detection/LLM/', 'llama.cpp/']) expect(ignore).toContain(p)
  })

  it('keeps one version in its source, read by the release workflow', () => {
    const version = /__version__ = "([^"]+)"/.exec(read('apps/rulz-image-crawler/crawler_app/__init__.py'))?.[1]
    expect(version).toMatch(/^\d+\.\d+\.\d+$/)
    for (const pkg of ['crawler_app', 'image_detection'])
      expect(read(`apps/rulz-image-crawler/${pkg}/pyproject.toml`)).toContain(`version = "${version}"`)
    expect(read('.github/workflows/apps.yml')).toContain(`"$env:DIR/crawler_app/__init__.py"`)
  })
})
