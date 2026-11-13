import { existsSync, readdirSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'
import { CATALOG, findApp, launchEnv } from '@shared/catalog'

const CATALOG_DIR = join(__dirname, '..', 'catalog')

describe('catalog', () => {
  it('has unique ids and folder names', () => {
    expect(new Set(CATALOG.map((a) => a.id)).size).toBe(CATALOG.length)
    expect(new Set(CATALOG.map((a) => a.folderName.toLowerCase())).size).toBe(CATALOG.length)
  })

  it.each(CATALOG.map((a) => [a.id, a] as const))('%s ships its README, icons and a screenshot folder', (id) => {
    for (const f of ['README.md', 'icon.png', 'icon.ico']) expect(existsSync(join(CATALOG_DIR, id, f)), f).toBe(true)
    expect(existsSync(join(CATALOG_DIR, id, 'screenshots'))).toBe(true)
  })

  it.each(CATALOG.map((a) => [a.id, a] as const))('%s has valid install patterns', (_id, app) => {
    const spec = app.install
    if (spec.kind !== 'source') {
      // A file name built from the pattern ("[\\d.]+" -> "1.2.3") must match it; a lost backslash breaks this.
      const sample = spec.asset.replace('[\\d.]+', '1.2.3').replace(/\\./g, '.').replace(/^\^|\$$/g, '')
      expect(new RegExp(spec.asset, 'i').test(sample), sample).toBe(true)
    }
    for (const c of app.components ?? []) for (const p of c.assets) expect(() => new RegExp(p)).not.toThrow()
    expect(app.launch.target).not.toMatch(/^[\\/]|^[a-z]:/i)
  })

  it('lists screenshots as images only', () => {
    for (const app of CATALOG) {
      const dir = join(CATALOG_DIR, app.id, 'screenshots')
      for (const f of readdirSync(dir)) expect(f).toMatch(/\.(png|jpe?g|webp|gif|md)$/i)
    }
  })

  it.each(CATALOG.map((a) => [a.id, a] as const))('%s has valid launch options', (_id, app) => {
    const options = app.launch.options ?? []
    expect(new Set(options.map((o) => o.id)).size).toBe(options.length)
    for (const o of options) {
      expect(o.label).toBeTruthy()
      for (const [k, v] of Object.entries(o.env)) {
        expect(k).toMatch(/^[A-Z_][A-Z0-9_]*$/)
        expect(typeof v).toBe('string')
      }
    }
  })

  it('starts RulzTrainer with RULZ_TAILSCALE=1 only while its Tailscale option is on', () => {
    const trainer = findApp('rulz-trainer')!
    expect(launchEnv(trainer, ['tailscale'])).toEqual({ RULZ_TAILSCALE: '1' })
    expect(launchEnv(trainer, [])).toEqual({})
    expect(launchEnv(trainer, undefined)).toEqual({})
    expect(launchEnv(trainer, ['gone'])).toEqual({})
    // A hand-edited string in settings.json must not match by substring.
    expect(launchEnv(trainer, 'tailscale' as unknown as string[])).toEqual({})
    // Another app's option ids do nothing.
    expect(launchEnv(findApp('image-organizer')!, ['tailscale'])).toEqual({})
  })
})
