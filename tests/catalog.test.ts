import { existsSync, readdirSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'
import { CATALOG } from '@shared/catalog'

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
    if (spec.kind !== 'source') expect(() => new RegExp(spec.asset)).not.toThrow()
    for (const c of app.components ?? []) for (const p of c.assets) expect(() => new RegExp(p)).not.toThrow()
    expect(app.launch.target).not.toMatch(/^[\\/]|^[a-z]:/i)
  })

  it('lists screenshots as images only', () => {
    for (const app of CATALOG) {
      const dir = join(CATALOG_DIR, app.id, 'screenshots')
      for (const f of readdirSync(dir)) expect(f).toMatch(/\.(png|jpe?g|webp|gif|md)$/i)
    }
  })
})
