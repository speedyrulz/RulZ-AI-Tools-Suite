import { describe, expect, it } from 'vitest'
import type { ModelEntry } from '@shared/types'
import { createModelCatalog } from '../src/main/models/catalog'

const entry = (id: string, extra: Partial<ModelEntry> = {}): ModelEntry => ({
  id,
  name: id,
  folder: '',
  path: id,
  files: [],
  sizeBytes: 1,
  info: null,
  modifiedAt: 0,
  ...extra
})

describe('createModelCatalog', () => {
  const local = [entry('A/a.gguf'), entry('B/b.gguf')]
  const library = { list: () => local, get: (id: string) => local.find((m) => m.id === id) }

  it('lists the library first and then the rentals, all of those marked remote', () => {
    const catalog = createModelCatalog(library, () => [entry('vast/big'), entry('vast/small', { remote: true })])
    expect(catalog.list().map((m) => [m.id, !!m.remote])).toEqual([
      ['A/a.gguf', false],
      ['B/b.gguf', false],
      ['vast/big', true],
      ['vast/small', true]
    ])
    expect(catalog.get('vast/big')?.remote).toBe(true)
    expect(catalog.get('B/b.gguf')).toBe(local[1])
    expect(catalog.get('nope')).toBeUndefined()
  })

  it('follows the remote list as it changes, and lets the library win on a clash', () => {
    let remote: ModelEntry[] = []
    const catalog = createModelCatalog(library, () => remote)
    expect(catalog.list().map((m) => m.id)).toEqual(['A/a.gguf', 'B/b.gguf'])
    remote = [entry('A/a.gguf', { name: 'impostor' }), entry('vast/x')]
    expect(catalog.list().map((m) => m.id)).toEqual(['A/a.gguf', 'B/b.gguf', 'vast/x'])
    expect(catalog.get('A/a.gguf')?.name).toBe('A/a.gguf')
    expect(catalog.get('A/a.gguf')?.remote).toBeUndefined()
  })
})
