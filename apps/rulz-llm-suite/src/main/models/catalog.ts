// Every model the app can use: the ones in the library on this PC, and the ones defined for vast.ai rentals.
import type { ModelEntry } from '@shared/types'

export interface ModelCatalog {
  /** A model by id: the library's, else a rental's. */
  get(id: string): ModelEntry | undefined
  /** The library's models, then the rentals' (all marked `remote`). */
  list(): ModelEntry[]
}

/**
 * Library ids are relative paths and remote ids start with `vast/`, so they cannot clash; if one ever did, the
 * library's model would win. A remote entry is not a file on this PC: it cannot be trashed, and a benchmark (which
 * loads models here) refuses it, so those callers check `entry.remote`.
 */
export function createModelCatalog(library: { get(id: string): ModelEntry | undefined; list(): ModelEntry[] }, remote: () => ModelEntry[]): ModelCatalog {
  const marked = (m: ModelEntry): ModelEntry => (m.remote === true ? m : { ...m, remote: true })
  return {
    get: (id) => library.get(id) ?? remote().map(marked).find((m) => m.id === id),
    list: () => {
      const local = library.list()
      const taken = new Set(local.map((m) => m.id))
      return [...local, ...remote().filter((m) => !taken.has(m.id)).map(marked)]
    }
  }
}
