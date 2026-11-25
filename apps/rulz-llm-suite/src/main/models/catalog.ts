// Every model the app can use: the ones in the library on this PC, the ones defined for vast.ai rentals, and OpenRouter's.
import type { ModelEntry } from '@shared/types'

export interface ModelCatalog {
  /** A model by id: the library's, else a rental's, else an OpenRouter one. */
  get(id: string): ModelEntry | undefined
  /** The library's models, then the rentals' (all marked `remote`), then OpenRouter's (they carry `cloud`, never `remote`). */
  list(): ModelEntry[]
}

/**
 * Library ids are relative paths, remote ids start with `vast/` and cloud ids with `openrouter/`, so they cannot clash;
 * if one ever did (a library folder called openrouter), the library's model would win. A remote or cloud entry is not
 * a file on this PC: it cannot be trashed, and a benchmark (which loads models here) refuses it, so those callers check
 * `entry.remote` and `entry.cloud`.
 */
export function createModelCatalog(
  library: { get(id: string): ModelEntry | undefined; list(): ModelEntry[] },
  remote: () => ModelEntry[],
  cloud: () => ModelEntry[] = () => []
): ModelCatalog {
  const marked = (m: ModelEntry): ModelEntry => (m.remote === true ? m : { ...m, remote: true })
  // A cloud entry is never a rental's, whatever the source says.
  const cloudOnly = (m: ModelEntry): ModelEntry => {
    if (m.remote === undefined) return m
    const { remote: _remote, ...rest } = m
    return rest
  }
  return {
    get: (id) => library.get(id) ?? remote().map(marked).find((m) => m.id === id) ?? cloud().map(cloudOnly).find((m) => m.id === id),
    list: () => {
      const local = library.list()
      const taken = new Set(local.map((m) => m.id))
      const rentals = remote().filter((m) => !taken.has(m.id)).map(marked)
      for (const m of rentals) taken.add(m.id)
      return [...local, ...rentals, ...cloud().filter((m) => !taken.has(m.id)).map(cloudOnly)]
    }
  }
}
