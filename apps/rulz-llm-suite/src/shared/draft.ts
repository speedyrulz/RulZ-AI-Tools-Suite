import type { DraftKind, LoadConfig, ModelEntry, SidecarFile } from './types'

/** Draft heads are named after their format, e.g. "mtp-<model>.gguf" or "<model>-eagle3.gguf". */
export function draftKind(fileName: string): DraftKind | undefined {
  const base = fileName.replace(/\.gguf$/i, '').toLowerCase()
  for (const kind of ['mtp', 'eagle3', 'dflash', 'dspark'] as const) {
    if (new RegExp(`(^|[-_. ])${kind}([-_. ]|$)`).test(base)) return kind
  }
  return undefined
}

/** Every separate draft file in the library, once each, by file name. */
export function draftFiles(models: ModelEntry[]): SidecarFile[] {
  const byPath = new Map<string, SidecarFile>()
  for (const m of models) {
    if (m.draftSidecar?.kind) byPath.set(m.draftSidecar.path, m.draftSidecar)
    if (m.draft?.kind && !m.draft.builtIn) byPath.set(m.draft.path, m.draft)
  }
  return [...byPath.values()].sort((a, b) => fileName(a.path).localeCompare(fileName(b.path)))
}

/** The draft a load will use: the chosen one, or the one found with the model; undefined = none. */
export function resolveDraft(model: ModelEntry, cfg: LoadConfig, models: ModelEntry[]): SidecarFile | undefined {
  const choice = cfg.draftFile
  if (choice === null) return model.draft?.kind ? model.draft : undefined
  if (choice === 'builtin') return model.info?.mtpLayers ? { path: model.path, sizeBytes: 0, kind: 'mtp', builtIn: true } : undefined
  const known = draftFiles(models).find((d) => d.path === choice)
  if (known) return known
  // A file that isn't in the library (moved, or outside the models folder): trust its name.
  return { path: choice, sizeBytes: 0, kind: draftKind(fileName(choice)) ?? 'mtp' }
}

export function fileName(path: string): string {
  return path.split(/[\\/]/).pop() ?? path
}
