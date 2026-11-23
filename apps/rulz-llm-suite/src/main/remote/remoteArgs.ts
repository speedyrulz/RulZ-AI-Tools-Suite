// llama-server arguments for a rental (docs/vast-design.md, section 7): paths on the instance, nothing that only means
// something on this PC. Built from the same pieces as a local load (buildServerArgs), then fixed up.
import type { LoadConfig, ModelEntry, SidecarFile } from '@shared/types'
import { buildServerArgs } from '../engine/args'
import { splitArgs } from '../engine/process'

export interface RemotePaths {
  /** First part of the model. */
  main: string
  mmproj?: string
  /** A separate draft file. */
  draft?: string
  /** In llama-server's adapter id order. */
  adapters: string[]
}

export interface RemoteArgsInput {
  /** The model's entry; its paths are ignored (`paths` has the ones on the instance). */
  model: ModelEntry
  config: LoadConfig
  /** A number, or the placeholder the start script replaces. */
  port: number | string
  alias: string
  paths: RemotePaths
  /** The draft this load uses (as InstanceManager's draftFor chooses it). */
  draft?: SidecarFile
}

/** Settings of the local server that name this PC; refused in the extra arguments. */
const REFUSED_FLAGS = new Map<string, string>([
  ['--host', 'The server always listens on the instance itself, so --host cannot be changed.'],
  ['--port', 'The port is chosen on the instance, so --port cannot be set.'],
  ['--model', 'The model file is set by the app, so --model cannot be used.'],
  ['-m', 'The model file is set by the app, so -m cannot be used.'],
  ['--alias', 'The alias is set by the app, so --alias cannot be used.'],
  ['--path', '--path points at a folder on this PC and has no meaning on the instance.']
])

const WINDOWS_PATH = /^(?:[A-Za-z]:[\\/]|\\\\|~[\\/])/

/** The sentence for the first extra argument that cannot be used on a rental; undefined when all can. */
export function extraArgsProblem(text: string): string | undefined {
  for (const tok of splitArgs(text)) {
    const [flag, ...rest] = tok.split('=')
    const why = REFUSED_FLAGS.get(flag)
    if (why) return why
    const value = rest.join('=')
    if (WINDOWS_PATH.test(tok) || (rest.length && WINDOWS_PATH.test(value))) {
      return `The extra argument ${tok.length > 60 ? `${tok.slice(0, 57)}...` : tok} is a path on this PC, which the instance cannot read.`
    }
  }
  return undefined
}

/**
 * llama-server arguments for a rental. Left out: threads (the instance has its own cores), the memory lock, the main
 * GPU index of a single-GPU setting (a local number), and the model's file paths on this PC. A custom tensor split is
 * kept. Throws a sentence for settings that cannot work there (a local chat template file, local paths in the extra
 * arguments, a projector or draft that is not on the instance).
 */
export function buildRemoteArgs({ model, config, port, alias, paths, draft }: RemoteArgsInput): string[] {
  if (config.chatTemplateFile.trim()) {
    throw new Error('A chat template file on this PC cannot be used on a rental. Clear it in the load settings (the model\'s own template is used) or put the template text in the model.')
  }
  const problem = extraArgsProblem(config.extraArgs)
  if (problem) throw new Error(problem)

  const cfg: LoadConfig = { ...config, threads: null, useMlock: false }
  let mmproj: SidecarFile | undefined
  if (cfg.useMmproj && model.mmproj) {
    if (!paths.mmproj) throw new Error('The vision projector is not on the instance. Set the model up there again.')
    mmproj = { ...model.mmproj, path: paths.mmproj }
  }
  let useDraft: SidecarFile | undefined
  if (cfg.useDraft && draft?.kind) {
    if (draft.builtIn) useDraft = { ...draft, path: paths.main }
    else if (paths.draft) useDraft = { ...draft, path: paths.draft }
    else throw new Error('The draft model is not on the instance. Set the model up there again.')
  }
  const remoteModel: ModelEntry = { ...model, path: paths.main, files: [paths.main], mmproj, draft: useDraft, draftSidecar: undefined }
  const args = buildServerArgs({ model: remoteModel, config: cfg, port: 0, alias, adapters: paths.adapters.map((path) => ({ path })) })

  const at = args.indexOf('--port')
  if (at >= 0) args[at + 1] = String(port)
  const gpu = args.indexOf('--main-gpu')
  if (gpu >= 0) args.splice(gpu, 2)
  return args
}
