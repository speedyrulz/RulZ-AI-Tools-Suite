import type { LoadConfig, ModelEntry } from '@shared/types'
import { splitArgs } from './process'

export interface ServerArgsInput {
  model: ModelEntry
  config: LoadConfig
  port: number
  alias: string
}

/** Turns a LoadConfig into llama-server arguments. Unset values are left to llama.cpp (incl. --fit). */
export function buildServerArgs({ model, config: c, port, alias }: ServerArgsInput): string[] {
  const args = [
    '--model', model.path,
    '--host', '127.0.0.1',
    '--port', String(port),
    '--alias', alias,
    '--no-webui',
    '--jinja',
    '--reasoning-format', 'deepseek',
    '--parallel', String(c.parallel)
  ] // prettier-ignore

  if (c.contextLength !== null) args.push('--ctx-size', String(c.contextLength))
  if (c.gpuLayers !== 'auto') args.push('--n-gpu-layers', String(c.gpuLayers))

  if (c.splitMode === 'single') args.push('--split-mode', 'none', '--main-gpu', String(c.mainGpu))
  else if (c.splitMode === 'custom' && c.tensorSplit.trim()) {
    args.push('--split-mode', 'layer', '--tensor-split', c.tensorSplit.replace(/\s+/g, ''))
  }

  if (c.flashAttention !== 'auto') args.push('--flash-attn', c.flashAttention)
  if (c.cacheTypeK !== 'f16') args.push('--cache-type-k', c.cacheTypeK)
  if (c.cacheTypeV !== 'f16') args.push('--cache-type-v', c.cacheTypeV)
  if (c.threads !== null) args.push('--threads', String(c.threads))
  if (c.batchSize !== null) args.push('--batch-size', String(c.batchSize))
  if (c.ubatchSize !== null) args.push('--ubatch-size', String(c.ubatchSize))
  if (!c.useMmap) args.push('--no-mmap')
  if (c.useMlock) args.push('--mlock')
  if (c.useMmproj && model.mmproj) args.push('--mmproj', model.mmproj.path)
  if (c.seed !== null) args.push('--seed', String(c.seed))
  if (c.chatTemplateFile.trim()) args.push('--chat-template-file', c.chatTemplateFile.trim())
  if (c.extraArgs.trim()) args.push(...splitArgs(c.extraArgs))
  return args
}
