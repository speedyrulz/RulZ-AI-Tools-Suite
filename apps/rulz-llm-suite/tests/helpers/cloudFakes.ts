// Fakes for the OpenRouter side shared by the main-process tests (instances, chat, API server, writing).
import type { CloudModelDef, CloudModelInfo } from '@shared/cloud'
import type { LoadedInstance, ModelEntry } from '@shared/types'

export const cloudInfo = (slug: string, over: Partial<CloudModelInfo> = {}): CloudModelInfo => ({
  slug,
  name: slug,
  description: '',
  contextLength: 200_000,
  maxReply: 8192,
  prices: { promptMax: 1, completionMax: 2, request: 0, image: 0, prompt: 1, completion: 2 },
  free: false,
  vision: true,
  tools: true,
  reasoning: null,
  supported: ['temperature', 'tools', 'max_tokens'],
  created: 1_700_000_000,
  expires: null,
  ...over
})

export const cloudDef = (slug: string, over: Partial<CloudModelDef> = {}): CloudModelDef => ({
  slug,
  contextLimit: 32_768,
  replyLimit: 8192,
  allowApps: false,
  info: cloudInfo(slug),
  addedAt: 0,
  ...over
})

/** A catalog entry of an OpenRouter model, the way the cloud service lists it: `cloud` set, `remote` not. */
export const cloudEntry = (slug: string, over: Partial<CloudModelDef> = {}): ModelEntry => ({
  id: `openrouter/${slug}`,
  name: slug,
  folder: '',
  path: '',
  files: [],
  sizeBytes: 0,
  info: null,
  modifiedAt: 0,
  cloud: { ...cloudDef(slug, over), provider: 'openrouter' }
})

export const localEntry = (id: string, name = id): ModelEntry => ({
  id,
  name,
  folder: '',
  path: `C:\\LLM\\${id}`,
  files: [],
  sizeBytes: 1,
  info: null,
  modifiedAt: 0
})

export const sse = (lines: object[], done = true): string => lines.map((l) => `data: ${JSON.stringify(l)}\n\n`).join('') + (done ? 'data: [DONE]\n\n' : '')

export const cloudInstance = (slug: string, over: Partial<LoadedInstance> = {}): LoadedInstance =>
  ({
    instanceId: `c-${slug}`,
    modelId: `openrouter/${slug}`,
    modelName: slug,
    port: 0,
    pid: null,
    status: 'ready',
    progress: 1,
    stage: 'Ready',
    config: {} as LoadedInstance['config'],
    startedAt: 0,
    contextLength: 32_768,
    hasVision: false,
    adapters: [],
    estimate: null,
    cloud: { provider: 'openrouter', free: false },
    ...over
  }) as LoadedInstance
