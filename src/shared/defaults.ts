import type { AppSettings, LoadConfig, SamplingParams } from './types'

export const DEFAULT_LOAD_CONFIG: LoadConfig = {
  contextLength: 8192,
  gpuLayers: 'auto',
  splitMode: 'auto',
  mainGpu: 0,
  tensorSplit: '',
  flashAttention: 'auto',
  cacheTypeK: 'f16',
  cacheTypeV: 'f16',
  threads: null,
  batchSize: null,
  ubatchSize: null,
  useMmap: true,
  useMlock: false,
  useMmproj: true,
  seed: null,
  chatTemplateFile: '',
  parallel: 1,
  extraArgs: ''
}

export const DEFAULT_SAMPLING: SamplingParams = {
  temperature: 0.8,
  topK: 40,
  topP: 0.95,
  minP: 0.05,
  repeatPenalty: 1.1,
  presencePenalty: 0,
  frequencyPenalty: 0,
  maxTokens: null,
  stop: [],
  seed: null,
  jsonSchema: ''
}

export const DEFAULT_SETTINGS: AppSettings = {
  modelsDir: 'C:\\LLM',
  enginePath: '',
  theme: 'dark',
  unloadPreviousOnLoad: true,
  defaultLoadConfig: DEFAULT_LOAD_CONFIG,
  defaultSampling: DEFAULT_SAMPLING,
  defaultSystemPrompt: '',
  contextOverflow: 'truncateMiddle',
  expandThinking: false,
  rightPanelOpen: true
}
