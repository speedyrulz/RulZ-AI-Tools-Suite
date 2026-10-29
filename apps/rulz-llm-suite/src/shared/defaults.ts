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
  useDraft: true,
  draftFile: null,
  // 2 beat llama.cpp's default of 3 in every matched run on a 26B-A4B MoE with its MTP head.
  draftTokens: 2,
  draftMinTokens: null,
  draftPMin: null,
  draftSampling: 'auto',
  draftGpu: 'auto',
  draftCacheTypeK: 'f16',
  draftCacheTypeV: 'f16',
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
  jsonSchema: '',
  thinking: 'auto',
  reasoningEffort: 'default',
  thinkingBudget: null,
  thinkingBudgetMessage: ''
}

export const DEFAULT_SETTINGS: AppSettings = {
  modelsDir: 'C:\\LLM',
  enginePath: '',
  theme: 'dark',
  themeChosen: false,
  unloadPreviousOnLoad: true,
  defaultLoadConfig: DEFAULT_LOAD_CONFIG,
  defaultSampling: DEFAULT_SAMPLING,
  defaultSystemPrompt: '',
  contextOverflow: 'truncateMiddle',
  expandThinking: false,
  keepThinkingOpen: false,
  rightPanelOpen: true,
  minimizeToTray: false,
  closeToTray: false,
  toolsEnabled: true,
  confirmToolCalls: true,
  allowedTools: [],
  disabledTools: [],
  toolMode: 'auto',
  toolBudgetTokens: 6000,
  compactToolResults: true,
  keepFullToolResults: 3,
  apiServer: {
    autoStart: false,
    port: 1234,
    bind: 'localhost',
    apiKey: '',
    cors: true,
    idleUnloadMinutes: 10,
    defaultModelId: ''
  }
}
