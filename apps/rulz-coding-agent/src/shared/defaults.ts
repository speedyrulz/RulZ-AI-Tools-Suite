import type { AppSettings, KvCacheType, ReasoningEffort, SamplingSettings } from './types'

export const DEFAULT_SETTINGS: AppSettings = {
  serverUrl: '',
  lastServerUrl: '',
  apiKey: '',
  model: '',
  modelSampling: {},
  approval: 'balanced',
  maxSteps: 40,
  commandTimeoutMin: 10,
  theme: 'dark',
  projectsDir: 'C:\\coding',
  instructions: '',
  firstRunDone: false,
  remoteEnabled: false,
  remotePort: 7420,
  remoteToken: ''
}

export const isQwenCoder = (model: string): boolean => /qwen3[-_.]?coder/i.test(model)

/** Qwen's published settings for Qwen3-Coder; other models use RulZ LLM Suite's own defaults. */
export function recommendedSampling(model: string): SamplingSettings {
  return isQwenCoder(model) ? { temperature: 0.7, topP: 0.8, topK: 20, repeatPenalty: 1.05 } : {}
}

/** Upper limit for one answer when the owner hasn't set one. */
export const DEFAULT_MAX_OUTPUT_TOKENS = 16384

/** The owner's settings over the recommended ones, as request fields for /v1/chat/completions. */
export function samplingRequest(model: string, own: SamplingSettings | undefined): Record<string, unknown> {
  const s = { ...recommendedSampling(model), ...own }
  const out: Record<string, unknown> = {}
  if (s.temperature !== undefined) out.temperature = s.temperature
  if (s.topP !== undefined) out.top_p = s.topP
  if (s.topK !== undefined) out.top_k = s.topK
  if (s.repeatPenalty !== undefined) out.repeat_penalty = s.repeatPenalty
  // The same fields RulZ LLM Suite's chat sends; llama-server reads them.
  if (s.thinking) out.chat_template_kwargs = { enable_thinking: s.thinking === 'on' }
  if (s.thinking !== 'off') {
    if (s.reasoningEffort) out.reasoning_effort = s.reasoningEffort
    if (s.thinkingBudget !== undefined) {
      out.thinking_budget_tokens = s.thinkingBudget
      if (s.thinkingBudgetMessage?.trim()) out.reasoning_budget_message = s.thinkingBudgetMessage
    }
  }
  return out
}

export const REASONING_EFFORTS: ReasoningEffort[] = ['minimal', 'low', 'medium', 'high', 'xhigh', 'max']

export const KV_CACHE_TYPES: KvCacheType[] = ['f16', 'bf16', 'f32', 'q8_0', 'q5_1', 'q5_0', 'q4_1', 'q4_0', 'iq4_nl']

/** Bumped when the messages between the computer and the phone app change in a way the other side must know. */
export const REMOTE_PROTOCOL = 1

/** Owner-written instructions go with every request, so they are kept short (notes and Settings alike). */
export const MAX_INSTRUCTION_CHARS = 4000

/** Addresses tried when finding RulZ LLM Suite (LM Studio often holds 1234, so RulZ may be on 1235). */
export const CANDIDATE_URLS = ['http://127.0.0.1:1234', 'http://127.0.0.1:1235']

export const APPROVAL_LEVELS: { value: AppSettings['approval']; label: string; hint: string }[] = [
  { value: 'careful', label: 'Careful', hint: 'Asks before every file change and every command.' },
  { value: 'balanced', label: 'Balanced', hint: 'Changes files freely (Undo covers them); asks before commands and deleting.' },
  { value: 'handsoff', label: 'Hands-off', hint: 'Only asks before deleting and before risky commands.' }
]
