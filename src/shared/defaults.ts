import type { AppSettings } from './types'

export const DEFAULT_SETTINGS: AppSettings = {
  serverUrl: '',
  lastServerUrl: '',
  apiKey: '',
  model: '',
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
