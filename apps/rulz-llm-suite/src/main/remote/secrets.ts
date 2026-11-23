// The vast.ai key and the Hugging Face token (docs/vast-design.md, section 3). A key typed in the app is kept encrypted
// (Electron safeStorage in production); without encryption it is refused, never written as text.
import type { Secrets } from './contract'

export interface SecretsDeps {
  /** Electron's safeStorage in production. */
  available: () => boolean
  encrypt: (plain: string) => Buffer
  decrypt: (data: Buffer) => string
  /** The text of a file, or null when it does not exist or cannot be read. */
  readFile: (path: string) => Promise<string | null>
  /** Writes JSON atomically (writeJsonAtomic in production). */
  writeFile: (path: string, data: unknown) => Promise<void>
  env: Record<string, string | undefined>
  homedir: string
  join: (...parts: string[]) => string
  /** remote-secrets.json */
  filePath: string
}

type Kind = 'vast' | 'hf'
const MIN_SCRUB_LENGTH = 8

export class SecretStore implements Secrets {
  /** Every secret value seen, so scrub() can remove it from log lines. */
  private readonly known = new Set<string>()

  constructor(private readonly deps: SecretsDeps) {}

  private async readSaved(): Promise<Partial<Record<Kind, string>>> {
    const text = await this.deps.readFile(this.deps.filePath)
    if (!text) return {}
    try {
      const raw = JSON.parse(text.replace(/^﻿/, '')) as Record<string, unknown>
      const out: Partial<Record<Kind, string>> = {}
      for (const k of ['vast', 'hf'] as const) {
        const b64 = raw?.[k]
        if (typeof b64 !== 'string' || !b64) continue
        try {
          const plain = this.deps.decrypt(Buffer.from(b64, 'base64')).trim()
          if (plain) out[k] = plain
        } catch {
          // Encrypted under another login or machine: same as not saved.
        }
      }
      return out
    } catch {
      return {}
    }
  }

  private async readTrimmed(path: string): Promise<string | null> {
    const t = (await this.deps.readFile(path))?.trim()
    return t ? t : null
  }

  private remember(value: string | undefined | null): void {
    if (value && value.length >= MIN_SCRUB_LENGTH) this.known.add(value)
  }

  /** Candidates in priority order; every value found is remembered for scrub(). */
  private async lookup(kind: Kind): Promise<{ value: string; source: string } | null> {
    const { env, homedir, join } = this.deps
    const found: { value: string; source: string }[] = []
    const add = (value: string | null | undefined, source: string): void => {
      const v = value?.trim()
      if (!v) return
      this.remember(v)
      found.push({ value: v, source })
    }
    add((await this.readSaved())[kind], 'saved in this app')
    if (kind === 'vast') {
      add(env.VAST_API_KEY, 'VAST_API_KEY environment variable')
      const files: string[] = []
      if (env.XDG_CONFIG_HOME) files.push(join(env.XDG_CONFIG_HOME, 'vastai', 'vast_api_key'))
      files.push(join(homedir, '.config', 'vastai', 'vast_api_key'), join(homedir, '.vast_api_key'))
      for (const f of files) add(await this.readTrimmed(f), `vastai CLI (${f})`)
    } else {
      add(env.HF_TOKEN, 'HF_TOKEN environment variable')
      const files: string[] = []
      if (env.HF_HOME) files.push(join(env.HF_HOME, 'token'))
      files.push(join(homedir, '.cache', 'huggingface', 'token'))
      for (const f of files) add(await this.readTrimmed(f), `Hugging Face CLI (${f})`)
    }
    return found[0] ?? null
  }

  vastKey(): Promise<{ value: string; source: string } | null> {
    return this.lookup('vast')
  }
  hfToken(): Promise<{ value: string; source: string } | null> {
    return this.lookup('hf')
  }

  private async save(kind: Kind, value: string): Promise<void> {
    const v = value.trim()
    const file: Record<string, string> = {}
    // Keep the other secret's ciphertext as it is on disk.
    const text = await this.deps.readFile(this.deps.filePath)
    if (text) {
      try {
        const raw = JSON.parse(text) as Record<string, unknown>
        for (const k of ['vast', 'hf']) if (typeof raw[k] === 'string' && raw[k]) file[k] = raw[k] as string
      } catch {
        // An unreadable file is replaced.
      }
    }
    if (v) {
      if (!this.deps.available()) {
        throw new Error('This PC cannot encrypt the key for saving. Set it as an environment variable instead (VAST_API_KEY or HF_TOKEN).')
      }
      this.remember(v)
      file[kind] = this.deps.encrypt(v).toString('base64')
    } else {
      delete file[kind]
    }
    await this.deps.writeFile(this.deps.filePath, file)
  }

  setVastKey(value: string): Promise<void> {
    return this.save('vast', value)
  }
  setHfToken(value: string): Promise<void> {
    return this.save('hf', value)
  }

  scrub(text: string): string {
    let out = text
    // Longest first, so a secret that contains another is removed whole.
    for (const s of [...this.known].sort((a, b) => b.length - a.length)) out = out.split(s).join('***')
    return out
  }
}
