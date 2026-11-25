// The OpenRouter key: its own encrypted file (cloud-secrets.json). No environment variable and no other file is read: a key
// found elsewhere must not start spending (docs/cloud-design.md, section 11).
import { promises as fs } from 'node:fs'
import { writeJsonAtomic } from '../util/jsonFile'

export interface KeyStoreDeps {
  path: string
  safeStorage: { isEncryptionAvailable(): boolean; encryptString(text: string): Buffer; decryptString(data: Buffer): string }
  write?: (path: string, data: unknown) => Promise<void>
}

const MIN_SCRUB_LENGTH = 6
/** OpenRouter keys look like this: removed from any text even when the store never saw them. */
const KEY_SHAPE = /sk-or-[A-Za-z0-9_-]{6,}/g

export class CloudKeyStore {
  private key: string | null = null
  /** Every key value seen (saved, or only tried), so scrub() can remove it. */
  private readonly known = new Set<string>()

  constructor(private readonly deps: KeyStoreDeps) {}

  /** Reads the saved key. A missing, unreadable or undecryptable file means no key. */
  async load(): Promise<void> {
    this.key = null
    let raw: Record<string, unknown>
    try {
      raw = JSON.parse((await fs.readFile(this.deps.path, 'utf8')).replace(/^﻿/, '')) as Record<string, unknown>
    } catch {
      return
    }
    const b64 = raw?.key
    if (typeof b64 !== 'string' || !b64) return
    try {
      const plain = this.deps.safeStorage.decryptString(Buffer.from(b64, 'base64')).trim()
      if (plain) {
        this.key = plain
        this.remember(plain)
      }
    } catch {
      // Encrypted under another login or machine: same as not saved.
    }
  }

  get(): string | null {
    return this.key
  }

  /** Notes a key (also one that is only being tried) for scrub(). */
  remember(key: string): void {
    const k = key.trim()
    if (k.length >= MIN_SCRUB_LENGTH) this.known.add(k)
  }

  /** Saves a key encrypted. Without encryption on this PC it is refused: it is never written as text. */
  async save(key: string): Promise<void> {
    const k = key.trim()
    if (!k) return this.clear()
    this.remember(k)
    if (!this.deps.safeStorage.isEncryptionAvailable()) {
      throw new Error('This PC cannot encrypt the key for saving, so the app will not keep it.')
    }
    await (this.deps.write ?? writeJsonAtomic)(this.deps.path, { key: this.deps.safeStorage.encryptString(k).toString('base64') })
    this.key = k
  }

  async clear(): Promise<void> {
    await (this.deps.write ?? writeJsonAtomic)(this.deps.path, {})
    this.key = null
  }

  scrub(text: string): string {
    let out = text
    // Longest first, so a key that contains another is removed whole.
    for (const s of [...this.known].sort((a, b) => b.length - a.length)) out = out.split(s).join('***')
    return out.replace(KEY_SHAPE, '***').replace(/(Bearer\s+)[^\s"']+/gi, '$1***')
  }
}
