import { promises as fs } from 'node:fs'
import { join } from 'node:path'
import { shell } from 'electron'
import type { Conversation, ConversationSummary } from '@shared/types'
import { parseSampling } from '../settings/schema'
import { readJson, writeJsonAtomic } from '../util/jsonFile'

const ID_RE = /^[\w-]{1,64}$/

function summarize(c: Conversation): ConversationSummary {
  return {
    id: c.id,
    title: c.title,
    folder: c.folder,
    pinned: c.pinned,
    updatedAt: c.updatedAt,
    messageCount: c.messages.length,
    modelId: c.modelId
  }
}

interface IndexEntry {
  summary: ConversationSummary
  size: number
  mtimeMs: number
}

interface IndexFile {
  v: number
  entries: Record<string, IndexEntry>
}

const INDEX_VERSION = 1

/**
 * One JSON file per conversation, plus a list of folder names. Summaries are kept in an index
 * checked against each file's size and mtime, so listing chats doesn't read every chat (images
 * included) in full.
 */
export class ConversationStore {
  private summaries: Map<string, ConversationSummary> | null = null
  private index: Record<string, IndexEntry> = {}
  private indexTimer: ReturnType<typeof setTimeout> | null = null
  private readonly foldersPath: string
  private readonly indexPath: string

  constructor(private readonly dir: string) {
    this.foldersPath = join(dir, '..', 'folders.json')
    this.indexPath = join(dir, '..', 'conversations-index.json')
  }

  private file(id: string): string {
    if (!ID_RE.test(id)) throw new Error(`Invalid conversation id: ${id}`)
    return join(this.dir, `${id}.json`)
  }

  async list(): Promise<ConversationSummary[]> {
    if (!this.summaries) {
      this.summaries = new Map()
      await fs.mkdir(this.dir, { recursive: true })
      const saved = await readJson<IndexFile | null>(this.indexPath, null)
      const known = saved?.v === INDEX_VERSION && saved.entries ? saved.entries : {}
      const next: Record<string, IndexEntry> = {}
      let changed = false
      for (const name of await fs.readdir(this.dir)) {
        if (!name.endsWith('.json')) continue
        const path = join(this.dir, name)
        const st = await fs.stat(path).catch(() => null)
        if (!st) continue
        const hit = known[name]
        if (hit && hit.size === st.size && hit.mtimeMs === st.mtimeMs) {
          next[name] = hit
          this.summaries.set(hit.summary.id, hit.summary)
          continue
        }
        changed = true
        const c = await readJson<Conversation | null>(path, null)
        if (!c?.id || !Array.isArray(c.messages)) continue
        const summary = summarize(c)
        next[name] = { summary, size: st.size, mtimeMs: st.mtimeMs }
        this.summaries.set(c.id, summary)
      }
      if (Object.keys(known).some((name) => !next[name])) changed = true
      this.index = next
      if (changed) this.saveIndexSoon()
    }
    return [...this.summaries.values()].sort((a, b) => b.updatedAt - a.updatedAt)
  }

  /** The index is only a cache (checked against size and mtime), so it is written lazily. */
  private saveIndexSoon(): void {
    if (this.indexTimer) clearTimeout(this.indexTimer)
    this.indexTimer = setTimeout(() => {
      this.indexTimer = null
      const file: IndexFile = { v: INDEX_VERSION, entries: this.index }
      void writeJsonAtomic(this.indexPath, file).catch(() => undefined)
    }, 1000)
  }

  async get(id: string): Promise<Conversation | null> {
    const c = await readJson<Conversation | null>(this.file(id), null)
    // Chats saved by older versions lack newer sampling fields.
    return c ? { ...c, sampling: parseSampling(c.sampling) } : null
  }

  async save(c: Conversation): Promise<ConversationSummary> {
    const path = this.file(c.id)
    await writeJsonAtomic(path, c)
    const s = summarize(c)
    ;(await this.ensureLoaded()).set(c.id, s)
    const st = await fs.stat(path).catch(() => null)
    if (st) {
      this.index[`${c.id}.json`] = { summary: s, size: st.size, mtimeMs: st.mtimeMs }
      this.saveIndexSoon()
    }
    return s
  }

  async delete(id: string): Promise<void> {
    const path = this.file(id)
    // Move to the Recycle Bin so a mistaken delete can be undone from Windows.
    await shell.trashItem(path).catch(() => fs.rm(path, { force: true }))
    ;(await this.ensureLoaded()).delete(id)
    delete this.index[`${id}.json`]
    this.saveIndexSoon()
  }

  async listFolders(): Promise<string[]> {
    return readJson<string[]>(this.foldersPath, [])
  }

  async saveFolders(folders: string[]): Promise<void> {
    await writeJsonAtomic(this.foldersPath, [...new Set(folders.filter((f) => f.trim()))])
  }

  private async ensureLoaded(): Promise<Map<string, ConversationSummary>> {
    if (!this.summaries) await this.list()
    return this.summaries!
  }
}
