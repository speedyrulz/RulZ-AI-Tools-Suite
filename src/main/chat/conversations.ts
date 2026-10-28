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

/** One JSON file per conversation, plus a list of folder names. */
export class ConversationStore {
  private summaries: Map<string, ConversationSummary> | null = null
  private readonly foldersPath: string

  constructor(private readonly dir: string) {
    this.foldersPath = join(dir, '..', 'folders.json')
  }

  private file(id: string): string {
    if (!ID_RE.test(id)) throw new Error(`Invalid conversation id: ${id}`)
    return join(this.dir, `${id}.json`)
  }

  async list(): Promise<ConversationSummary[]> {
    if (!this.summaries) {
      this.summaries = new Map()
      await fs.mkdir(this.dir, { recursive: true })
      for (const name of await fs.readdir(this.dir)) {
        if (!name.endsWith('.json')) continue
        const c = await readJson<Conversation | null>(join(this.dir, name), null)
        if (c?.id && Array.isArray(c.messages)) this.summaries.set(c.id, summarize(c))
      }
    }
    return [...this.summaries.values()].sort((a, b) => b.updatedAt - a.updatedAt)
  }

  async get(id: string): Promise<Conversation | null> {
    const c = await readJson<Conversation | null>(this.file(id), null)
    // Chats saved by older versions lack newer sampling fields.
    return c ? { ...c, sampling: parseSampling(c.sampling) } : null
  }

  async save(c: Conversation): Promise<ConversationSummary> {
    await writeJsonAtomic(this.file(c.id), c)
    const s = summarize(c)
    ;(await this.ensureLoaded()).set(c.id, s)
    return s
  }

  async delete(id: string): Promise<void> {
    const path = this.file(id)
    // Move to the Recycle Bin so a mistaken delete can be undone from Windows.
    await shell.trashItem(path).catch(() => fs.rm(path, { force: true }))
    ;(await this.ensureLoaded()).delete(id)
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
