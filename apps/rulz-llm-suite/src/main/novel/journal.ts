// Unsaved-edit journal: .rulz/unsaved/<path with / as __>.json (data safety rule 4).
import { promises as fs } from 'node:fs'
import { join } from 'node:path'
import { assertAppPath, writeBytesAtomic } from './files'
import { moveToTrash } from './trash'

export interface Journal {
  text: string
  baseHash: string
  savedAt: string
}

const dirOf = (folder: string): string => join(folder, '.rulz', 'unsaved')
const nameOf = (rel: string): string => `${rel.replace(/\//g, '__')}.json`
export const journalRel = (rel: string): string => `.rulz/unsaved/${nameOf(rel)}`

export async function writeJournal(folder: string, rel: string, text: string, baseHash: string, now = new Date()): Promise<void> {
  await assertAppPath(folder, dirOf(folder))
  const j: Journal = { text, baseHash, savedAt: now.toISOString() }
  await writeBytesAtomic(folder, join(dirOf(folder), nameOf(rel)), join(folder, '.rulz', 'tmp'), Buffer.from(JSON.stringify(j), 'utf8'))
}

/** The journal of a path, or null when there is none or it cannot be read. */
export async function readJournal(folder: string, rel: string): Promise<Journal | null> {
  await assertAppPath(folder, join(dirOf(folder), nameOf(rel)))
  try {
    const j = JSON.parse(await fs.readFile(join(dirOf(folder), nameOf(rel)), 'utf8')) as Partial<Journal>
    if (typeof j.text !== 'string' || typeof j.savedAt !== 'string') return null
    return { text: j.text, baseHash: typeof j.baseHash === 'string' ? j.baseHash : '', savedAt: j.savedAt }
  } catch {
    return null
  }
}

export async function deleteJournal(folder: string, rel: string): Promise<void> {
  await assertAppPath(folder, join(dirOf(folder), nameOf(rel)))
  await fs.rm(join(dirOf(folder), nameOf(rel)), { force: true }).catch(() => undefined)
}

/** Discard = move to the project's trash, so even a discarded edit can be found again. */
export async function discardJournal(folder: string, rel: string): Promise<void> {
  const j = journalRel(rel)
  await moveToTrash(folder, [j], `Unsaved edits of ${rel}`)
}
