import { existsSync, mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterAll, describe, expect, it, vi } from 'vitest'
import type { Conversation } from '@shared/types'

// No Recycle Bin in tests: the store falls back to deleting the file.
vi.mock('electron', () => ({ shell: { trashItem: () => Promise.reject(new Error('no shell')) } }))
const { ConversationStore, DeletedConversationError } = await import('../src/main/chat/conversations')

const root = mkdtempSync(join(tmpdir(), 'rulz-convs-'))
afterAll(() => rmSync(root, { recursive: true, force: true }))

const conv = (id: string): Conversation => ({ id, title: 't', messages: [], updatedAt: 1 }) as unknown as Conversation

describe('ConversationStore', () => {
  it("doesn't bring a deleted chat back when a reply finishes after the delete", async () => {
    const store = new ConversationStore(join(root, 'conversations'))
    await store.save(conv('c1'))
    expect(existsSync(join(root, 'conversations', 'c1.json'))).toBe(true)
    await store.delete('c1')
    await expect(store.save(conv('c1'))).rejects.toBeInstanceOf(DeletedConversationError)
    expect(existsSync(join(root, 'conversations', 'c1.json'))).toBe(false)
    expect((await store.list()).map((s) => s.id)).toEqual([])
  })
})
