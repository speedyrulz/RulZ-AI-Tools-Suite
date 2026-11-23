import { existsSync, mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterAll, describe, expect, it, vi } from 'vitest'
import type { Conversation } from '@shared/types'

// No Recycle Bin in tests: the store falls back to deleting the file.
vi.mock('electron', () => ({ shell: { trashItem: () => Promise.reject(new Error('no shell')) } }))
const { ConversationStore, DeletedConversationError, StaleConversationError } = await import('../src/main/chat/conversations')

const root = mkdtempSync(join(tmpdir(), 'rulz-convs-'))
afterAll(() => rmSync(root, { recursive: true, force: true }))

const conv = (id: string, poolRev?: number): Conversation => ({ id, title: 't', messages: [], updatedAt: 1, ...(poolRev === undefined ? {} : { poolRev }) }) as unknown as Conversation

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

describe('ConversationStore and the practice pool', () => {
  const fresh = () => new ConversationStore(join(mkdtempSync(join(root, 'p-')), 'conversations'))

  it('refuses the window\'s save and delete of a chat the pool owns, and takes saves again after release', async () => {
    const store = fresh()
    await store.save(conv('a'))
    store.own('a')
    expect(store.isOwned('a')).toBe(true)
    expect(store.ownedIds()).toEqual(['a'])
    await expect(store.save(conv('a'))).rejects.toBeInstanceOf(StaleConversationError)
    await expect(store.save(conv('a'))).rejects.toThrow(/The batch run is using this chat, so this copy was not saved\./)
    await expect(store.delete('a')).rejects.toThrow(/using this chat/)
    // The pool's own save is not refused, and does not need to be the window's.
    await store.savePool(conv('a', 1))
    store.release('a')
    expect(store.isOwned('a')).toBe(false)
    // The window's copy at the pool's last revision is the newest: accepted.
    await expect(store.save(conv('a', 1))).resolves.toMatchObject({ id: 'a' })
  })

  it('refuses a window copy older than the pool\'s newest save, for as long as the app runs', async () => {
    const store = fresh()
    await store.savePool(conv('b', 1))
    await store.savePool(conv('b', 2))
    await store.savePool(conv('b', 3))
    // A copy made before the pool's last save (revision 2, or none at all).
    await expect(store.save(conv('b', 2))).rejects.toThrow(/A batch run changed this chat after this copy was made, so this copy was not saved\./)
    await expect(store.save(conv('b'))).rejects.toBeInstanceOf(StaleConversationError)
    await expect(store.save(conv('b', 3))).resolves.toBeDefined()
    await expect(store.save(conv('b', 4))).resolves.toBeDefined()
    // A chat the pool never wrote saves as before.
    await expect(store.save(conv('plain'))).resolves.toBeDefined()
    await expect(store.save(conv('plain'))).resolves.toBeDefined()
  })

  it('never lowers the newest revision, and lists what the pool saved', async () => {
    const store = fresh()
    await store.savePool(conv('c', 5))
    await store.savePool(conv('c', 2))
    await expect(store.save(conv('c', 4))).rejects.toBeInstanceOf(StaleConversationError)
    expect((await store.list()).map((x) => x.id)).toEqual(['c'])
    expect((await store.get('c'))?.poolRev).toBe(2)
  })
})
