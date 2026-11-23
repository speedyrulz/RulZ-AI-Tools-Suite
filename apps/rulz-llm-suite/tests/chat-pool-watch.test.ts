// The chat store while the practice pool owns a chat: the answer to re:watch must not replace a newer copy.
import { beforeAll, describe, expect, it, vi } from 'vitest'
import type { Conversation } from '@shared/types'

const listeners = new Map<string, ((p: unknown) => void)[]>()
const watchReplies: { id: string | null; resolve: (c: Conversation | null) => void }[] = []

vi.stubGlobal('window', {
  matchMedia: () => ({ matches: false, addEventListener: () => undefined }),
  rulz: {
    invoke: (channel: string, ...args: unknown[]) => {
      if (channel === 're:watch') return new Promise((resolve) => watchReplies.push({ id: args[0] as string | null, resolve }))
      if (channel === 'conversations:list' || channel === 'folders:list' || channel === 'presets:list' || channel === 're:poolChats') return Promise.resolve([])
      return Promise.resolve(null)
    },
    on: (channel: string, fn: (p: unknown) => void) => {
      listeners.set(channel, [...(listeners.get(channel) ?? []), fn])
      return () => undefined
    }
  }
})

const fire = (channel: string, payload: unknown): void => listeners.get(channel)?.forEach((fn) => fn(payload))
const conv = (id: string, poolRev: number, extra: Partial<Conversation> = {}): Conversation =>
  ({ id, title: id, folder: null, pinned: false, createdAt: 0, updatedAt: 0, modelId: null, systemPrompt: '', messages: [], poolRev, ...extra }) as unknown as Conversation

let useChat: typeof import('../src/renderer/src/stores/chat').useChat
beforeAll(async () => {
  ;({ useChat } = await import('../src/renderer/src/stores/chat'))
  await useChat.getState().init()
})

const tick = (): Promise<void> => new Promise((r) => setTimeout(r, 0))

describe('watching a chat the pool runs', () => {
  it('asks to watch the chat on screen once the pool owns it, and shows the reply when nothing newer came', async () => {
    useChat.setState({ current: conv('A', 3) })
    fire('re:poolChats', ['A'])
    expect(watchReplies.at(-1)?.id).toBe('A')
    watchReplies.at(-1)!.resolve(conv('A', 4, { title: 'from main' }))
    await tick()
    expect(useChat.getState().current?.title).toBe('from main')
  })

  it('ignores a reply that is older than a live update which arrived first', async () => {
    useChat.setState({ current: conv('B', 3) })
    fire('re:poolChats', ['B'])
    const reply = watchReplies.at(-1)!
    expect(reply.id).toBe('B')
    fire('re:chatLive', conv('B', 6, { title: 'live' }))
    reply.resolve(conv('B', 5, { title: 'older reply' }))
    await tick()
    expect(useChat.getState().current).toMatchObject({ title: 'live', poolRev: 6 })
  })

  it('ignores a reply that is older than what is shown, even with no live update', async () => {
    useChat.setState({ current: conv('C', 9) })
    fire('re:poolChats', ['C'])
    watchReplies.at(-1)!.resolve(conv('C', 7, { title: 'stale' }))
    await tick()
    expect(useChat.getState().current?.poolRev).toBe(9)
  })

  it('ignores a reply for a chat that is no longer the watched one', async () => {
    useChat.setState({ current: conv('D', 1) })
    fire('re:poolChats', ['D', 'E'])
    const reply = watchReplies.at(-1)!
    expect(reply.id).toBe('D')
    // The user moves to another pool chat before the answer comes, and back.
    useChat.setState({ current: conv('E', 1) })
    await tick()
    useChat.setState({ current: conv('D', 1, { title: 'shown again' }) })
    reply.resolve(conv('D', 2, { title: 'late' }))
    await tick()
    expect(useChat.getState().current?.title).toBe('shown again')
  })
})
