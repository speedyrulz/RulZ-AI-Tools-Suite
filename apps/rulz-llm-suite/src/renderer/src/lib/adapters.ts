// Pure helpers for the LoRA adapter UI: the load form's list, a chat's strengths, batch labels and the
// converter's default file name. No window, no React, so tests can import it.
import { clampScale } from '@shared/adapters'
import type { InstanceAdapter, LoadAdapter } from '@shared/types'

/** Shown scale, without float noise: 0.8, 1, -0.25. */
export function formatScale(scale: number): string {
  return String(Math.round(scale * 1000) / 1000)
}

/** Ticks or unticks an adapter in the load form. A new one goes last, at strength 1. */
export function toggleLoadAdapter(list: LoadAdapter[], id: string, on: boolean): LoadAdapter[] {
  const has = list.some((a) => a.id === id)
  if (on) return has ? list : [...list, { id, scale: 1 }]
  return has ? list.filter((a) => a.id !== id) : list
}

/** Sets the default strength of a ticked adapter. */
export function setLoadAdapterScale(list: LoadAdapter[], id: string, scale: number): LoadAdapter[] {
  return list.map((a) => (a.id === id ? { ...a, scale: clampScale(scale) } : a))
}

export interface ChatAdapter {
  id: string
  name: string
  /** The strength the model was loaded with. */
  defaultScale: number
  /** The strength this chat uses. */
  scale: number
  /** True when the chat sets its own strength for it. */
  overridden: boolean
}

/**
 * The adapters a chat uses on a running model: each loaded adapter at the chat's strength where the chat
 * sets one, else the default. Ids in the chat's list that the model did not load are ignored.
 */
export function chatAdapters(loaded: InstanceAdapter[], chat: LoadAdapter[] | null | undefined): ChatAdapter[] {
  return loaded.map((a) => {
    const own = chat?.find((c) => c.id === a.id)
    return {
      id: a.id,
      name: a.name,
      defaultScale: a.scale,
      scale: own ? clampScale(own.scale) : a.scale,
      overridden: !!own && clampScale(own.scale) !== a.scale
    }
  })
}

/** The chat's adapter list after setting one adapter's strength: every loaded adapter, the others unchanged. */
export function withChatScale(loaded: InstanceAdapter[], chat: LoadAdapter[] | null | undefined, id: string, scale: number): LoadAdapter[] {
  return chatAdapters(loaded, chat).map((a) => ({ id: a.id, scale: a.id === id ? clampScale(scale) : a.scale }))
}

/** The strength a switched-on adapter comes back to: its default, or 1 when the default is 0. */
export function onScale(a: Pick<ChatAdapter, 'defaultScale'>): number {
  return a.defaultScale !== 0 ? a.defaultScale : 1
}

/** What a batch run records of a running instance's adapters; undefined when it has none. */
export function batchAdapters(loaded: InstanceAdapter[] | undefined): { name: string; scale: number }[] | undefined {
  return loaded?.length ? loaded.map((a) => ({ name: a.name, scale: a.scale })) : undefined
}

/** "+ my-lora x0.8, other x1" for the runs list; empty when the run used none. */
export function batchAdapterLabel(adapters: { name: string; scale: number }[] | undefined): string {
  if (!adapters?.length) return ''
  return adapters.map((a) => `+ ${a.name} x${formatScale(a.scale)}`).join(', ')
}

/** Output file name for converting a PEFT folder: the folder's name plus ".gguf". */
export function defaultConvertName(peftDir: string): string {
  const last =
    peftDir
      .split(/[\\/]+/)
      .filter(Boolean)
      .pop() ?? ''
  const clean = /^[A-Za-z]:$/.test(last) ? '' : last.replace(/[<>:"/\\|?*,\u0000-\u001f]/g, '_').trim().replace(/\.gguf$/i, '')
  return `${clean || 'adapter'}.gguf`
}
