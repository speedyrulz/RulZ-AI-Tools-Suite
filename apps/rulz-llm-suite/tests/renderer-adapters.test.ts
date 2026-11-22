import { describe, expect, it } from 'vitest'
import type { InstanceAdapter } from '../src/shared/types'
import {
  batchAdapterLabel,
  batchAdapters,
  sameBatchAdapters,
  chatAdapters,
  defaultConvertName,
  formatScale,
  onScale,
  setLoadAdapterScale,
  toggleLoadAdapter,
  withChatScale
} from '../src/renderer/src/lib/adapters'

const loaded: InstanceAdapter[] = [
  { id: 'a/one.gguf', name: 'one', path: 'C:/m/a/one.gguf', scale: 0.8 },
  { id: 'two.gguf', name: 'two', path: 'C:/m/two.gguf', scale: 1 }
]

describe('load form adapters', () => {
  it('keeps the order the user ticked them in', () => {
    let list = toggleLoadAdapter([], 'b', true)
    list = toggleLoadAdapter(list, 'a', true)
    expect(list.map((a) => a.id)).toEqual(['b', 'a'])
    expect(toggleLoadAdapter(list, 'b', true)).toBe(list)
    expect(toggleLoadAdapter(list, 'b', false).map((a) => a.id)).toEqual(['a'])
  })

  it('clamps the default strength', () => {
    const list = toggleLoadAdapter([], 'a', true)
    expect(setLoadAdapterScale(list, 'a', 9)[0].scale).toBe(4)
    expect(setLoadAdapterScale(list, 'a', -9)[0].scale).toBe(-4)
    expect(setLoadAdapterScale(list, 'a', 0.55)[0].scale).toBe(0.55)
    expect(setLoadAdapterScale(list, 'zzz', 2)).toEqual(list)
  })
})

describe('chat adapter strengths', () => {
  it('uses the model defaults when the chat sets none', () => {
    for (const chat of [null, undefined, []]) {
      expect(chatAdapters(loaded, chat).map((a) => [a.scale, a.overridden])).toEqual([
        [0.8, false],
        [1, false]
      ])
    }
  })

  it('applies the chat strengths, keeps defaults for adapters left out and ignores unknown ids', () => {
    const r = chatAdapters(loaded, [
      { id: 'two.gguf', scale: 0 },
      { id: 'gone.gguf', scale: 2 }
    ])
    expect(r.map((a) => [a.id, a.scale, a.defaultScale, a.overridden])).toEqual([
      ['a/one.gguf', 0.8, 0.8, false],
      ['two.gguf', 0, 1, true]
    ])
  })

  it('does not call a strength equal to the default an override', () => {
    expect(chatAdapters(loaded, [{ id: 'two.gguf', scale: 1 }])[1].overridden).toBe(false)
  })

  it('creates the full list from the defaults on the first change', () => {
    expect(withChatScale(loaded, null, 'two.gguf', 1.5)).toEqual([
      { id: 'a/one.gguf', scale: 0.8 },
      { id: 'two.gguf', scale: 1.5 }
    ])
    // A second change keeps the first.
    const first = withChatScale(loaded, null, 'two.gguf', 1.5)
    expect(withChatScale(loaded, first, 'a/one.gguf', 0)).toEqual([
      { id: 'a/one.gguf', scale: 0 },
      { id: 'two.gguf', scale: 1.5 }
    ])
    expect(withChatScale(loaded, null, 'two.gguf', 50)[1].scale).toBe(4)
  })

  it('switches back on to the default, or 1 when the default is 0', () => {
    expect(onScale({ defaultScale: 0.8 })).toBe(0.8)
    expect(onScale({ defaultScale: 0 })).toBe(1)
  })
})

describe('batch run adapters', () => {
  it('records name and default strength, and nothing when none are loaded', () => {
    expect(batchAdapters(loaded)).toEqual([
      { name: 'one', scale: 0.8 },
      { name: 'two', scale: 1 }
    ])
    expect(batchAdapters([])).toBeUndefined()
    expect(batchAdapters(undefined)).toBeUndefined()
  })

  it('words them for the runs list', () => {
    expect(batchAdapterLabel([{ name: 'my-lora', scale: 0.8 }])).toBe('+ my-lora x0.8')
    expect(batchAdapterLabel([{ name: 'a', scale: 1 }, { name: 'b', scale: 0.35 }])).toBe('+ a x1, + b x0.35')
    expect(batchAdapterLabel(undefined)).toBe('')
    expect(batchAdapterLabel([])).toBe('')
    expect(formatScale(0.30000000000000004)).toBe('0.3')
  })
})

describe('defaultConvertName', () => {
  it('uses the PEFT folder name', () => {
    expect(defaultConvertName('C:\\loras\\my-lora')).toBe('my-lora.gguf')
    expect(defaultConvertName('/home/me/loras/my-lora/')).toBe('my-lora.gguf')
    expect(defaultConvertName('D:\\')).toBe('adapter.gguf')
    expect(defaultConvertName('')).toBe('adapter.gguf')
  })

  it('keeps commas out, since llama.cpp cannot load such a path', () => {
    expect(defaultConvertName('C:\\x\\a,b')).toBe('a_b.gguf')
  })
})

describe('sameBatchAdapters', () => {
  const a = [{ name: 'x', scale: 1 }, { name: 'y', scale: 0.5 }]
  it('is equal for the same names, strengths and order, and for both absent', () => {
    expect(sameBatchAdapters(a, [...a.map((e) => ({ ...e }))])).toBe(true)
    expect(sameBatchAdapters(undefined, undefined)).toBe(true)
    expect(sameBatchAdapters(undefined, [])).toBe(true)
  })
  it('differs on strength, order, name or presence', () => {
    expect(sameBatchAdapters(a, [a[0], { name: 'y', scale: 0.6 }])).toBe(false)
    expect(sameBatchAdapters(a, [a[1], a[0]])).toBe(false)
    expect(sameBatchAdapters(a, [a[0], { name: 'z', scale: 0.5 }])).toBe(false)
    expect(sameBatchAdapters(a, undefined)).toBe(false)
    expect(sameBatchAdapters(undefined, a)).toBe(false)
  })
})
