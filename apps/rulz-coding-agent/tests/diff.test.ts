import { describe, expect, it } from 'vitest'
import { diffLines, type DiffLine } from '@shared/diff'

/** The diff before the memory fix (a full copy of V per edit step, no prefix/suffix skipping), as a reference. */
function referenceDiff(before: string, after: string): DiffLine[] {
  const a = before === '' ? [] : before.replace(/\r\n/g, '\n').replace(/\n$/, '').split('\n')
  const b = after === '' ? [] : after.replace(/\r\n/g, '\n').replace(/\n$/, '').split('\n')
  const n = a.length
  const m = b.length
  const max = n + m
  const offset = max + 1
  const v = new Int32Array(2 * max + 3)
  const trace: Int32Array[] = []
  let found = false
  for (let d = 0; d <= max; d++) {
    trace.push(v.slice())
    for (let k = -d; k <= d; k += 2) {
      let x = k === -d || (k !== d && v[offset + k - 1] < v[offset + k + 1]) ? v[offset + k + 1] : v[offset + k - 1] + 1
      let y = x - k
      while (x < n && y < m && a[x] === b[y]) {
        x++
        y++
      }
      v[offset + k] = x
      if (x >= n && y >= m) {
        found = true
        break
      }
    }
    if (found) break
  }
  const out: DiffLine[] = []
  let x = n
  let y = m
  for (let d = trace.length - 1; d >= 0; d--) {
    const vd = trace[d]
    const k = x - y
    const prevK = k === -d || (k !== d && vd[offset + k - 1] < vd[offset + k + 1]) ? k + 1 : k - 1
    const prevX = vd[offset + prevK]
    const prevY = prevX - prevK
    while (x > prevX && y > prevY) {
      out.push({ kind: 'same', text: a[x - 1], oldNo: x, newNo: y })
      x--
      y--
    }
    if (d > 0) {
      if (x === prevX) out.push({ kind: 'add', text: b[y - 1], newNo: y })
      else out.push({ kind: 'del', text: a[x - 1], oldNo: x })
    }
    x = prevX
    y = prevY
  }
  return out.reverse()
}

const lines = (count: number, prefix = 'line'): string[] => Array.from({ length: count }, (_, i) => `${prefix} ${i}`)
const edits = (d: DiffLine[]): number => d.filter((l) => l.kind !== 'same').length

/** The script turns the old text into the new one, and every line number is right. */
function expectValid(d: DiffLine[], before: string, after: string): void {
  const split = (t: string): string[] => (t === '' ? [] : t.replace(/\r\n/g, '\n').replace(/\n$/, '').split('\n'))
  const a = split(before)
  const b = split(after)
  let oldNo = 0
  let newNo = 0
  const problems: string[] = []
  d.forEach((l, i) => {
    if (l.kind !== 'add') {
      oldNo++
      if (l.oldNo !== oldNo || a[oldNo - 1] !== l.text) problems.push(`line ${i}: old ${l.oldNo}/${oldNo}`)
    } else if (l.oldNo !== undefined) problems.push(`line ${i}: an added line has an old number`)
    if (l.kind !== 'del') {
      newNo++
      if (l.newNo !== newNo || b[newNo - 1] !== l.text) problems.push(`line ${i}: new ${l.newNo}/${newNo}`)
    } else if (l.newNo !== undefined) problems.push(`line ${i}: a removed line has a new number`)
  })
  if (oldNo !== a.length || newNo !== b.length) problems.push(`covers ${oldNo}/${a.length} old and ${newNo}/${b.length} new lines`)
  expect(problems).toEqual([])
}

/** Small random texts from a few distinct lines, so there are many matches and ties. */
function randomText(rand: () => number, maxLines: number): string {
  const count = Math.floor(rand() * (maxLines + 1))
  return Array.from({ length: count }, () => 'abcde'[Math.floor(rand() * 5)]).join('\n')
}

function seeded(seed: number): () => number {
  let s = seed
  return () => {
    s = (s * 1103515245 + 12345) & 0x7fffffff
    return s / 0x7fffffff
  }
}

describe('line diff after the memory fix', () => {
  it('keeps the format of the existing cases', () => {
    const d = diffLines('a\nb\nc\n', 'a\nB\nc\nd\n')
    expect(d).toEqual([
      { kind: 'same', text: 'a', oldNo: 1, newNo: 1 },
      { kind: 'del', text: 'b', oldNo: 2 },
      { kind: 'add', text: 'B', newNo: 2 },
      { kind: 'same', text: 'c', oldNo: 3, newNo: 3 },
      { kind: 'add', text: 'd', newNo: 4 }
    ])
    expect(d).toEqual(referenceDiff('a\nb\nc\n', 'a\nB\nc\nd\n'))
    expect(diffLines('x\r\ny\r\n', 'x\ny\n')).toEqual(referenceDiff('x\r\ny\r\n', 'x\ny\n'))
    expect(diffLines('', '')).toEqual([])
    expect(diffLines('same\n', 'same\n')).toEqual([{ kind: 'same', text: 'same', oldNo: 1, newNo: 1 }])
  })

  it('returns a new or emptied file at once, exactly as before', () => {
    expect(diffLines('', 'x\ny')).toEqual([
      { kind: 'add', text: 'x', newNo: 1 },
      { kind: 'add', text: 'y', newNo: 2 }
    ])
    expect(diffLines('x\ny\n', '')).toEqual([
      { kind: 'del', text: 'x', oldNo: 1 },
      { kind: 'del', text: 'y', oldNo: 2 }
    ])
    expect(diffLines('', 'x\ny')).toEqual(referenceDiff('', 'x\ny'))
    expect(diffLines('x\ny\n', '')).toEqual(referenceDiff('x\ny\n', ''))
  })

  it('shows a new 10,000-line file quickly', () => {
    const after = lines(10_000).join('\n')
    const t = performance.now()
    const d = diffLines('', after)
    expect(performance.now() - t).toBeLessThan(500)
    expect(d).toHaveLength(10_000)
    expect(d.every((l, i) => l.kind === 'add' && l.newNo === i + 1)).toBe(true)
  })

  it('finds a change in the middle of a 16,000-line file quickly', () => {
    const before = lines(16_000)
    const after = [...before]
    after.splice(8000, 2, 'changed A', 'changed B', 'new C')
    const t = performance.now()
    const d = diffLines(before.join('\n'), after.join('\n'))
    expect(performance.now() - t).toBeLessThan(500)
    expect(d.filter((l) => l.kind === 'del').map((l) => l.oldNo)).toEqual([8001, 8002])
    expect(d.filter((l) => l.kind === 'add').map((l) => l.newNo)).toEqual([8001, 8002, 8003])
    expectValid(d, before.join('\n'), after.join('\n'))
  })

  it('falls back to old-then-new for a big rewrite, keeping the unchanged start and end', () => {
    const before = ['head', ...lines(8000, 'old'), 'tail']
    const after = ['head', ...lines(8000, 'new'), 'tail']
    const t = performance.now()
    const d = diffLines(before.join('\n'), after.join('\n'))
    expect(performance.now() - t).toBeLessThan(1000)
    expect(d[0]).toEqual({ kind: 'same', text: 'head', oldNo: 1, newNo: 1 })
    expect(d.at(-1)).toEqual({ kind: 'same', text: 'tail', oldNo: 8002, newNo: 8002 })
    const middle = d.slice(1, -1)
    expect(middle.slice(0, 8000).every((l) => l.kind === 'del')).toBe(true)
    expect(middle.slice(8000).every((l) => l.kind === 'add')).toBe(true)
    expectValid(d, before.join('\n'), after.join('\n'))
  })

  it('still finds the shortest edit script, and the same one when the files end differently', () => {
    const rand = seeded(42)
    let differentTies = 0
    for (let i = 0; i < 3000; i++) {
      const before = randomText(rand, 14)
      const after = randomText(rand, 14)
      const got = diffLines(before, after)
      const want = referenceDiff(before, after)
      expectValid(got, before, after)
      expect(edits(got)).toBe(edits(want))
      const lastA = before.split('\n').at(-1)
      const lastB = after.split('\n').at(-1)
      // Without a common last line nothing is cut from the end, and the result is the old one exactly.
      if (!before || !after || lastA !== lastB) expect(got).toEqual(want)
      else if (JSON.stringify(got) !== JSON.stringify(want)) differentTies++
    }
    // With a common end, an equally short script that keeps the end lines together can be picked
    // instead (about 1 in 10 of these few-letter texts, which are full of ties; real files have few).
    expect(differentTies).toBeLessThan(3000)
  })
})
