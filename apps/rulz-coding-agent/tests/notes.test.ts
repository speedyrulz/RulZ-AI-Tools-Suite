import { describe, expect, it } from 'vitest'
import { cutTurnContent, NoteQueue, noteMessage } from '../src/main/agent/notes'

describe('NoteQueue', () => {
  it('keeps notes in order until taken, and cancels by id', () => {
    const q = new NoteQueue()
    expect(q.add('   ', false)).toBeNull()
    const a = q.add(' first ', false)!
    q.add('second', false)
    expect(a.text).toBe('first')
    expect(q.cancel(a.id)).toBe(true)
    expect(q.cancel(a.id)).toBe(false)
    expect(q.take().map((n) => n.text)).toEqual(['second'])
    expect(q.size).toBe(0)
  })

  it('cuts only the current model turn, never the task', () => {
    const q = new NoteQueue()
    const task = new AbortController()
    // Between turns there is nothing to cut; the note simply waits.
    q.add('later', true)
    const turn = q.beginTurn(task.signal)
    expect(turn.aborted).toBe(false)
    q.add('now', true)
    expect(turn.aborted).toBe(true)
    expect(task.signal.aborted).toBe(false)
    expect(q.endTurn()).toBe(true)
    // The next turn starts clean.
    const next = q.beginTurn(task.signal)
    expect(next.aborted).toBe(false)
    expect(q.endTurn()).toBe(false)
  })

  it('a stopped task aborts the turn without counting as a cut', () => {
    const q = new NoteQueue()
    const task = new AbortController()
    const turn = q.beginTurn(task.signal)
    task.abort()
    expect(turn.aborted).toBe(true)
    expect(q.endTurn()).toBe(false)
  })
})

describe('noteMessage', () => {
  it('passes notes on between steps', () => {
    expect(noteMessage(['Use blue'])).toMatch(/added a note while you were working[\s\S]*\nUse blue$/)
    expect(noteMessage(['a', 'b'])).toMatch(/1\. a\n2\. b$/)
  })
  it('quotes the thinking of a cut turn', () => {
    const text = noteMessage(['It is 42'], { thinking: 'maybe 7\nor 8' })
    expect(text).toContain('> maybe 7\n> or 8')
    expect(text).toMatch(/It is 42$/)
    expect(noteMessage(['x'], { thinking: '' })).not.toContain('thinking so far')
  })
  it('never leaves the cut turn empty', () => {
    expect(cutTurnContent('  ')).toMatch(/stopped me/)
    expect(cutTurnContent('Half')).toBe('Half')
  })
})
