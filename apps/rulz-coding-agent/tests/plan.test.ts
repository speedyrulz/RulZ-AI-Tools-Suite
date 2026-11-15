import { describe, expect, it } from 'vitest'
import { dropPlanNotes, MAX_PLAN_ITEMS, parsePlan, planInHistory, planNote, planResult } from '../src/main/agent/plan'
import type { WireMessage } from '../src/shared/types'

const plan = (args: Record<string, unknown>) => {
  const r = parsePlan(args)
  if ('error' in r) throw new Error(r.error)
  return r.plan
}

describe('the checklist (set_plan)', () => {
  it('reads the checklist as one text, a line per step with a box in front', () => {
    expect(plan({ steps: '[x] Build the main page\n[>] Add the buttons\n[ ] Check that it works\n' })).toEqual([
      { text: 'Build the main page', status: 'done' },
      { text: 'Add the buttons', status: 'doing' },
      { text: 'Check that it works', status: 'todo' }
    ])
    // Windows line ends, a box with a word in it, an empty box, and no box at all.
    expect(plan({ steps: '[done] One\r\n[In Progress] Two\r\n[] Three\r\n[todo] Four\r\nFive' }).map((p) => `${p.status}:${p.text}`)).toEqual([
      'done:One',
      'doing:Two',
      'todo:Three',
      'todo:Four',
      'todo:Five'
    ])
  })

  it('also reads steps sent as a list of objects', () => {
    expect(
      plan({
        steps: [
          { text: 'Build the main page', status: 'done' },
          { text: 'Add the buttons', status: 'doing' },
          { text: 'Check that it works', status: 'todo' }
        ]
      })
    ).toEqual([
      { text: 'Build the main page', status: 'done' },
      { text: 'Add the buttons', status: 'doing' },
      { text: 'Check that it works', status: 'todo' }
    ])
  })

  it("accepts other tools' status words, other names for the text and done flags", () => {
    expect(
      plan({
        todos: [
          { content: 'One', status: 'completed' },
          { title: 'Two', state: 'in_progress' },
          { step: 'Three', status: 'pending' },
          { task: 'Four', done: true },
          { description: 'Five', status: 'something else' }
        ]
      }).map((p) => p.status)
    ).toEqual(['done', 'doing', 'todo', 'done', 'todo'])
  })

  it('accepts plain lines, with or without boxes and numbers', () => {
    expect(plan({ steps: ['- [x] Make the folder', '2. [>] Write the page', '[ ] Test it', 'Tell the owner'] })).toEqual([
      { text: 'Make the folder', status: 'done' },
      { text: 'Write the page', status: 'doing' },
      { text: 'Test it', status: 'todo' },
      { text: 'Tell the owner', status: 'todo' }
    ])
  })

  it('accepts the list as one text: JSON, or a line per step', () => {
    expect(plan({ steps: '[{"text": "A", "status": "done"}, {"text": "B", "status": "todo"}]' }).map((p) => p.status)).toEqual(['done', 'todo'])
    expect(plan({ plan: '1. First thing\n2. Second thing\n\n' }).map((p) => p.text)).toEqual(['First thing', 'Second thing'])
  })

  it('keeps a step that starts with a number or a bracket word as it is', () => {
    expect(plan({ steps: ['3D viewer for the model', '[Optional] dark mode'] }).map((p) => p.text)).toEqual(['3D viewer for the model', '[Optional] dark mode'])
  })

  it('shortens long steps to one line and keeps at most the first steps', () => {
    const long = plan({ steps: [{ text: `Do\n  this   ${'x'.repeat(400)}`, status: 'todo' }] })[0].text
    expect(long.length).toBeLessThanOrEqual(160)
    expect(long.startsWith('Do this x')).toBe(true)
    const many = parsePlan({ steps: Array.from({ length: MAX_PLAN_ITEMS + 5 }, (_, i) => `Step ${i + 1}`) })
    expect('plan' in many && many.plan).toHaveLength(MAX_PLAN_ITEMS)
    expect('plan' in many && many.dropped).toBe(5)
  })

  it('explains what it needs when there is no usable list', () => {
    expect(parsePlan({})).toMatchObject({ error: expect.stringContaining('"steps"') })
    expect(parsePlan({ steps: 7 })).toMatchObject({ error: expect.stringContaining('"steps"') })
    expect(parsePlan({ steps: [{ status: 'todo' }, '', null] })).toMatchObject({ error: expect.stringContaining('at least one step') })
  })

  it('answers with where the plan stands', () => {
    expect(planResult(plan({ steps: [{ text: 'A', status: 'done' }, { text: 'B', status: 'doing' }, { text: 'C', status: 'todo' }] }))).toBe(
      'Checklist saved. 1 of 3 steps done. Now: B'
    )
    expect(planResult(plan({ steps: [{ text: 'A', status: 'done' }, { text: 'B', status: 'todo' }] }))).toBe('Checklist saved. 1 of 2 steps done. Next: B')
    expect(planResult(plan({ steps: [{ text: 'A', status: 'done' }] }))).toBe('Checklist saved. All 1 steps are done.')
    expect(planResult(plan({ steps: ['A'] }), 3)).toMatch(/Only the first 20 steps are kept/)
  })

  it('writes the checklist as a note and recognises it, or the call, in the conversation', () => {
    const note = planNote(plan({ steps: [{ text: 'A', status: 'done' }, { text: 'B', status: 'doing' }, { text: 'C', status: 'todo' }] }))
    expect(note).toContain('[x] A\n[>] B\n[ ] C')
    expect(note).toContain('set_plan')
    expect(planInHistory([{ role: 'user', content: 'Make a page' }])).toBe(false)
    expect(planInHistory([{ role: 'user', content: note }])).toBe(true)
    expect(
      planInHistory([{ role: 'assistant', content: '', tool_calls: [{ id: 'c1', type: 'function', function: { name: 'set_plan', arguments: '{}' } }] }])
    ).toBe(true)
    // Only the model's own call counts, not a tool result that happens to mention the tool.
    expect(planInHistory([{ role: 'tool', tool_call_id: 'c1', content: 'set_plan' }])).toBe(false)
  })

  it('takes the checklist notes out again, and keeps what else is in the message', () => {
    const note = planNote(plan({ steps: '[x] A\n[>] B' }))
    const messages: WireMessage[] = [
      { role: 'user', content: 'Make a page', task: 't1' },
      { role: 'assistant', content: 'Working on it.' },
      { role: 'user', content: note },
      { role: 'tool', tool_call_id: 'c1', content: note },
      { role: 'user', content: `${note}\n\n[Note: 1 background command is still running.]` },
      { role: 'user', content: `Before.\n\n${note}`, task: 't2' }
    ]
    dropPlanNotes(messages)
    expect(messages.map((m) => m.content)).toEqual([
      'Make a page',
      'Working on it.',
      // Only notes the app wrote (user messages) are taken out; a tool result is left as it was.
      note,
      '[Note: 1 background command is still running.]',
      'Before.'
    ])
    expect(planInHistory(messages.filter((m) => m.role === 'user'))).toBe(false)
  })
})
