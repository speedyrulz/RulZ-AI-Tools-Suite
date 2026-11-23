// Adventure prompts: messages, the choosing schema, the forgiving reply parser, context fitting in blocks of 8,
// the prose cleaner, the Markdown export, the world digest and the starters. Pure functions, no model.
import { describe, expect, it } from 'vitest'
import { ADVENTURE_FORMAT, noChanges, emptyState, type Adventure, type AdventureNode, type AdventureSetup, type NodeInput } from '../src/shared/adventure'
import {
  ADVENTURE_STARTERS,
  BEGIN_MESSAGE,
  LEFT_OUT_LINE,
  choosingFinal,
  choosingSchema,
  cleanPassage,
  exportFileName,
  exportMarkdown,
  fitStory,
  parseChoosing,
  passageFinal,
  stepLine,
  systemMessage,
  worldDigest,
  type StoryMessage
} from '../src/shared/adventurePrompts'

const setup = (over: Partial<AdventureSetup> = {}): AdventureSetup => ({
  title: 'The Lantern Road',
  genre: 'Fantasy',
  premise: 'A keeper finds a map.',
  character: 'Wren, a lamplighter',
  tone: 'Dry humour',
  length: 'medium',
  choiceCount: 3,
  modelId: null,
  temperature: null,
  ...over
})

function node(i: number, parentId: string | null, input: NodeInput, text: string, extra: Partial<AdventureNode> = {}): AdventureNode {
  return { id: `n${i}`, parentId, input, text, choices: [], ended: false, changes: noChanges(), state: emptyState(), edited: false, rev: 1, createdAt: i, ...extra }
}

/** A straight story of `count` passages: n0 is the opening, the rest follow typed actions. */
function chain(count: number): AdventureNode[] {
  const out: AdventureNode[] = []
  for (let i = 0; i < count; i++) {
    out.push(node(i, i === 0 ? null : `n${i - 1}`, i === 0 ? { kind: 'start' } : { kind: 'action', text: `step ${i}` }, `Passage ${i}. ${'word '.repeat(40)}`))
  }
  return out
}

describe('messages', () => {
  it('builds a fixed system message from the setup, with the world text when there is one', () => {
    const plain = systemMessage(setup())
    expect(plain).toContain('Genre: Fantasy')
    expect(plain).toContain('Tone: Dry humour')
    expect(plain).toContain('Premise: A keeper finds a map.')
    expect(plain).toContain('Wren, a lamplighter')
    expect(plain).toContain('second person')
    expect(plain).toContain('present tense')
    expect(plain).toContain('about 250 words')
    expect(plain).toContain('never list the choices')
    expect(plain).not.toContain('author\'s notes')
    const withWorld = systemMessage(setup({ length: 'short', world: { folder: 'C:\\n', title: 'Book', text: '## Mara (character)\nA ferrywoman.' } }))
    expect(withWorld).toContain('Use these characters and places; stay consistent with them.')
    expect(withWorld).toContain('A ferrywoman.')
    expect(withWorld).toContain('about 120 words')
    // No state in it: it must stay the same for the whole adventure.
    expect(systemMessage(setup())).toBe(plain)
    expect(systemMessage(setup({ character: '' }))).toContain('invent a fitting one')
  })

  it('writes the reader\'s steps as a user line and puts the state in the last message', () => {
    expect(stepLine({ kind: 'start' })).toBe(BEGIN_MESSAGE)
    expect(stepLine({ kind: 'choice', choiceId: 'c1', text: 'Open the door' })).toBe('The reader chooses: Open the door')
    expect(stepLine({ kind: 'action', text: 'Climb the tree' })).toBe('The reader does: Climb the tree')
    const state = { inventory: ['rusty key'], stats: { Health: 8, Mood: 'wary' }, flags: ['met the ferryman'] }
    const last = passageFinal({ kind: 'action', text: 'Wait' }, state)
    expect(last).toContain('The reader does: Wait')
    expect(last).toContain('Inventory: rusty key')
    expect(last).toContain('Health 8; Mood wary')
    expect(last).toContain('met the ferryman')
    expect(last).toContain('The character only has these items')
    expect(passageFinal({ kind: 'start' }, emptyState())).toBe('Begin the story.')
    expect(passageFinal({ kind: 'action', text: 'Wait' }, emptyState())).toContain('Inventory: nothing')
  })

  it('asks for the right number of choices, and for starting stats at the opening', () => {
    const opening = choosingFinal(setup({ choiceCount: 4 }), emptyState(), true)
    expect(opening).toContain('exactly 4 options')
    expect(opening).toContain('starting stats')
    expect(opening).not.toContain('State before that passage')
    const later = choosingFinal(setup({ choiceCount: 2 }), { inventory: ['lamp'], stats: {}, flags: [] }, false)
    expect(later).toContain('exactly 2 options')
    expect(later).toContain('State before that passage')
    expect(later).toContain('Inventory: lamp')
    expect(later).not.toContain('starting stats')
  })

  it('has a schema whose setStats is a list of name/value', () => {
    const schema = choosingSchema(3) as { required: string[]; properties: Record<string, any> }
    expect(schema.required).toEqual(['choices', 'ended', 'changes'])
    expect(schema.properties.choices.maxItems).toBe(3)
    expect(schema.properties.changes.properties.setStats.type).toBe('array')
    expect(schema.properties.changes.properties.setStats.items.required).toEqual(['name', 'value'])
    expect(() => JSON.stringify(schema)).not.toThrow()
  })
})

describe('parseChoosing', () => {
  const full = JSON.stringify({
    choices: ['Open the door', 'Run away', 'Call out'],
    ended: false,
    changes: { addItems: ['lamp'], removeItems: [], setStats: [{ name: 'Health', value: 7 }], removeStats: [], addFlags: ['saw the door'], removeFlags: [] }
  })

  it('reads a clean reply', () => {
    const p = parseChoosing(full, 3)
    expect(p.ok).toBe(true)
    if (!p.ok) return
    expect(p.value.choices).toEqual(['Open the door', 'Run away', 'Call out'])
    expect(p.value.ended).toBe(false)
    expect(p.value.changes.addItems).toEqual(['lamp'])
    expect(p.value.changes.setStats).toEqual([{ name: 'Health', value: 7 }])
  })

  it('reads the object inside a fence or between sentences, and fills missing change fields', () => {
    const fenced = parseChoosing(`Here you go:\n\`\`\`json\n${full}\n\`\`\`\nHope that helps.`, 3)
    expect(fenced.ok && fenced.value.choices.length).toBe(3)
    const around = parseChoosing(`Sure! ${JSON.stringify({ choices: ['A', 'B'], ended: false })} Done.`, 3)
    expect(around.ok).toBe(true)
    if (around.ok) expect(around.value.changes).toEqual(noChanges())
    const braces = parseChoosing(`{"note": "{x}"} then ${JSON.stringify({ choices: ['Go', 'Stay'], ended: false })}`, 3)
    expect(braces.ok && braces.value.choices).toEqual(['Go', 'Stay'])
  })

  it('trims, de-duplicates, drops empties, strips numbering, and cuts to the asked count', () => {
    const p = parseChoosing(JSON.stringify({ choices: ['  1. Open the door ', 'open the door', '', '- Run away', { text: 'Hide' }, 'Fight', 'Plead'], ended: false }), 3)
    expect(p.ok && p.value.choices).toEqual(['Open the door', 'Run away', 'Hide'])
  })

  it('gives an ended story no choices and accepts a numeric stat written as a string', () => {
    const p = parseChoosing(JSON.stringify({ choices: ['x'], ended: 'true', changes: { setStats: { Gold: '12', Mood: 'calm' }, addItems: 'coin' } }), 3)
    expect(p.ok).toBe(true)
    if (!p.ok) return
    expect(p.value.ended).toBe(true)
    expect(p.value.choices).toEqual([])
    expect(p.value.changes.setStats).toEqual([{ name: 'Gold', value: 12 }, { name: 'Mood', value: 'calm' }])
    expect(p.value.changes.addItems).toEqual(['coin'])
  })

  it('falls back to a list of lines with a warning, and refuses a reply with nothing usable', () => {
    const list = parseChoosing('1. Open the door\n2. Run away\n3. Call out', 3)
    expect(list.ok && list.value.choices).toEqual(['Open the door', 'Run away', 'Call out'])
    expect(list.ok && list.warnings.length).toBe(1)
    expect(parseChoosing('I cannot do that.', 3).ok).toBe(false)
    expect(parseChoosing('{"choices": [], "ended": false}', 3)).toEqual({ ok: false, error: 'The reply had no choices.' })
    expect(parseChoosing('', 3).ok).toBe(false)
  })
})

describe('cleanPassage', () => {
  it('strips a Passage label, fences and surrounding quotes', () => {
    expect(cleanPassage('Passage: You wake in the dark.')).toBe('You wake in the dark.')
    expect(cleanPassage('**Passage 3:** You wake in the dark.')).toBe('You wake in the dark.')
    expect(cleanPassage('## Passage 2\nYou wake in the dark.')).toBe('You wake in the dark.')
    expect(cleanPassage('"You wake in the dark."')).toBe('You wake in the dark.')
    expect(cleanPassage('```\nYou wake in the dark.\n```')).toBe('You wake in the dark.')
    expect(cleanPassage('Passage through the door was easy.')).toBe('Passage through the door was easy.')
  })

  it('removes a list of choices the model wrote anyway, with or without a header', () => {
    expect(cleanPassage('You stand at the gate.\n\nWhat do you do?\n1. Knock\n2. Climb the wall\n3. Leave')).toBe('You stand at the gate.')
    expect(cleanPassage('You stand at the gate.\n\n**Choices:**\n- Knock\n- Climb the wall')).toBe('You stand at the gate.')
    expect(cleanPassage('You stand at the gate.\n\n1. Knock\n\n2. Climb the wall')).toBe('You stand at the gate.')
  })

  it('keeps a single list-like line and an honest closing question', () => {
    expect(cleanPassage('You stand at the gate.\n\n1. Knock')).toBe('You stand at the gate.\n\n1. Knock')
    expect(cleanPassage('You stand at the gate. What do you do?')).toBe('You stand at the gate. What do you do?')
  })
})

describe('fitStory', () => {
  const countChars = async (m: StoryMessage[]): Promise<number | null> => m.reduce((n, x) => n + x.content.length, 0)
  const sys = 'SYSTEM'

  it('keeps everything when it fits and builds alternating turns with the final user message last', async () => {
    const path = chain(3)
    const fit = await fitStory({ system: sys, path, final: 'FINAL', countMessages: countChars, limit: 100000 })
    expect(fit.dropped).toBe(0)
    expect(fit.tooBig).toBe(false)
    expect(fit.messages.map((m) => m.role)).toEqual(['system', 'user', 'assistant', 'user', 'assistant', 'user', 'assistant', 'user'])
    expect(fit.messages[1].content).toBe(BEGIN_MESSAGE)
    expect(fit.messages[3].content).toBe('The reader does: step 1')
    expect(fit.messages[fit.messages.length - 1].content).toBe('FINAL')
  })

  it('drops the oldest passages in blocks of 8, keeping the opening and putting one line in their place', async () => {
    const path = chain(30)
    const one = path[1].text.length
    // Room for about 12 passages beyond the opening.
    const limit = 12 * (one + 30) + 400
    const fit = await fitStory({ system: sys, path, final: 'FINAL', countMessages: countChars, limit })
    expect(fit.tooBig).toBe(false)
    expect(fit.dropped % 8).toBe(0)
    expect(fit.dropped).toBeGreaterThan(0)
    // The opening stays, the newest stays.
    expect(fit.messages[2].content).toBe(path[0].text)
    expect(fit.messages.some((m) => m.content.includes(path[29].text))).toBe(true)
    // One gap line, right after the opening, on the first user turn that follows.
    expect(fit.messages.filter((m) => m.content.includes(LEFT_OUT_LINE))).toHaveLength(1)
    expect(fit.messages[3].content.startsWith(LEFT_OUT_LINE)).toBe(true)
    expect(fit.messages[3].content).toContain(`step ${fit.dropped + 1}`)
    // The cut point is the same for the next turn when only a passage or two is added (cache friendly).
    const next = await fitStory({ system: sys, path: chain(31), final: 'FINAL', countMessages: countChars, limit: limit + one })
    expect(next.dropped).toBe(fit.dropped)
  })

  it('drops the opening too when nothing else helps, and says when even that is too big', async () => {
    const path = chain(3)
    const small = await fitStory({ system: sys, path, final: 'FINAL', countMessages: countChars, limit: path[2].text.length + 120 })
    expect(small.tooBig).toBe(false)
    expect(small.messages[1].content.startsWith(LEFT_OUT_LINE)).toBe(true)
    expect(small.messages.some((m) => m.content === path[0].text)).toBe(false)
    expect(small.messages.some((m) => m.content === path[2].text)).toBe(true)
    const huge = await fitStory({ system: sys, path, final: 'FINAL', countMessages: countChars, limit: 10 })
    expect(huge.tooBig).toBe(true)
  })

  it('with keepNewest the newest passage is never dropped: when it cannot fit the result is tooBig', async () => {
    const path = chain(3)
    const plain = await fitStory({ system: sys, path, final: 'FINAL', countMessages: countChars, limit: 50 })
    expect(plain.tooBig).toBe(false)
    expect(plain.dropped).toBe(3)
    const keep = await fitStory({ system: sys, path, final: 'FINAL', countMessages: countChars, limit: 50, keepNewest: true })
    expect(keep.tooBig).toBe(true)
    expect(keep.messages.some((m) => m.content === path[2].text)).toBe(true)
    // A single passage (the opening) is kept too.
    const one = await fitStory({ system: sys, path: chain(1), final: 'FINAL', countMessages: countChars, limit: 30, keepNewest: true })
    expect(one.tooBig).toBe(true)
    expect(one.messages.some((m) => m.content === chain(1)[0].text)).toBe(true)
    // And it still fits normally when there is room.
    const roomy = await fitStory({ system: sys, path, final: 'FINAL', countMessages: countChars, limit: 100000, keepNewest: true })
    expect(roomy.dropped).toBe(0)
  })

  it('without the opening it keeps dropping in blocks of 8 before it falls to the newest passage only', async () => {
    const path = chain(30)
    path[0] = { ...path[0], text: 'OPENING '.repeat(400) }
    const one = path[1].text.length
    const limit = 8 * (one + 30) + 400
    const fit = await fitStory({ system: sys, path, final: 'FINAL', countMessages: countChars, limit, keepNewest: true })
    expect(fit.tooBig).toBe(false)
    expect(fit.dropped).toBe(24)
    expect(fit.messages.some((m) => m.content === path[0].text)).toBe(false)
    expect(fit.messages[1].content.startsWith(LEFT_OUT_LINE)).toBe(true)
    expect(fit.messages.some((m) => m.content === path[29].text)).toBe(true)
    expect(fit.messages.filter((m) => m.role === 'assistant')).toHaveLength(6)
  })

  it('estimates 4 characters a token when counting fails, and says so', async () => {
    const path = chain(2)
    const fit = await fitStory({ system: sys, path, final: 'FINAL', countMessages: async () => null, limit: 100000 })
    expect(fit.estimated).toBe(true)
    expect(fit.tooBig).toBe(false)
    const thrown = await fitStory({ system: sys, path, final: 'FINAL', countMessages: async () => Promise.reject(new Error('no')), limit: 3 })
    expect(thrown.estimated).toBe(true)
    expect(thrown.tooBig).toBe(true)
  })

  it('handles the opening request with an empty path', async () => {
    const fit = await fitStory({ system: sys, path: [], final: BEGIN_MESSAGE, countMessages: countChars, limit: 1000 })
    expect(fit.messages).toEqual([{ role: 'system', content: sys }, { role: 'user', content: BEGIN_MESSAGE }])
  })
})

describe('exportMarkdown and exportFileName', () => {
  const adv = (): Adventure => {
    const n0 = node(0, null, { kind: 'start' }, 'The gate is shut.', { choices: [{ id: 'c1', text: 'Knock' }] })
    const n1 = node(1, 'n0', { kind: 'choice', choiceId: 'c1', text: 'Knock' }, 'A voice answers.')
    const n2 = node(2, 'n1', { kind: 'action', text: 'Say hello' }, 'It is the end.', { ended: true })
    const side = node(3, 'n0', { kind: 'action', text: 'Climb' }, 'Another path.')
    return {
      format: ADVENTURE_FORMAT,
      id: 'a-1234567',
      setup: setup({ world: { folder: 'C:\\book', title: 'My Book', text: 'x' } }),
      nodes: { n0, n1, n2, n3: side },
      rootId: 'n0',
      currentId: 'n2',
      rev: 5,
      createdAt: 1,
      updatedAt: 2
    }
  }

  it('writes the title, a genre line, the passages with the reader\'s steps in italics, and the end', () => {
    const md = exportMarkdown(adv(), 'n2')
    expect(md).toBe(
      ['# The Lantern Road', '', '*Fantasy \u00b7 From the Writing project My Book*', '', 'The gate is shut.', '', '*You chose: Knock*', '', 'A voice answers.', '', '*You: Say hello*', '', 'It is the end.', '', '**The end.**', ''].join('\n')
    )
    // Another path gives another story; a path that did not end has no end line.
    const other = exportMarkdown(adv(), 'n3')
    expect(other).toContain('*You: Climb*')
    expect(other).not.toContain('**The end.**')
    expect(other).not.toContain('A voice answers.')
  })

  it('makes a file name Windows accepts', () => {
    expect(exportFileName('The Lantern Road')).toBe('The Lantern Road.md')
    expect(exportFileName('What? "Now": a/b\\c|d*')).toBe('What Now a b c d.md')
    expect(exportFileName('   ...  ')).toBe('Adventure.md')
    expect(exportFileName('CON')).toBe('CON_.md')
    expect(exportFileName('x'.repeat(200)).length).toBe(83)
  })
})

describe('worldDigest', () => {
  const entry = (name: string, kind: string, body: string) => ({ name, kind, body })

  it('puts characters before places before the rest, whole entries only', () => {
    const text = worldDigest({
      title: 'Book',
      chapters: [{ title: 'One', summary: 'A ferry crosses.' }, { title: 'Two', summary: '' }],
      bible: [entry('Old Law', 'lore', 'Nobody sails at night.'), entry('The Dock', 'place', 'A wet dock.'), entry('Mara', 'character', 'A ferrywoman.')]
    })
    expect(text.indexOf('A ferry crosses.')).toBeGreaterThan(-1)
    expect(text).not.toContain('Two:')
    expect(text.indexOf('Mara')).toBeLessThan(text.indexOf('The Dock'))
    expect(text.indexOf('The Dock')).toBeLessThan(text.indexOf('Old Law'))
  })

  it('stays under the cap and skips an entry that does not fit instead of cutting it', () => {
    const big = 'x'.repeat(500)
    const text = worldDigest({ title: 'B', chapters: [], bible: [entry('Big', 'character', big), entry('Small', 'character', 'tiny'), entry('Huge', 'place', big)] }, 700)
    expect(text.length).toBeLessThanOrEqual(700)
    expect(text).toContain('Big')
    expect(text).toContain('Small')
    expect(text).not.toContain('Huge')
    // Whole entries: every included body is complete.
    expect(text).toContain(big)
  })

  it('still gives something when the first entry is bigger than the cap', () => {
    const text = worldDigest({ title: 'B', chapters: [], bible: [entry('Epic', 'character', 'y'.repeat(5000))] }, 1000)
    expect(text.length).toBeLessThanOrEqual(1000)
    expect(text).toContain('Epic')
    expect(text.endsWith('\u2026')).toBe(true)
    expect(worldDigest({ title: 'B', chapters: [], bible: [] })).toBe('')
  })
})

describe('starters', () => {
  it('has eight varied starters with unique ids and every field filled', () => {
    expect(ADVENTURE_STARTERS).toHaveLength(8)
    expect(new Set(ADVENTURE_STARTERS.map((s) => s.id)).size).toBe(8)
    expect(new Set(ADVENTURE_STARTERS.map((s) => s.genre)).size).toBe(8)
    for (const s of ADVENTURE_STARTERS) {
      for (const v of [s.id, s.title, s.genre, s.premise, s.character, s.tone]) expect(v.trim().length).toBeGreaterThan(2)
    }
  })
})
