import { describe, expect, it } from 'vitest'
import { DEFAULT_SAMPLING } from '../src/shared/defaults'
import type { BibleEntry, NovelAction, NovelChapter, NovelScene } from '../src/shared/novel'
import {
  CHECK_SCHEMA,
  cleanProse,
  defaultProject,
  entryListText,
  entryShortText,
  entryText,
  FICTION_SAMPLING,
  interviewSystem,
  isProseAction,
  OUTLINE_CHAPTERS_SCHEMA,
  OUTLINE_SCENES_SCHEMA,
  parseCheckReply,
  parseCheckReplyResult,
  parseOutlineChapters,
  parseOutlineScenes,
  parseSummary,
  parseUpdateStateReply,
  parseUpdateStateReplyResult,
  resolveEntryRef,
  sceneText,
  SUMMARY_SCHEMA,
  summaryText,
  systemText,
  taskText,
  UPDATE_STATE_SCHEMA
} from '../src/shared/novelPrompts'

const ACTIONS: NovelAction[] = [
  'continue', 'rewrite', 'expand', 'condense', 'dialogue', 'draftScene',
  'summarizeScene', 'summarizeChapter', 'outlineChapters', 'outlineScenes'
]

const entry = (over: Partial<BibleEntry> = {}): BibleEntry => ({
  id: 'mara-venn', name: 'Mara Venn', kind: 'character', aliases: [], triggers: [], priority: 50, always: false, body: '', problems: [], ...over
})

describe('defaults', () => {
  it('has the fiction sampling preset on top of the app defaults', () => {
    expect(FICTION_SAMPLING).toEqual({
      ...DEFAULT_SAMPLING,
      temperature: 0.9,
      minP: 0.05,
      topP: 1,
      topK: 0,
      repeatPenalty: 1,
      dryMultiplier: 0.8,
      xtcProbability: 0.5,
      xtcThreshold: 0.1,
      thinking: 'off'
    })
  })

  it('creates a project with the documented fields', () => {
    const before = Date.now()
    const project = defaultProject('The Salt Road')
    expect(project.format).toBe(1)
    expect(project.id).toMatch(/^p-[0-9a-z]{6}$/)
    expect(project.title).toBe('The Salt Road')
    expect(project.modelId).toBeNull()
    expect(project.adapters).toBeNull()
    expect(project.povTense).toBe('')
    expect(project.budget).toEqual({ replyTokens: 1024, proseShare: 0.35, bibleShare: 0.3 })
    expect(project.sampling).toEqual(FICTION_SAMPLING)
    expect(project.createdAt).toBe(project.updatedAt)
    expect(Date.parse(project.createdAt)).toBeGreaterThanOrEqual(before - 1000)
    expect(new Date(project.createdAt).toISOString()).toBe(project.createdAt)
  })

  it('gives each project its own id and its own copy of the preset lists', () => {
    const a = defaultProject('A')
    const b = defaultProject('B')
    expect(a.id).not.toBe(b.id)
    a.sampling.avoidWords.push('suddenly')
    a.sampling.drySequenceBreakers.push('x')
    expect(b.sampling.avoidWords).toEqual([])
    expect(FICTION_SAMPLING.avoidWords).toEqual([])
    expect(FICTION_SAMPLING.drySequenceBreakers).toEqual(DEFAULT_SAMPLING.drySequenceBreakers)
  })
})

describe('prompt texts', () => {
  it('knows which actions write prose', () => {
    expect(ACTIONS.filter(isProseAction)).toEqual(['continue', 'rewrite', 'expand', 'condense', 'dialogue', 'draftScene'])
  })

  it('builds the system text from the style and the POV/tense line', () => {
    const text = systemText('continue', '  Dry, spare, no adverbs.  ', 'Close third, past tense')
    expect(text).toContain('You are a fiction writer working on a novel with the author.')
    expect(text).toContain('Dry, spare, no adverbs.')
    expect(text).toContain('Close third, past tense')
    expect(text).toMatch(/no commentary/)
    expect(text).toMatch(/do not\s+resolve the scene early/)
    expect(text).toMatch(/Stay within the beats/)
    // Style comes before the POV line, instructions last.
    expect(text.indexOf('Dry, spare')).toBeLessThan(text.indexOf('Close third'))
    expect(text.indexOf('Close third')).toBeLessThan(text.indexOf('no commentary'))
  })

  it('leaves out an empty style and POV line, and asks structured actions for JSON only', () => {
    const prose = systemText('draftScene', '', '   ')
    expect(prose).not.toContain("The author's style notes")
    expect(prose).not.toContain('Point of view')
    const structured = systemText('summarizeScene', 'Style', '')
    expect(structured).toContain('JSON')
    expect(structured).not.toContain('Write only the story text')
  })

  it('gives every action a task text', () => {
    for (const action of ACTIONS) expect(taskText(action, {}).length).toBeGreaterThan(20)
  })

  it('continue: about N words from the exact point, without repeating the last sentence', () => {
    const text = taskText('continue', { targetWords: 300 })
    expect(text).toContain('exact point')
    expect(text).toContain('about 300 words')
    expect(text).toMatch(/Do not repeat the last sentence/)
    expect(taskText('continue', {})).not.toContain('about')
  })

  it('rewrite: one instruction per variant, the selection and only the replacement', () => {
    expect(taskText('rewrite', { variant: 'tighter' })).toMatch(/tighter/)
    expect(taskText('rewrite', { variant: 'vivid' })).toMatch(/more vivid/)
    expect(taskText('rewrite', { variant: 'tone', note: 'bleakly funny' })).toMatch(/different tone/)
    expect(taskText('rewrite', { variant: 'show' })).toMatch(/show instead of tell/)
    expect(taskText('rewrite', { variant: 'custom', note: 'make her angrier' })).toMatch(/author's note/)
    const withSelection = taskText('rewrite', { variant: 'vivid', selection: 'The rain fell.', note: 'keep the rain' })
    expect(withSelection).toContain('Selected passage:\nThe rain fell.')
    expect(withSelection).toContain("The author's note: keep the rain")
    expect(withSelection).toMatch(/only the replacement/)
    expect(taskText('rewrite', {})).toMatch(/tighter/)
  })

  it('expand, condense, dialogue and draft scene', () => {
    expect(taskText('expand', { selection: 'x' })).toMatch(/Expand/)
    expect(taskText('condense', { selection: 'x' })).toMatch(/Condense/)
    expect(taskText('dialogue', {})).toMatch(/dialogue for the current beat/)
    const draft = taskText('draftScene', { targetWords: 2500 })
    expect(draft).toContain('from its beats')
    expect(draft).toContain('about 2500 words')
  })

  it('summaries: at most 200 words, past tense, JSON for SUMMARY_SCHEMA', () => {
    for (const action of ['summarizeScene', 'summarizeChapter'] as const) {
      const text = taskText(action, {})
      expect(text).toContain('200 words')
      expect(text).toContain('past tense')
      expect(text).toContain('"summary"')
    }
    expect(taskText('summarizeChapter', {})).toContain('scene summaries')
  })

  it('outline actions use the count, and scenes point at the bible ids', () => {
    expect(taskText('outlineChapters', { count: 12 })).toContain('12 chapters')
    expect(taskText('outlineChapters', {})).not.toMatch(/\d+ chapters/)
    expect(taskText('outlineScenes', { count: 4 })).toContain('4 scenes')
    expect(taskText('outlineScenes', {})).toMatch(/ids of the bible entries listed in the prompt/)
  })

  it('does not put the alternatives count in a prose prompt', () => {
    expect(taskText('continue', { count: 3 })).toBe(taskText('continue', {}))
    expect(taskText('rewrite', { count: 4, variant: 'tighter' })).toBe(taskText('rewrite', { variant: 'tighter' }))
  })

  it('describes a scene: title, chapter, POV, present, target words, beats', () => {
    const scene: NovelScene = {
      id: 's-1', title: 'Night crossing', file: '01.md', beats: 'Mara bribes the ferryman.', summary: '', summaryOf: '',
      status: 'beats', pov: 'mara-venn', present: ['mara-venn', 'gone'], sticky: [], targetWords: 2500
    }
    const chapter: NovelChapter = { id: 'c-1', title: 'The ferry', act: '', summary: '', summaryOf: '', scenes: [scene] }
    const text = sceneText(scene, chapter, new Map([['mara-venn', entry()]]))
    expect(text).toBe(
      'Chapter: The ferry\nScene: Night crossing\nPOV: Mara Venn\nPresent: Mara Venn, gone\nTarget length: 2500 words\nBeats:\nMara bribes the ferryman.'
    )
    const bare = sceneText({ ...scene, pov: null, present: [], targetWords: null, beats: '  ' }, chapter, new Map())
    expect(bare).toBe('Chapter: The ferry\nScene: Night crossing')
  })

  it('shows a bible entry with its kind, aliases and body', () => {
    const e = entry({ aliases: ['Mara', 'the Salt Witch'], body: 'Tall, scarred.\n\nHates boats.' })
    expect(entryText(e)).toBe('### Mara Venn (character)\nAliases: Mara, the Salt Witch\n\nTall, scarred.\n\nHates boats.')
    expect(entryShortText(e)).toBe('### Mara Venn (character)\nAliases: Mara, the Salt Witch\n\nTall, scarred.')
    expect(entryText(entry({ body: '' }))).toBe('### Mara Venn (character)')
    expect(entryShortText(entry({ body: '\n\nOnly one.\n' }))).toBe('### Mara Venn (character)\n\nOnly one.')
  })

  it('shows a summary under its label', () => {
    expect(summaryText('Chapter 2: The ferry', ' They crossed. ')).toBe('### Chapter 2: The ferry\nThey crossed.')
  })
})

describe('JSON schemas', () => {
  it('are plain objects with required fields and no extra properties', () => {
    for (const schema of [OUTLINE_CHAPTERS_SCHEMA, OUTLINE_SCENES_SCHEMA, SUMMARY_SCHEMA]) {
      expect(schema.type).toBe('object')
      expect(schema.additionalProperties).toBe(false)
      expect(schema.required.length).toBeGreaterThan(0)
      expect(() => JSON.stringify(schema)).not.toThrow()
    }
    expect(SUMMARY_SCHEMA.required).toEqual(['summary'])
    expect(OUTLINE_CHAPTERS_SCHEMA.properties.chapters.items.required).toEqual(['title', 'act', 'summary'])
    expect(OUTLINE_SCENES_SCHEMA.properties.scenes.items.required).toEqual(['title', 'beats', 'pov', 'present'])
    expect(OUTLINE_SCENES_SCHEMA.properties.scenes.items.additionalProperties).toBe(false)
  })
})

describe('structured reply parsers', () => {
  const summary = '{"summary":"They crossed the river."}'
  const chapters = '{"chapters":[{"title":"The ferry","act":"Act I","summary":"Mara leaves."}]}'
  const scenes = '{"scenes":[{"title":"Night","beats":"Bribe.","pov":"mara-venn","present":["mara-venn"]}]}'

  it('parses plain JSON', () => {
    expect(parseSummary(summary)).toEqual({ ok: true, value: { summary: 'They crossed the river.' } })
    expect(parseOutlineChapters(chapters)).toEqual({
      ok: true,
      value: { chapters: [{ title: 'The ferry', act: 'Act I', summary: 'Mara leaves.' }] }
    })
    expect(parseOutlineScenes(scenes)).toEqual({
      ok: true,
      value: { scenes: [{ title: 'Night', beats: 'Bribe.', pov: 'mara-venn', present: ['mara-venn'] }] }
    })
  })

  it('parses JSON in a fence, with or without the json tag, and with leading text', () => {
    expect(parseSummary('```json\n' + summary + '\n```')).toMatchObject({ ok: true })
    expect(parseSummary('```\n' + summary + '\n```')).toMatchObject({ ok: true })
    expect(parseSummary('Here is the summary:\n\n```json\n' + summary + '\n```\nHope that helps.')).toMatchObject({ ok: true })
    expect(parseSummary('Sure! ' + summary)).toMatchObject({ ok: true })
    expect(parseSummary(summary + '\n\nLet me know if you want changes.')).toMatchObject({ ok: true })
    expect(parseOutlineChapters('Result:\n' + chapters)).toMatchObject({ ok: true })
    expect(parseOutlineScenes('```json\r\n' + scenes + '\r\n```')).toMatchObject({ ok: true })
  })

  it('accepts a null or empty pov and turns "" into null', () => {
    const none = parseOutlineScenes('{"scenes":[{"title":"A","beats":"B","pov":null,"present":[]},{"title":"C","beats":"D","pov":"","present":[]}]}')
    expect(none.ok && none.value.scenes.map((s) => s.pov)).toEqual([null, null])
  })

  it('trims the summary', () => {
    expect(parseSummary('{"summary":"  Done.\\n"}')).toEqual({ ok: true, value: { summary: 'Done.' } })
  })

  it('rejects text with no JSON and invalid JSON, with a clear error', () => {
    for (const bad of ['', 'no json here', '{"summary": "unterminated', '{summary: 1}', '```json\n{oops}\n```']) {
      const result = parseSummary(bad)
      expect(result.ok).toBe(false)
      expect(!result.ok && result.error).toBe('The reply is not valid JSON.')
    }
  })

  it('rejects JSON of the wrong shape and names the problem', () => {
    const wrongType = parseSummary('{"summary":4}')
    expect(wrongType.ok).toBe(false)
    expect(!wrongType.ok && wrongType.error).toContain('summary')
    expect(!wrongType.ok && wrongType.error).toContain('wrong shape')
    expect(parseSummary('{"text":"x"}').ok).toBe(false)
    expect(parseSummary('{"summary":"   "}').ok).toBe(false)
    expect(parseSummary('[1,2]').ok).toBe(false)
    expect(parseOutlineChapters('{"chapters":[]}').ok).toBe(false)
    expect(parseOutlineChapters('{"chapters":[{"title":"A"}]}').ok).toBe(false)
    expect(parseOutlineScenes('{"scenes":[{"title":"A","beats":"B","pov":3,"present":[]}]}').ok).toBe(false)
    expect(parseOutlineScenes('{"scenes":[{"title":"A","beats":"B","pov":null}]}').ok).toBe(false)
    expect(parseOutlineScenes(chapters).ok).toBe(false)
  })

  it('ignores extra keys the model adds', () => {
    expect(parseSummary('{"summary":"ok","notes":"x"}')).toEqual({ ok: true, value: { summary: 'ok' } })
  })
})

describe('cleanProse', () => {
  it('returns clean prose unchanged apart from surrounding whitespace', () => {
    expect(cleanProse('  The rain fell.\n\nShe waited.\n')).toBe('The rain fell.\n\nShe waited.')
    expect(cleanProse('')).toBe('')
  })

  it('removes a preamble line', () => {
    expect(cleanProse('Sure, here is the continuation:\n\nThe rain fell.')).toBe('The rain fell.')
    expect(cleanProse("Sure! Here's the next part:\nThe rain fell.")).toBe('The rain fell.')
    expect(cleanProse('Certainly!\n\nThe rain fell.')).toBe('The rain fell.')
    expect(cleanProse('Here is the rewritten passage:\n\nThe rain fell.')).toBe('The rain fell.')
    expect(cleanProse('Of course. Here are three paragraphs:\n\nThe rain fell.')).toBe('The rain fell.')
    expect(cleanProse('Sure, here is the passage:\n\nHere is the thing: it rained.')).toBe('Here is the thing: it rained.')
  })

  it('does not touch prose that merely starts with those words', () => {
    expect(cleanProse('Sure enough, it rained.')).toBe('Sure enough, it rained.')
    expect(cleanProse('Sure enough, it rained.\n\nShe ran.')).toBe('Sure enough, it rained.\n\nShe ran.')
    expect(cleanProse('Certainly not, she said.\nHe left.')).toBe('Certainly not, she said.\nHe left.')
    expect(cleanProse('Sure.\nThe door opened.')).toBe('Sure.\nThe door opened.')
    expect(cleanProse('Here is the thing, she said, and then she left.\nNo more.')).toBe(
      'Here is the thing, she said, and then she left.\nNo more.'
    )
    expect(cleanProse('Of course he came.\nNobody was surprised.')).toBe('Of course he came.\nNobody was surprised.')
  })

  it('removes a wrapping code fence', () => {
    expect(cleanProse('```\nThe rain fell.\n```')).toBe('The rain fell.')
    expect(cleanProse('```markdown\nThe rain fell.\n\nShe waited.\n```')).toBe('The rain fell.\n\nShe waited.')
    expect(cleanProse('Sure, here is the passage:\n```\nA dark night.\n```\n\n(Word count: 3)')).toBe('A dark night.')
  })

  it('removes leading and trailing rules', () => {
    expect(cleanProse('---\nThe rain fell.\n---')).toBe('The rain fell.')
    expect(cleanProse('The rain fell.\n\n---\n')).toBe('The rain fell.')
    // A scene break in the middle is story text.
    expect(cleanProse('One.\n\n---\n\nTwo.')).toBe('One.\n\n---\n\nTwo.')
  })

  it('removes trailing notes', () => {
    expect(cleanProse('The rain fell.\n\n(Word count: 312)')).toBe('The rain fell.')
    expect(cleanProse('The rain fell.\n\n[Word count: 312]')).toBe('The rain fell.')
    expect(cleanProse('The rain fell.\n\nWord count: 312')).toBe('The rain fell.')
    expect(cleanProse('The rain fell.\n\nLet me know if you want it darker!')).toBe('The rain fell.')
    expect(cleanProse('The rain fell.\n\n*Note: I kept the tone.*')).toBe('The rain fell.')
    expect(cleanProse('The rain fell.\n\n---\n\nI hope this helps.')).toBe('The rain fell.')
  })

  it('keeps story text that looks a little like a note', () => {
    expect(cleanProse('She read it twice.\n\nNote: the door is locked.')).toBe('She read it twice.\n\nNote: the door is locked.')
    expect(cleanProse('"Let me know," she said.')).toBe('"Let me know," she said.')
  })

  it('removes one pair of quotes around a whole multi-paragraph reply', () => {
    expect(cleanProse('"The rain fell.\n\nShe waited."')).toBe('The rain fell.\n\nShe waited.')
    expect(cleanProse('“The rain fell.\n\nShe waited.”')).toBe('The rain fell.\n\nShe waited.')
    expect(cleanProse('"The rain fell.\nShe waited."')).toBe('The rain fell.\nShe waited.')
  })

  it('does not strip dialogue that starts and ends with a quote', () => {
    expect(cleanProse('"Go," she said.\n"Now."')).toBe('"Go," she said.\n"Now."')
    expect(cleanProse('"Go now," she said, "before they come."')).toBe('"Go now," she said, "before they come."')
    expect(cleanProse('"We leave at dawn."')).toBe('"We leave at dawn."')
    expect(cleanProse('“Go,” she said. “Now.”')).toBe('“Go,” she said. “Now.”')
    expect(cleanProse('"Go."\n\nHe went.\n\n"Why?"')).toBe('"Go."\n\nHe went.\n\n"Why?"')
  })

  it('peels several layers at once', () => {
    const raw = 'Sure, here is the scene:\n\n```\n---\n"The rain fell.\n\nShe waited."\n---\n```\n\n(Word count: 5)\nLet me know!'
    expect(cleanProse(raw)).toBe('The rain fell.\n\nShe waited.')
  })

  it('keeps dialogue-first prose after a preamble', () => {
    expect(cleanProse('Sure, here is the continuation:\n\n"Go," she said.\n"Now."')).toBe('"Go," she said.\n"Now."')
  })
})

describe('review fixes', () => {
  it('never unquotes a single line or paragraph of speech', () => {
    for (const speech of [
      '"The ferry doesn\'t run after dark. You\'ll have to wait."',
      '"Mr. Venn is waiting."',
      '"I... I don\'t know."',
      '“The ferry doesn\'t run after dark. You\'ll have to wait.”'
    ]) {
      expect(cleanProse(speech)).toBe(speech)
    }
  })

  it('handles CRLF replies', () => {
    expect(cleanProse('Sure, here is the passage:\r\n\r\nThe rain fell.\r\n\r\n(Word count: 3)')).toBe('The rain fell.')
    expect(cleanProse('Certainly!\r\nThe rain fell.\r\nShe waited.')).toBe('The rain fell.\nShe waited.')
  })

  it('keeps a real last line that starts like a sign-off, drops the usual ones', () => {
    const letter = 'She wrote on.\n\nI hope this letter reaches you before winter.'
    expect(cleanProse(letter)).toBe(letter)
    expect(cleanProse('The rain fell.\n\nI hope this helps!')).toBe('The rain fell.')
    expect(cleanProse('The rain fell.\n\nI hope this captures the mood.')).toBe('The rain fell.')
  })

  it('shows entry ids in the list given to the outline prompts', () => {
    const list = entryListText([entry({ aliases: ['Mara'] }), entry({ id: 'oskar', name: 'Oskar', kind: 'place' })])
    expect(list).toBe('- Mara Venn (character, id: mara-venn)\n- Oskar (place, id: oskar)')
    expect(entryListText([])).toBe('')
  })

  it('resolves a reply reference by id, then name or alias', () => {
    const entries = [entry({ aliases: ['the Salt Witch'] }), entry({ id: 'oskar', name: 'Oskar' }), entry({ id: 'x', name: 'oskar-venn' })]
    expect(resolveEntryRef(entries, 'oskar')).toBe('oskar')
    expect(resolveEntryRef(entries, ' mara-venn ')).toBe('mara-venn')
    expect(resolveEntryRef(entries, 'MARA VENN')).toBe('mara-venn')
    expect(resolveEntryRef(entries, 'the salt witch')).toBe('mara-venn')
    expect(resolveEntryRef(entries, 'Nobody')).toBeNull()
    expect(resolveEntryRef(entries, '  ')).toBeNull()
  })

  it('rejects blank titles, summaries and beats and drops blank present ids', () => {
    expect(parseOutlineChapters('{"chapters":[{"title":" ","act":"","summary":"x"}]}').ok).toBe(false)
    expect(parseOutlineChapters('{"chapters":[{"title":"A","act":"","summary":"  "}]}').ok).toBe(false)
    expect(parseOutlineChapters('{"chapters":[{"title":" A ","act":" I ","summary":"x"}]}')).toEqual({
      ok: true,
      value: { chapters: [{ title: 'A', act: 'I', summary: 'x' }] }
    })
    expect(parseOutlineScenes('{"scenes":[{"title":"","beats":"b","pov":null,"present":[]}]}').ok).toBe(false)
    expect(parseOutlineScenes('{"scenes":[{"title":"A","beats":" ","pov":null,"present":[]}]}').ok).toBe(false)
    const ok = parseOutlineScenes('{"scenes":[{"title":"A","beats":"b","pov":null,"present":["mara", "", "  ", "oskar"]}]}')
    expect(ok.ok && ok.value.scenes[0].present).toEqual(['mara', 'oskar'])
  })

  it('asks the schemas for at least one chapter or scene', () => {
    expect(OUTLINE_CHAPTERS_SCHEMA.properties.chapters.minItems).toBe(1)
    expect(OUTLINE_SCENES_SCHEMA.properties.scenes.minItems).toBe(1)
  })

  it('asks rewrite, expand and condense to match the voice, and continue to continue it', () => {
    expect(systemText('continue', '', '')).toContain('Continue in the same voice')
    for (const action of ['rewrite', 'expand', 'condense'] as const) {
      const text = systemText(action, '', '')
      expect(text).toContain('Match the voice')
      expect(text).not.toContain('Continue in the same voice')
    }
  })

  it('gives tone and custom rewrites a usable instruction without a note', () => {
    expect(taskText('rewrite', { variant: 'tone' })).toMatch(/noticeably different tone/)
    expect(taskText('rewrite', { variant: 'tone', note: '  ' })).toMatch(/noticeably different tone/)
    expect(taskText('rewrite', { variant: 'custom' })).toMatch(/to improve it/)
    expect(taskText('rewrite', { variant: 'custom' })).not.toMatch(/note asks/)
  })
})

// ---------------------------------------------------------------------------------------------
// Stage 3
// ---------------------------------------------------------------------------------------------

describe('stage 3 task texts', () => {
  it('has a task for every stage 3 action, and the JSON ones are not prose', () => {
    for (const action of ['checkScene', 'updateState', 'interview'] as NovelAction[]) {
      expect(taskText(action, {}).length).toBeGreaterThan(20)
    }
    expect(isProseAction('checkScene')).toBe(false)
    expect(isProseAction('updateState')).toBe(false)
    expect(systemText('checkScene', '', '')).toContain('JSON')
  })

  it('check text asks for at most 20 findings with the exact quote and lists the types', () => {
    const text = taskText('checkScene', {})
    expect(text).toContain('at most 20')
    expect(text).toContain('exact words')
    for (const type of ['continuity', 'character', 'timeline', 'pov', 'tense', 'other']) expect(text).toContain(type)
    expect(text).toContain('"findings"')
  })

  it('update-state text carries the state with ids and what is already recorded', () => {
    const text = taskText('updateState', {
      stateBeforeText: '[f1a2b3] Mara Venn - location: the lighthouse',
      recordedText: '[f9z9z9] Mara Venn - carries: the brass key'
    })
    expect(text).toContain('"changes"')
    expect(text).toContain('State before this scene (fact ids in brackets):\n[f1a2b3] Mara Venn - location: the lighthouse')
    expect(text).toContain('Already recorded at this scene:\n[f9z9z9] Mara Venn - carries: the brass key')
    expect(text.indexOf('State before')).toBeGreaterThan(text.indexOf('Reply as JSON'))
  })

  it('update-state text says (none) for empty lists, and other actions do not get the lists', () => {
    const text = taskText('updateState', {})
    expect(text).toContain('State before this scene (fact ids in brackets):\n(none)')
    expect(text).toContain('Already recorded at this scene:\n(none)')
    expect(taskText('checkScene', { stateBeforeText: 'x' })).not.toContain('State before this scene (fact')
  })

  it('keeps the author note on the stage 3 actions', () => {
    expect(taskText('checkScene', { note: 'watch the timeline' })).toContain("The author's note: watch the timeline")
  })
})

describe('CHECK_SCHEMA and UPDATE_STATE_SCHEMA', () => {
  it('check: findings may be empty and are capped at 20', () => {
    const findings = CHECK_SCHEMA.properties.findings
    expect(findings.minItems).toBe(0)
    expect(findings.maxItems).toBe(20)
    expect(findings.items.required).toEqual(['type', 'quote', 'problem', 'conflictsWith', 'suggestion'])
    expect(findings.items.properties.type.enum).toEqual(['continuity', 'character', 'timeline', 'pov', 'tense', 'other'])
  })

  it('update state: changes may be empty; add and end shapes', () => {
    const changes = UPDATE_STATE_SCHEMA.properties.changes
    expect(changes.minItems).toBe(0)
    expect(changes.items.anyOf).toHaveLength(2)
    const add = changes.items.anyOf[0] as { required: string[]; properties: { kind: { enum: string[] } } }
    expect(add.required).toEqual(['op', 'entity', 'kind', 'value', 'about'])
    expect(add.properties.kind.enum).toEqual([
      'location', 'knows', 'carries', 'condition', 'relationship', 'other'
    ])
    expect(changes.items.anyOf[1].required).toEqual(['op', 'factId', 'reason'])
  })
})

describe('parseCheckReply', () => {
  const finding = (over: Record<string, unknown> = {}): Record<string, unknown> => ({
    type: 'continuity',
    quote: 'her blue eyes',
    problem: 'Her eyes were green.',
    conflictsWith: 'Mara Venn: green eyes',
    suggestion: 'Say green.',
    ...over
  })

  it('reads findings and trims every value', () => {
    const reply = parseCheckReply(
      JSON.stringify({ findings: [finding({ quote: '  her blue eyes \n', problem: ' wrong ', conflictsWith: ' x ', suggestion: ' y ' })] })
    )
    expect(reply.findings).toEqual([
      { type: 'continuity', quote: 'her blue eyes', problem: 'wrong', conflictsWith: 'x', suggestion: 'y' }
    ])
  })

  it('accepts an empty list, fenced JSON and JSON after some text', () => {
    expect(parseCheckReply('{"findings": []}')).toEqual({ findings: [] })
    expect(parseCheckReply('```json\n{"findings": []}\n```').findings).toEqual([])
    expect(parseCheckReply(`Here you go: ${JSON.stringify({ findings: [finding()] })}`).findings).toHaveLength(1)
  })

  it('drops unknown types, empty quote or problem, and items that are not objects', () => {
    const reply = parseCheckReply(
      JSON.stringify({
        findings: [
          finding({ type: 'grammar' }),
          finding({ quote: '  ' }),
          finding({ problem: '' }),
          finding({ type: undefined }),
          'nonsense',
          null,
          finding({ type: 'POV' }),
          finding({ conflictsWith: undefined, suggestion: 5 })
        ]
      })
    )
    expect(reply.findings.map((f) => f.type)).toEqual(['pov', 'continuity'])
    expect(reply.findings[1]).toMatchObject({ conflictsWith: '', suggestion: '' })
  })

  it('keeps at most 20 findings, the first ones', () => {
    const many = Array.from({ length: 30 }, (_, i) => finding({ quote: `q${i}` }))
    const reply = parseCheckReply(JSON.stringify({ findings: many }))
    expect(reply.findings).toHaveLength(20)
    expect(reply.findings[19].quote).toBe('q19')
  })

  it('throws for text that is not the expected JSON, and the Result variant reports it', () => {
    expect(() => parseCheckReply('not json')).toThrow(/not valid JSON/)
    expect(() => parseCheckReply('{"findings": "none"}')).toThrow(/wrong shape/)
    expect(() => parseCheckReply('{"other": []}')).toThrow()
    expect(parseCheckReplyResult('nope').ok).toBe(false)
    expect(parseCheckReplyResult('{"findings": []}').ok).toBe(true)
  })
})

describe('parseUpdateStateReply', () => {
  it('reads add and end changes', () => {
    const reply = parseUpdateStateReply(
      JSON.stringify({
        changes: [
          { op: 'add', entity: ' Mara Venn ', kind: 'carries', value: ' the brass key ', about: null },
          { op: 'end', factId: ' f1a2b3 ', reason: ' she left ' }
        ]
      })
    )
    expect(reply.changes).toEqual([
      { op: 'add', entity: 'Mara Venn', kind: 'carries', value: 'the brass key', about: null },
      { op: 'end', factId: 'f1a2b3', reason: 'she left' }
    ])
  })

  it('keeps about for a relationship only', () => {
    const reply = parseUpdateStateReply(
      JSON.stringify({
        changes: [
          { op: 'add', entity: 'Mara', kind: 'relationship', value: 'distrusts him', about: ' Oskar Lind ' },
          { op: 'add', entity: 'Mara', kind: 'location', value: 'the pier', about: 'Oskar Lind' },
          { op: 'add', entity: 'Mara', kind: 'relationship', value: 'trusts', about: '' }
        ]
      })
    )
    expect(reply.changes).toEqual([
      { op: 'add', entity: 'Mara', kind: 'relationship', value: 'distrusts him', about: 'Oskar Lind' },
      { op: 'add', entity: 'Mara', kind: 'location', value: 'the pier', about: null },
      { op: 'add', entity: 'Mara', kind: 'relationship', value: 'trusts', about: null }
    ])
  })

  it('drops unknown ops and kinds and empty entity, value or fact id', () => {
    const reply = parseUpdateStateReply(
      JSON.stringify({
        changes: [
          { op: 'delete', factId: 'f1' },
          { op: 'add', entity: 'Mara', kind: 'mood', value: 'sad', about: null },
          { op: 'add', entity: '', kind: 'location', value: 'x', about: null },
          { op: 'add', entity: 'Mara', kind: 'location', value: '  ', about: null },
          { op: 'end', factId: ' ', reason: 'r' },
          42,
          { op: 'add', entity: 'Mara', kind: 'Location', value: 'the pier', about: null }
        ]
      })
    )
    expect(reply.changes).toEqual([{ op: 'add', entity: 'Mara', kind: 'location', value: 'the pier', about: null }])
  })

  it('makes values one line and caps them at 300 characters', () => {
    const reply = parseUpdateStateReply(
      JSON.stringify({
        changes: [
          { op: 'add', entity: 'Mara', kind: 'condition', value: 'tired\nand   cold', about: null },
          { op: 'add', entity: 'Mara', kind: 'knows', value: 'x'.repeat(500), about: null },
          { op: 'end', factId: 'f1', reason: 'y'.repeat(500) }
        ]
      })
    )
    expect(reply.changes[0]).toMatchObject({ value: 'tired and cold' })
    expect((reply.changes[1] as { value: string }).value).toHaveLength(300)
    expect((reply.changes[2] as { reason: string }).reason).toHaveLength(300)
  })

  it('accepts an empty list and throws for bad replies', () => {
    expect(parseUpdateStateReply('{"changes": []}')).toEqual({ changes: [] })
    expect(parseUpdateStateReply('```\n{"changes": []}\n```').changes).toEqual([])
    expect(() => parseUpdateStateReply('???')).toThrow()
    expect(() => parseUpdateStateReply('{"changes": 3}')).toThrow()
    expect(parseUpdateStateReplyResult('{"changes": 3}').ok).toBe(false)
  })
})

describe('interviewSystem', () => {
  const input = {
    name: 'Mara Venn',
    body: 'Tall and quiet.\n\n## Voice\n- dry',
    stateLines: '- location: the lighthouse\n- carries: the brass key',
    summaries: ['Scene: Night crossing\nMara bribed the ferryman.', 'Chapter 1: She arrived.']
  }

  it('tells the model to answer in character and to know only what the character knows', () => {
    const text = interviewSystem(input)
    expect(text).toContain('You are Mara Venn. Answer in character, in your own voice.')
    expect(text).toContain('You know only what Mara Venn knows by this point in the story')
    expect(text).toContain('if asked about later events, you do not know them')
  })

  it("labels the author's notes, then adds state lines and summaries in that order", () => {
    const text = interviewSystem(input)
    const notes = text.indexOf("The author's notes about you (they may mention things you have not lived yet; do not reveal those):")
    expect(notes).toBeGreaterThan(0)
    expect(text.indexOf('Tall and quiet.')).toBeGreaterThan(notes)
    const state = text.indexOf('- location: the lighthouse')
    expect(state).toBeGreaterThan(text.indexOf('- dry'))
    expect(text.indexOf('Mara bribed the ferryman.')).toBeGreaterThan(state)
    expect(text.indexOf('Chapter 1: She arrived.')).toBeGreaterThan(text.indexOf('Mara bribed'))
  })

  it('leaves out sections that are empty', () => {
    const text = interviewSystem({ name: 'Mara', body: '  ', stateLines: '', summaries: ['', '  '] })
    expect(text).not.toContain("author's notes")
    expect(text).not.toContain('What is true of you')
    expect(text).not.toContain('What has happened')
    expect(text.startsWith('You are Mara.')).toBe(true)
  })
})

describe('review fixes', () => {
  it('strips brackets around a fact id', () => {
    const reply = parseUpdateStateReply(JSON.stringify({ changes: [{ op: 'end', factId: ' [f1a2b3] ', reason: 'r' }] }))
    expect(reply.changes).toEqual([{ op: 'end', factId: 'f1a2b3', reason: 'r' }])
  })
})
