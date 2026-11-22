import { describe, expect, it } from 'vitest'
import { appendVoiceNote, shortWithVoice, voiceSection } from '../src/shared/novelVoice'
import { parseEntry } from '../src/shared/frontMatter'

const BOM = String.fromCharCode(0xfeff)
const FRONT = '---\nname: Mara Venn\nkind: character\naliases: [Mara]\n---\n'

describe('voiceSection', () => {
  it('returns the content of a level 2 or 3 Voice section, any case', () => {
    expect(voiceSection('Intro.\n\n## Voice\n\n- dry\n- short sentences\n')).toBe('- dry\n- short sentences')
    expect(voiceSection('Intro.\n\n### VOICE\nplain\n')).toBe('plain')
    expect(voiceSection('## voice ##\nx')).toBe('x')
  })

  it('is empty when there is no voice section', () => {
    expect(voiceSection('')).toBe('')
    expect(voiceSection('Just a body.\n\n## History\nold')).toBe('')
    expect(voiceSection('# Voice\nlevel one is not a voice section')).toBe('')
    expect(voiceSection('#### Voice\nlevel four is not either')).toBe('')
    expect(voiceSection('## Voices\nnot the word')).toBe('')
  })

  it('ends at the next heading of the same or a higher level but not at a deeper one', () => {
    const body = '## Voice\n- a\n### Deeper\n- b\n## Next\n- c\n'
    expect(voiceSection(body)).toBe('- a\n### Deeper\n- b')
    expect(voiceSection('### Voice\n- a\n## Higher\n- b')).toBe('- a')
    expect(voiceSection('### Voice\n- a\n### Same\n- b')).toBe('- a')
    expect(voiceSection('## Voice\n- a\n# Top\n- b')).toBe('- a')
  })

  it('reads CRLF text', () => {
    expect(voiceSection('Intro.\r\n\r\n## Voice\r\n- a\r\n- b\r\n\r\n## Next\r\nx\r\n')).toBe('- a\n- b')
  })

  it('does not treat a # without a space as a heading', () => {
    expect(voiceSection('## Voice\n- a\n#hashtag\n- b')).toBe('- a\n#hashtag\n- b')
  })
})

describe('appendVoiceNote', () => {
  it('creates the section at the end of an entry without one, keeping the front matter byte for byte', () => {
    const file = `${FRONT}Tall, quiet.\n`
    const out = appendVoiceNote(file, 'Never swears.')
    expect(out).toBe(`${FRONT}Tall, quiet.\n\n## Voice\n\n- Never swears.\n`)
    expect(out.startsWith(FRONT)).toBe(true)
  })

  it('creates the section for an empty body and for a file without front matter', () => {
    expect(appendVoiceNote(FRONT, 'x')).toBe(`${FRONT}## Voice\n\n- x\n`)
    expect(appendVoiceNote('Plain body.', 'x')).toBe('Plain body.\n\n## Voice\n\n- x\n')
    expect(appendVoiceNote('', 'x')).toBe('## Voice\n\n- x\n')
  })

  it('adds the missing line break after a closing --- at the end of the file', () => {
    expect(appendVoiceNote('---\nname: A\n---', 'x')).toBe('---\nname: A\n---\n## Voice\n\n- x\n')
  })

  it('appends to the end of an existing section', () => {
    const file = `${FRONT}Body.\n\n## Voice\n- dry\n- brief\n`
    expect(appendVoiceNote(file, 'asks questions')).toBe(`${FRONT}Body.\n\n## Voice\n- dry\n- brief\n- asks questions\n`)
  })

  it('appends to a section in the middle of the body, before the blank line and the next heading', () => {
    const file = `${FRONT}Intro.\n\n## Voice\n- dry\n\n## History\nOld things.\n`
    expect(appendVoiceNote(file, 'brief')).toBe(`${FRONT}Intro.\n\n## Voice\n- dry\n- brief\n\n## History\nOld things.\n`)
    // Same for a level 3 heading and for an empty section.
    expect(appendVoiceNote('### Voice\n\n## Next\nx', 'a')).toBe('### Voice\n- a\n\n## Next\nx')
  })

  it('keeps a deeper heading inside the section and puts the note after its content', () => {
    const file = '## Voice\n- a\n### Detail\n- b\n## Next\n'
    expect(appendVoiceNote(file, 'c')).toBe('## Voice\n- a\n### Detail\n- b\n- c\n## Next\n')
  })

  it('handles a voice section that ends the file without a line break', () => {
    expect(appendVoiceNote('Body.\n\n## Voice\n- a', 'b')).toBe('Body.\n\n## Voice\n- a\n- b\n')
  })

  it('uses CRLF for a CRLF file and changes no other byte', () => {
    const front = '---\r\nname: Mara\r\n---\r\n'
    const file = `${front}Body line.\r\n\r\n## Voice\r\n- dry\r\n\r\n## Other\r\nx\r\n`
    const out = appendVoiceNote(file, 'brief')
    expect(out).toBe(`${front}Body line.\r\n\r\n## Voice\r\n- dry\r\n- brief\r\n\r\n## Other\r\nx\r\n`)
    expect(out.includes('\n')).toBe(true)
    expect(out.replace(/\r\n/g, '').includes('\n')).toBe(false)
    const created = appendVoiceNote(`${front}Body line.\r\n`, 'brief')
    expect(created).toBe(`${front}Body line.\r\n\r\n## Voice\r\n\r\n- brief\r\n`)
    expect(created.replace(/\r\n/g, '').includes('\n')).toBe(false)
  })

  it('keeps a BOM in front of the front matter', () => {
    const out = appendVoiceNote(BOM + '---\nname: A\n---\nBody.\n', 'x')
    expect(out.charCodeAt(0)).toBe(0xfeff)
    expect(out).toBe(BOM + '---\nname: A\n---\nBody.\n\n## Voice\n\n- x\n')
  })

  it('collapses the note to one line, strips leading # and caps it at 500 characters', () => {
    expect(appendVoiceNote('B.\n', '  line one\n\nline   two\t three  ')).toContain('- line one line two three\n')
    expect(appendVoiceNote('B.\n', '## Heading-like note')).toContain('\n- Heading-like note\n')
    expect(appendVoiceNote('B.\n', '###not a heading')).toContain('\n- not a heading\n')
    const long = appendVoiceNote('B.\n', 'x'.repeat(900))
    expect(long).toContain(`- ${'x'.repeat(500)}\n`)
    expect(long).not.toContain('x'.repeat(501))
  })

  it('returns the text unchanged for an empty note', () => {
    expect(appendVoiceNote(`${FRONT}Body.\n`, '   \n ')).toBe(`${FRONT}Body.\n`)
    expect(appendVoiceNote(`${FRONT}Body.\n`, '##')).toBe(`${FRONT}Body.\n`)
  })

  it('treats a file with an unclosed front matter as all body, like parseEntry does', () => {
    const file = '---\nname: A\nno close\n'
    expect(appendVoiceNote(file, 'x')).toBe('---\nname: A\nno close\n\n## Voice\n\n- x\n')
  })

  it('round-trips through parseEntry: front matter fields and voice section', () => {
    const out = appendVoiceNote(appendVoiceNote(`${FRONT}Tall.\n`, 'dry'), 'brief')
    const entry = parseEntry('mara-venn', out)
    expect(entry.name).toBe('Mara Venn')
    expect(entry.aliases).toEqual(['Mara'])
    expect(voiceSection(entry.body)).toBe('- dry\n- brief')
  })
})

describe('shortWithVoice', () => {
  it('keeps the first paragraph and the voice section', () => {
    const body = 'Tall and quiet.\n\nLong history here.\n\n## Voice\n- dry\n- brief\n\n## Secrets\nhidden\n'
    expect(shortWithVoice(body)).toBe('Tall and quiet.\n\n## Voice\n- dry\n- brief')
  })

  it('is the first paragraph when there is no voice section', () => {
    expect(shortWithVoice('One.\n\nTwo.')).toBe('One.')
    expect(shortWithVoice('')).toBe('')
  })

  it('takes the first paragraph from outside the voice section when the body starts with it', () => {
    expect(shortWithVoice('## Voice\n- dry\n\n## Looks\nHe is tall.\n')).toBe('## Looks\nHe is tall.\n\n## Voice\n- dry')
    expect(shortWithVoice('## Voice\n- dry\n')).toBe('## Voice\n- dry')
  })

  it('caps the result at maxChars and keeps most of the voice section', () => {
    const body = `${'a'.repeat(2000)}\n\n## Voice\n${Array.from({ length: 60 }, (_, i) => `- note ${i}`).join('\n')}`
    const out = shortWithVoice(body, 400)
    expect(out.length).toBeLessThanOrEqual(400)
    expect(out).toContain('## Voice\n- note 0')
    expect(out).toContain('…')
    expect(shortWithVoice('b'.repeat(3000)).length).toBeLessThanOrEqual(1200)
  })

  it('reads a CRLF body', () => {
    expect(shortWithVoice('First.\r\n\r\nSecond.\r\n\r\n## Voice\r\n- dry\r\n')).toBe('First.\n\n## Voice\n- dry')
  })
})

describe('review fixes', () => {
  it('shortWithVoice keeps the voice heading and part of a long first note', () => {
    const out = shortWithVoice(`Tall.\n\n## Voice\n- ${'n'.repeat(300)}`, 400)
    expect(out.length).toBeLessThanOrEqual(400)
    expect(out).toContain('## Voice\n- nnn')
    expect(out).toContain('…')
  })

  it('appendVoiceNote with a BOM and no front matter', () => {
    const withVoice = appendVoiceNote(`${BOM}## Voice\n- a\n`, 'b')
    expect(withVoice).toBe(`${BOM}## Voice\n- a\n- b\n`)
    expect(appendVoiceNote(`${BOM}`, 'b')).toBe(`${BOM}## Voice\n\n- b\n`)
    expect(appendVoiceNote(`${BOM}Body.\n`, 'b')).toBe(`${BOM}Body.\n\n## Voice\n\n- b\n`)
    const unclosed = appendVoiceNote(`${BOM}---\nname: A\n\n## Voice\n- a\n`, 'b')
    expect(unclosed).toBe(`${BOM}---\nname: A\n\n## Voice\n- a\n- b\n`)
  })
})
