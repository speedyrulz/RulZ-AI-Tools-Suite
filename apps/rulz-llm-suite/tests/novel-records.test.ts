// Stage 4: the pure parts of recording and export (src/main/novel/records.ts): line validation, the join, the rows.
import { describe, expect, it } from 'vitest'
import type { RecordAlt, RecordFinalLine, RecordLine, RecordOutcomeLine, RecordRequestLine, RecordsExportOptions } from '@shared/novel'
import { MAX_RECORD_BYTES, RecordJoin, type RecordMeta, exportRequest, localDate, localStamp, parseRecordBytes, validateRecordLine } from '../src/main/novel/records'

const ID = 'aaaaaaaaaaaaaaaa'
const AT = '2026-11-22T10:00:00.000Z'
const OFF: RecordsExportOptions = { editPairs: false }
const ON: RecordsExportOptions = { editPairs: true }
const GEN = 'The rain came down over the harbour in long grey sheets.'

const alt = (index: number, generated: string, finish: RecordAlt['finish'] = 'stop'): RecordAlt => ({ index, seed: 100 + index, generated, finish })

function req(over: Partial<RecordRequestLine> = {}): RecordRequestLine {
  return {
    format: 1,
    type: 'request',
    id: ID,
    at: AT,
    app: '0.23.0',
    action: 'continue',
    sceneId: 's-1',
    chapterId: 'c-1',
    model: { id: 'm1', name: 'Model One' },
    sampling: { temperature: 0.8 } as RecordRequestLine['sampling'],
    adapters: [{ name: 'style', scale: 0.7 }],
    messages: [
      { role: 'system', content: 'You write fiction.' },
      { role: 'user', content: 'Continue the scene.' }
    ],
    alts: [alt(0, GEN)],
    ...over
  }
}

const outcome = (over: Partial<RecordOutcomeLine> = {}): RecordOutcomeLine => ({ format: 1, type: 'outcome', id: ID, at: AT, kind: 'accepted', alt: 0, region: { start: 10, end: 60 }, replaced: '', ...over })
const final = (text: string | null, reason: RecordFinalLine['reason'] = 'idle'): RecordFinalLine => ({ format: 1, type: 'final', id: ID, at: AT, text, reason })

function joined(...lines: RecordLine[]): RecordJoin {
  const j = new RecordJoin()
  for (const l of lines) j.add(l)
  return j
}
const join = (...lines: RecordLine[]): RecordMeta | undefined => joined(...lines).get(ID)
const viewed = (...alts: number[]): RecordLine => ({ format: 1, type: 'viewed', id: ID, at: AT, alts })

describe('validateRecordLine', () => {
  it('accepts each of the four line types and keeps only the known fields', () => {
    expect(validateRecordLine(req())).toEqual(req())
    expect(validateRecordLine(viewed(0, 2))).toEqual(viewed(0, 2))
    expect(validateRecordLine(outcome())).toEqual(outcome())
    expect(validateRecordLine(outcome({ kind: 'rejected', alt: null, region: null }))).toBeTruthy()
    expect(validateRecordLine(final('x', 'outside'))).toEqual(final('x', 'outside'))
    expect(validateRecordLine(final(null, 'reloaded'))).toEqual(final(null, 'reloaded'))
    expect(validateRecordLine({ ...req(), extra: 1 })).toEqual(req())
    expect(validateRecordLine(req({ variant: 'tighter' }))).toMatchObject({ variant: 'tighter' })
  })

  it('rejects an unknown type, a bad id, a bad format or time', () => {
    expect(validateRecordLine({ ...req(), type: 'card' })).toBeNull()
    expect(validateRecordLine({ ...req(), type: undefined })).toBeNull()
    for (const id of ['', 'AAAAAAAAAAAAAAAA', 'aaaaaaaaaaaaaaa', 'aaaaaaaaaaaaaaaaa', 'gggggggggggggggg', 5]) expect(validateRecordLine({ ...req(), id })).toBeNull()
    expect(validateRecordLine({ ...req(), format: 2 })).toBeNull()
    expect(validateRecordLine({ ...req(), at: 'yesterday' })).toBeNull()
    expect(validateRecordLine({ ...req(), at: 5 })).toBeNull()
    for (const bad of [null, 'text', 5, [], undefined]) expect(validateRecordLine(bad)).toBeNull()
  })

  it('rejects wrong field types per type', () => {
    expect(validateRecordLine({ ...req(), action: 'checkScene' })).toBeNull()
    expect(validateRecordLine({ ...req(), action: 'interview' })).toBeNull()
    expect(validateRecordLine({ ...req(), action: 'nope' })).toBeNull()
    expect(validateRecordLine({ ...req(), sceneId: 3 })).toBeNull()
    expect(validateRecordLine({ ...req(), sceneId: undefined })).toBeNull()
    expect(validateRecordLine({ ...req(), model: { id: 'm' } })).toBeNull()
    expect(validateRecordLine({ ...req(), sampling: 'hot' })).toBeNull()
    expect(validateRecordLine({ ...req(), adapters: [{ name: 'a', scale: 'x' }] })).toBeNull()
    expect(validateRecordLine({ ...req(), adapters: [{ name: 'a', scale: Infinity }] })).toBeNull()
    expect(validateRecordLine({ ...req(), messages: [] })).toBeNull()
    expect(validateRecordLine({ ...req(), messages: [{ role: 'tool', content: 'x' }] })).toBeNull()
    expect(validateRecordLine({ ...req(), messages: [{ role: 'user', content: 5 }] })).toBeNull()
    expect(validateRecordLine({ ...req(), alts: [] })).toBeNull()
    expect(validateRecordLine({ ...req(), alts: [alt(0, 'a'), alt(0, 'b')] })).toBeNull()
    expect(validateRecordLine({ ...req(), alts: [{ ...alt(0, 'a'), finish: 'done' }] })).toBeNull()
    expect(validateRecordLine({ ...req(), alts: [{ ...alt(-1, 'a') }] })).toBeNull()
    expect(validateRecordLine({ ...req(), alts: [{ ...alt(0, 'a'), generated: 5 }] })).toBeNull()
    expect(validateRecordLine({ ...viewed(0), alts: [0.5] })).toBeNull()
    expect(validateRecordLine({ ...viewed(0), alts: 'x' })).toBeNull()
    expect(validateRecordLine({ ...outcome(), kind: 'merged' })).toBeNull()
    expect(validateRecordLine({ ...outcome(), alt: 'one' })).toBeNull()
    expect(validateRecordLine({ ...outcome(), region: { start: 5, end: 2 } })).toBeNull()
    expect(validateRecordLine({ ...outcome(), region: { start: 'a', end: 2 } })).toBeNull()
    expect(validateRecordLine({ ...outcome(), replaced: 5 })).toBeNull()
    expect(validateRecordLine({ ...final('x'), reason: 'later' })).toBeNull()
    expect(validateRecordLine({ ...final('x'), text: 5 })).toBeNull()
    expect(validateRecordLine({ ...final('x'), text: undefined })).toBeNull()
  })

  it('parseRecordBytes reports why a line is dropped', () => {
    expect(parseRecordBytes(Buffer.from(JSON.stringify(req()), 'utf8'))).toMatchObject({ ok: true })
    expect(parseRecordBytes(Buffer.from('  \r'))).toEqual({ ok: 'blank' })
    expect(parseRecordBytes(Buffer.from('{"format":1,"ty'))).toEqual({ ok: false, reason: 'badJson' })
    expect(parseRecordBytes(Buffer.from('{"format":1,"type":"x"}'))).toEqual({ ok: false, reason: 'badShape' })
    expect(parseRecordBytes(Buffer.from([0x7b, 0xff, 0xfe, 0x7d]))).toEqual({ ok: false, reason: 'badUtf8' })
    expect(parseRecordBytes(null)).toEqual({ ok: false, reason: 'tooLong' })
  })

  it('has an 8 MB line limit', () => {
    expect(MAX_RECORD_BYTES).toBe(8 * 1024 * 1024)
  })
})

describe('dates', () => {
  it('names days and exports in local time', () => {
    const d = new Date(2026, 10, 2, 3, 4, 5)
    expect(localDate(d)).toBe('2026-11-02')
    expect(localStamp(d)).toBe('20261102-030405')
  })
})

describe('RecordJoin', () => {
  it('keeps the first outcome, merges viewed and keeps the last final', () => {
    const j = joined(outcome({ kind: 'accepted' }), outcome({ kind: 'rejected' }), viewed(0), viewed(2, 1), final('one'), final('two'))
    const m = j.get(ID)!
    expect(m.outcome?.kind).toBe('accepted')
    expect([...m.viewed].sort()).toEqual([0, 1, 2])
    expect(m.final?.text).toBe('two')
    expect(j.ids()).toEqual([ID])
  })

  it('ignores request lines', () => {
    expect(joined(req()).size).toBe(0)
  })
})

describe('exportRequest: train rows', () => {
  it('accepted with a final: the final text, meta.final true', () => {
    const edited = `${GEN} She turned from the window.`
    const r = exportRequest(req(), join(outcome(), final(edited)), OFF)
    expect(r.drop).toBeNull()
    expect(r.train?.messages.at(-1)).toEqual({ role: 'assistant', content: edited })
    expect(r.train?.messages).toHaveLength(3)
    expect(r.train?.meta).toEqual({ id: ID, action: 'continue', at: AT, model: { id: 'm1', name: 'Model One' }, kind: 'accepted', final: true })
  })

  it('accepted without a final: the generated text, meta.final false', () => {
    const r = exportRequest(req(), join(outcome()), OFF)
    expect(r.train?.messages.at(-1)?.content).toBe(GEN)
    expect(r.train?.meta.final).toBe(false)
  })

  it('accepted with a final whose text is null (outside / reloaded): the generated text, final false', () => {
    const r = exportRequest(req(), join(outcome(), final(null, 'outside')), OFF)
    expect(r.train?.messages.at(-1)?.content).toBe(GEN)
    expect(r.train?.meta.final).toBe(false)
  })

  it('mixed with a final is exported with kind mixed; mixed without a final is dropped as noFinal', () => {
    const mixed = outcome({ kind: 'mixed' })
    const text = `${GEN} Only some of it stayed.`
    expect(exportRequest(req(), join(mixed, final(text)), OFF).train?.meta).toMatchObject({ kind: 'mixed', final: true })
    expect(exportRequest(req(), join(mixed), OFF)).toEqual({ train: null, pairs: [], drop: 'noFinal' })
    expect(exportRequest(req(), join(mixed, final(null, 'reloaded')), OFF).drop).toBe('noFinal')
  })

  it('rejected, abandoned and a missing outcome give no rows', () => {
    expect(exportRequest(req(), join(outcome({ kind: 'rejected', alt: null, region: null })), OFF)).toEqual({ train: null, pairs: [], drop: 'rejected' })
    expect(exportRequest(req(), join(outcome({ kind: 'abandoned', alt: null, region: null })), OFF).drop).toBe('abandoned')
    expect(exportRequest(req(), join(), OFF).drop).toBe('noOutcome')
    expect(exportRequest(req(), undefined, OFF).drop).toBe('noOutcome')
    expect(exportRequest(req(), join(outcome({ kind: 'edited' })), OFF).drop).toBe('badOutcome')
    expect(exportRequest(req(), join(outcome({ alt: 5 })), OFF).drop).toBe('badOutcome')
  })

  it('a reverted final (equal to the replaced text) counts as rejected, with no pairs either', () => {
    const r = req({ action: 'rewrite', alts: [alt(0, GEN), alt(1, 'A quite different way of saying it, longer than twenty.')] })
    const sel = 'The rain fell on the harbour.'
    const j = join(outcome({ replaced: sel }), viewed(0, 1), final(sel))
    expect(exportRequest(r, j, ON)).toEqual({ train: null, pairs: [], drop: 'rejected' })
    // continue: the generated text was deleted again, so the final is ''
    expect(exportRequest(req(), join(outcome(), final('')), OFF).drop).toBe('rejected')
    // a final that merely differs from the replaced text is not reverted
    expect(exportRequest(r, join(outcome({ replaced: sel }), final(`${sel} And more of it, long enough.`)), OFF).train).toBeTruthy()
  })

  it('drops assistant text shorter than 20 characters after trimming', () => {
    const short = req({ alts: [alt(0, '   Too short.   ')] })
    expect(exportRequest(short, join(outcome()), OFF)).toEqual({ train: null, pairs: [], drop: 'tooShort' })
    expect(exportRequest(req(), join(outcome(), final('  short final  ')), OFF).drop).toBe('tooShort')
    const edge = req({ alts: [alt(0, `${'a'.repeat(19)}  `)] })
    expect(exportRequest(edge, join(outcome()), OFF).drop).toBe('tooShort')
    const ok = req({ alts: [alt(0, 'a'.repeat(20))] })
    expect(exportRequest(ok, join(outcome()), OFF).train).toBeTruthy()
  })
})

describe('exportRequest: pairs', () => {
  const second = 'The harbour lay silent while the rain kept coming down.'
  const three = req({ alts: [alt(0, GEN), alt(1, second), alt(2, 'A cut-off third alternative that ran on and on', 'length')] })

  it('pairs the accepted alt with each other alt that was viewed, finished with stop and differs', () => {
    const r = exportRequest(three, join(outcome(), viewed(0, 1, 2)), OFF)
    expect(r.pairs).toEqual([{ prompt: three.messages, chosen: GEN, rejected: second, meta: { id: ID, source: 'alts' } }])
  })

  it('never pairs a length alt, an unviewed alt, an error or stopped alt, or an alt equal after trimming', () => {
    const j = join(outcome(), viewed(0, 2))
    expect(exportRequest(three, j, OFF).pairs).toEqual([])
    const mixedAlts = req({ alts: [alt(0, GEN), alt(1, 'x'.repeat(30), 'error'), alt(2, 'y'.repeat(30), 'stopped'), alt(3, `  ${GEN} \n`)] })
    expect(exportRequest(mixedAlts, join(outcome(), viewed(0, 1, 2, 3)), OFF).pairs).toEqual([])
  })

  it('a cut-off accepted alt is in no pair, neither alt pairs nor edit pairs; its train row stays', () => {
    const cut = req({ alts: [alt(0, GEN, 'length'), alt(1, second)] })
    const edited = 'The rain came down over the harbour, grey and slow, without end.'
    const r = exportRequest(cut, join(outcome(), viewed(0, 1), final(edited)), ON)
    expect(r.pairs).toEqual([])
    expect(r.train?.messages.at(-1)?.content).toBe(edited)
  })

  it('an emptied passage (too short) gives no train row and no pairs at all', () => {
    const r = exportRequest(three, join(outcome({ replaced: 'Old text that was here.' }), viewed(0, 1), final('')), ON)
    expect(r.train).toBeNull()
    expect(r.drop).toBe('tooShort')
    expect(r.pairs.filter((x) => x.meta.source === 'edit')).toEqual([])
    expect(r.pairs.every((x) => x.chosen.trim().length >= 20 && x.rejected.trim().length >= 20)).toBe(true)
  })

  it('makes pairs only from accepted outcomes', () => {
    const j = join(outcome({ kind: 'mixed' }), viewed(0, 1), final(`${GEN} and edited`))
    expect(exportRequest(three, j, OFF).pairs).toEqual([])
  })

  it('edit pairs: off by default, on with the option, chosen is the final', () => {
    const edited = 'The rain came down over the harbour, grey and slow, without end.'
    const j = join(outcome(), final(edited))
    expect(exportRequest(req(), j, OFF).pairs).toEqual([])
    const r = exportRequest(req(), j, ON)
    expect(r.pairs).toEqual([{ prompt: req().messages, chosen: edited, rejected: GEN, meta: { id: ID, source: 'edit' } }])
    expect(r.train?.messages.at(-1)?.content).toBe(edited)
  })

  it('edit pairs skip an append-only change, an unchanged final, a missing final and a reverted final', () => {
    expect(exportRequest(req(), join(outcome(), final(`${GEN} She turned away.`)), ON).pairs).toEqual([])
    expect(exportRequest(req(), join(outcome(), final(`${GEN}  \n`)), ON).pairs).toEqual([])
    expect(exportRequest(req(), join(outcome(), final(GEN)), ON).pairs).toEqual([])
    expect(exportRequest(req(), join(outcome()), ON).pairs).toEqual([])
    expect(exportRequest(req(), join(outcome(), final(null, 'outside')), ON).pairs).toEqual([])
    // text added at the front is a real edit
    expect(exportRequest(req(), join(outcome(), final(`Before. ${GEN}`)), ON).pairs).toHaveLength(1)
  })
})
