import { describe, expect, it } from 'vitest'
import type { NovelTextFile } from '../src/shared/novel'
import {
  afterSave,
  draftKey,
  findDraft,
  draftToOffer,
  edit,
  isDirty,
  keepMine,
  loadBuffer,
  onChanged,
  saveFailed,
  saveRequest,
  saving,
  takeDisk,
  withSnapshot,
  type Buffer
} from '../src/lib/autosave'

const FOLDER = 'C:\\Novels\\Ferry'
const PATH = 'manuscript/01-ferry.md'

function file(text: string, hash = 'h1', extra: Partial<NovelTextFile> = {}): NovelTextFile {
  return { path: PATH, text, hash, meta: { bom: false, eol: 'lf' }, readOnly: false, journal: null, ...extra }
}
const fresh = (text = 'The ferry left at dusk.', hash = 'h1'): Buffer => loadBuffer(FOLDER, file(text, hash))

describe('buffer and save request', () => {
  it('is clean after loading and dirty after an edit', () => {
    const b = fresh()
    expect(isDirty(b)).toBe(false)
    expect(saveRequest(b)).toBeNull()
    const e = edit(b, 'The ferry left at dawn.')
    expect(isDirty(e)).toBe(true)
  })

  it('saves with the base hash and never forces', () => {
    const req = saveRequest(edit(fresh(), 'New text.'))
    expect(req).toEqual({ folder: FOLDER, path: PATH, text: 'New text.', baseHash: 'h1' })
    expect(req).not.toHaveProperty('force')
  })

  it('does not save while a save is running, in conflict or read-only', () => {
    const dirty = edit(fresh(), 'x')
    expect(saveRequest(saving(dirty))).toBeNull()
    expect(saveRequest({ ...dirty, conflict: { diskHash: 'h2', diskText: 'y' } })).toBeNull()
    const ro = edit(loadBuffer(FOLDER, file('a', 'h1', { readOnly: true })), 'b')
    expect(ro.text).toBe('a')
    expect(saveRequest(ro)).toBeNull()
  })
})

describe('save results', () => {
  it('takes the new hash and marks the sent text as saved', () => {
    const b = saving(edit(fresh(), 'Sent text.'))
    const after = afterSave(b, 'Sent text.', { ok: true, hash: 'h2' })
    expect(after).toMatchObject({ baseHash: 'h2', savedText: 'Sent text.', saving: null, error: null })
    expect(isDirty(after)).toBe(false)
  })

  it('stays dirty when more was typed during the save, and the next save uses the new hash', () => {
    const typing = edit(saving(edit(fresh(), 'Sent.')), 'Sent. And more.')
    const after = afterSave(typing, 'Sent.', { ok: true, hash: 'h2' })
    expect(isDirty(after)).toBe(true)
    expect(saveRequest(after)).toMatchObject({ text: 'Sent. And more.', baseHash: 'h2' })
  })

  it('keeps the text and shows both versions on a conflict', () => {
    const b = saving(edit(fresh(), 'Mine.'))
    const after = afterSave(b, 'Mine.', { ok: false, reason: 'conflict', diskHash: 'h9', diskText: 'Theirs.' })
    expect(after.text).toBe('Mine.')
    expect(after.conflict).toEqual({ diskHash: 'h9', diskText: 'Theirs.' })
    expect(after.baseHash).toBe('h1')
    expect(saveRequest(after)).toBeNull()
  })

  it('keeps unsaved text with a message when the save is refused or fails', () => {
    const b = saving(edit(fresh(), 'Mine.'))
    expect(afterSave(b, 'Mine.', { ok: false, reason: 'error', error: 'Disk full' })).toMatchObject({ error: 'Disk full', saving: null })
    expect(afterSave(b, 'Mine.', { ok: false, reason: 'blocked' }).error).toMatch(/read-only/)
    expect(afterSave(b, 'Mine.', { ok: false, reason: 'readOnly' })).toMatchObject({ readOnly: true })
    const lost = saveFailed(b, 'Lost the connection to jeff-ai.')
    expect(lost).toMatchObject({ error: 'Lost the connection to jeff-ai.', saving: null })
    expect(isDirty(lost)).toBe(true)
    expect(saveRequest(lost)).not.toBeNull()
  })
})

describe('conflict choices', () => {
  const conflicted = (): Buffer => afterSave(saving(edit(fresh(), 'Mine.')), 'Mine.', { ok: false, reason: 'conflict', diskHash: 'h9', diskText: 'Theirs.' })

  it('"keep mine" saves again on the disk version’s hash, without forcing', () => {
    const b = keepMine(conflicted())
    expect(b.conflict).toBeNull()
    expect(saveRequest(b)).toEqual({ folder: FOLDER, path: PATH, text: 'Mine.', baseHash: 'h9', snapshot: true })
  })

  it('"keep mine" still writes when the text was undone back to the old saved text, and the snapshot flag is spent by a successful save', () => {
    const undone = edit(keepMine(conflicted()), fresh().savedText)
    expect(isDirty(undone)).toBe(true)
    expect(saveRequest(undone)).toMatchObject({ text: 'The ferry left at dusk.', baseHash: 'h9', snapshot: true })
    const done = afterSave(saving(undone), undone.text, { ok: true, hash: 'h10' })
    expect(done.snapshotNext).toBe(false)
  })

  it('can ask for a snapshot of the text an accepted result replaces', () => {
    expect(saveRequest(withSnapshot(edit(fresh(), 'AI text.')))).toMatchObject({ snapshot: true })
    expect(saveRequest(edit(fresh(), 'Typed.'))).not.toHaveProperty('snapshot')
  })

  it('"use the computer’s text" takes the disk version and is clean', () => {
    const b = takeDisk(conflicted())
    expect(b).toMatchObject({ text: 'Theirs.', savedText: 'Theirs.', baseHash: 'h9', conflict: null })
    expect(isDirty(b)).toBe(false)
  })

  it('does nothing without a conflict', () => {
    const b = edit(fresh(), 'x')
    expect(keepMine(b)).toBe(b)
    expect(takeDisk(b)).toBe(b)
  })
})

describe('changes on the computer', () => {
  const text = (hash: string, path = PATH, folder = FOLDER) => ({ kind: 'text' as const, folder, path, hash })

  it('ignores our own save coming back, other files and other folders', () => {
    const b = fresh()
    expect(onChanged(b, text('h1')).action).toBe('ignore')
    expect(onChanged(b, text('h7', 'manuscript/02.md')).action).toBe('ignore')
    expect(onChanged(b, text('h7', PATH, 'C:\\Other')).action).toBe('ignore')
    expect(onChanged(b, { kind: 'state', folder: FOLDER, state: {} as never }).action).toBe('ignore')
  })

  it('reloads a clean buffer', () => {
    expect(onChanged(fresh(), text('h2')).action).toBe('reload')
  })

  it('warns when there are unsaved edits, and keeps them', () => {
    const b = edit(fresh(), 'Mine.')
    const r = onChanged(b, text('h2'))
    expect(r.action).toBe('banner')
    expect(r.buffer).toMatchObject({ text: 'Mine.', changedOnDisk: 'changed' })
  })

  it('does not react while a save is in flight (its answer tells)', () => {
    expect(onChanged(saving(edit(fresh(), 'Mine.')), text('h2')).action).toBe('ignore')
  })

  it('warns when the file was removed', () => {
    const r = onChanged(edit(fresh(), 'Mine.'), { kind: 'deleted', folder: FOLDER, path: PATH })
    expect(r.action).toBe('banner')
    expect(r.buffer.changedOnDisk).toBe('deleted')
  })
})

describe('unsaved drafts kept on the phone', () => {
  it('uses one key per folder and file', () => {
    expect(draftKey(FOLDER, PATH)).not.toBe(draftKey(FOLDER, 'manuscript/02.md'))
  })

  it('offers a draft that differs from the file', () => {
    const offer = draftToOffer({ text: 'Draft', baseHash: 'h1', savedAt: 1 }, file('Disk', 'h1'))
    expect(offer).toEqual({ draft: { text: 'Draft', baseHash: 'h1', savedAt: 1 }, fileChanged: false })
  })

  it('says when the file also changed since the draft was made', () => {
    expect(draftToOffer({ text: 'Draft', baseHash: 'h1', savedAt: 1 }, file('Disk', 'h2'))?.fileChanged).toBe(true)
  })

  it('offers nothing when there is no draft, it equals the file, or the file is read-only', () => {
    expect(draftToOffer(undefined, file('Disk'))).toBeNull()
    expect(draftToOffer({ text: 'Disk', baseHash: 'h0', savedAt: 1 }, file('Disk'))).toBeNull()
    expect(draftToOffer({ text: 'Draft', baseHash: 'h1', savedAt: 1 }, file('Disk', 'h1', { readOnly: true }))).toBeNull()
  })
})

describe('draft keys per computer', () => {
  it('keeps two computers’ drafts for the same project path apart, and finds the old key', () => {
    expect(draftKey(FOLDER, PATH, 'pc1')).not.toBe(draftKey(FOLDER, PATH, 'pc2'))
    const map = { [draftKey(FOLDER, PATH)]: { text: 'old', baseHash: 'h', savedAt: 1 }, [draftKey(FOLDER, PATH, 'pc2')]: { text: 'two', baseHash: 'h', savedAt: 1 } }
    expect(findDraft(map, FOLDER, PATH, 'pc1')?.text).toBe('old')
    expect(findDraft(map, FOLDER, PATH, 'pc2')?.text).toBe('two')
    expect(findDraft({}, FOLDER, PATH, 'pc1')).toBeUndefined()
  })
})

describe('changes that arrive during our own save', () => {
  const text = (hash: string) => ({ kind: 'text' as const, folder: FOLDER, path: PATH, hash })

  it('ignores our own hash coming back, then ends clean', () => {
    const sent = saving(edit(fresh(), 'Mine.'))
    const during = onChanged(sent, text('h2')).buffer
    const after = afterSave(during, 'Mine.', { ok: true, hash: 'h2' })
    expect(after.needsReload).toBe(false)
    expect(after.changedOnDisk).toBe('none')
  })

  it('reloads after the save when the newest hash differs from the one our save returned', () => {
    const sent = saving(edit(fresh(), 'Mine.'))
    let during = onChanged(sent, text('h2')).buffer
    during = onChanged(during, text('h3')).buffer
    expect(afterSave(during, 'Mine.', { ok: true, hash: 'h2' }).needsReload).toBe(true)
  })

  it('shows the banner instead when more was typed during the save', () => {
    const sent = saving(edit(fresh(), 'Mine.'))
    const during = edit(onChanged(sent, text('h3')).buffer, 'Mine. More.')
    const after = afterSave(during, 'Mine.', { ok: true, hash: 'h2' })
    expect(after).toMatchObject({ changedOnDisk: 'changed', needsReload: false })
  })
})
