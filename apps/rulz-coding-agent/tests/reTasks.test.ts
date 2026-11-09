import { describe, expect, it } from 'vitest'
import { practiceTask, reTask } from '../src/main/agent/reTasks'

describe('RE tasks', () => {
  it('fills the target into an analyse goal and asks for an ANSWER line', () => {
    const t = reTask({ kind: 'analyse', target: 'C:\\App\\tool.exe', goal: 'Analyze {target}: how are keys checked?' }, 'D:\\proj')
    expect(t.text).toContain('Analyze C:\\App\\tool.exe: how are keys checked?')
    expect(t.text).toContain('ANSWER: <value>')
    expect(t.shown).toContain('tool.exe')
    expect(t.info).toEqual({ kind: 'analyse', target: 'C:\\App\\tool.exe', goal: 'Analyze {target}: how are keys checked?', record: true })
  })
  it('adds the target when the goal does not mention it', () => {
    expect(reTask({ kind: 'analyse', target: 'C:\\a.exe', goal: 'What does it do?', record: false }, 'D:\\p').text).toContain('Target: C:\\a.exe\n\nWhat does it do?')
  })
  it('builds clone tasks that write into the project', () => {
    const pieces = reTask({ kind: 'clone', target: 'C:\\a.exe', scope: 'pieces', details: 'The save format.', language: 'Rust' }, 'D:\\proj')
    expect(pieces.text).toContain('only these pieces')
    expect(pieces.text).toContain('The save format.')
    expect(pieces.text).toContain('in Rust, in this project folder (D:\\proj)')
    expect(pieces.text).toContain('NOTES.md')
    expect(pieces.info).toMatchObject({ kind: 'clone-pieces', language: 'Rust', record: true })
    const full = reTask({ kind: 'clone', target: 'C:\\a.exe', scope: 'full', details: '', language: '' }, 'D:\\proj')
    expect(full.text).toContain('the whole program')
    expect(full.info.kind).toBe('clone-full')
  })
  it('refuses incomplete requests', () => {
    expect(() => reTask({ kind: 'analyse', target: ' ', goal: 'x' }, 'D:\\p')).toThrow(/Choose/)
    expect(() => reTask({ kind: 'analyse', target: 'C:\\a.exe', goal: ' ' }, 'D:\\p')).toThrow(/find out/)
    expect(() => reTask({ kind: 'clone', target: 'C:\\a.exe', scope: 'pieces', details: '', language: '' }, 'D:\\p')).toThrow(/pieces/)
  })
  it('builds a practice task from the prepared copy', () => {
    const t = practiceTask({ target: 'C:\\w\\x.exe', prompt: 'Key in C:\\w\\x.exe?', family: 'xor', tier: 't1' }, 'xor-1', true)
    expect(t.info).toEqual({ kind: 'practice', target: 'C:\\w\\x.exe', goal: 'Key in C:\\w\\x.exe?', record: true, practice: { taskId: 'xor-1', family: 'xor', tier: 't1' } })
  })
})
