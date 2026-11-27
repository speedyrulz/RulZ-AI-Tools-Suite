import { describe, expect, it } from 'vitest'
import { TerminalBuffer } from '../src/renderer/src/lib/terminal'

describe('TerminalBuffer', () => {
  it('splits lines', () => {
    const t = new TerminalBuffer()
    t.write('a\nb\r\nc')
    expect(t.text()).toBe('a\nb\nc')
  })

  it('overwrites the line after a carriage return', () => {
    const t = new TerminalBuffer()
    t.write('10%\r50%\r100%')
    expect(t.text()).toBe('100%')
    t.write('\rab')
    expect(t.text()).toBe('ab0%')
  })

  it('erase-in-line clears the old tail', () => {
    const t = new TerminalBuffer()
    t.write('long progress line\r\x1b[Kshort')
    expect(t.text()).toBe('short')
    t.write('\r\x1b[2Kx')
    expect(t.text()).toBe('x')
    t.write('\rabc\x1b[0K')
    expect(t.text()).toBe('abc')
  })

  it('strips other escape codes', () => {
    const t = new TerminalBuffer()
    t.write('\x1b[32mgreen\x1b[0m done')
    expect(t.text()).toBe('green done')
  })

  it('carries an escape sequence split across chunks', () => {
    const t = new TerminalBuffer()
    t.write('abcdef\r\x1b')
    t.write('[')
    t.write('Kxy')
    expect(t.text()).toBe('xy')
    t.write('\x1b[3')
    t.write('2mok')
    expect(t.text()).toBe('xyok')
  })

  it('keeps only the last lines', () => {
    const t = new TerminalBuffer(3)
    t.write('1\n2\n3\n4\n5')
    expect(t.text()).toBe('3\n4\n5')
  })
})
