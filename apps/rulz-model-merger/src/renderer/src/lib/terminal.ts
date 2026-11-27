// A tiny terminal model: "\r" returns to the start of the current line so progress bars update in place,
// and erase-in-line (ESC [ K) clears the old tail. Other escape sequences are dropped.

const ESC = '\x1b'
const CSI = /\x1b\[([0-9;?]*)([ -/]*)([@-~])/y
const CSI_PREFIX = /^\x1b\[[0-9;?]*[ -/]*$/

export class TerminalBuffer {
  lines: string[] = ['']
  private col = 0
  /** An escape sequence cut off at the end of the last chunk. */
  private pending = ''

  constructor(private readonly maxLines = 2000) {}

  reset(): void {
    this.lines = ['']
    this.col = 0
    this.pending = ''
  }

  private eraseInLine(mode: string): void {
    const last = this.lines.length - 1
    const line = this.lines[last]
    if (mode === '2') {
      this.lines[last] = ''
    } else if (mode === '1') {
      this.lines[last] = ' '.repeat(Math.min(this.col + 1, line.length)) + line.slice(this.col + 1)
    } else {
      this.lines[last] = line.slice(0, this.col)
    }
  }

  write(chunk: string): void {
    const text = this.pending + chunk
    this.pending = ''
    let i = 0
    while (i < text.length) {
      const ch = text[i]
      if (ch === ESC) {
        if (i + 1 >= text.length) {
          this.pending = text.slice(i)
          break
        }
        if (text[i + 1] !== '[') {
          i += 2
          continue
        }
        CSI.lastIndex = i
        const m = CSI.exec(text)
        if (m) {
          if (m[3] === 'K') this.eraseInLine(m[1] === '' ? '0' : m[1])
          i += m[0].length
        } else if (CSI_PREFIX.test(text.slice(i))) {
          this.pending = text.slice(i)
          break
        } else {
          i++
        }
      } else if (ch === '\r') {
        this.col = 0
        i++
      } else if (ch === '\n') {
        this.lines.push('')
        this.col = 0
        i++
      } else {
        let j = i
        while (j < text.length && text[j] !== '\r' && text[j] !== '\n' && text[j] !== ESC) j++
        const seg = text.slice(i, j)
        const last = this.lines.length - 1
        const line = this.lines[last]
        this.lines[last] = line.slice(0, this.col).padEnd(this.col, ' ') + seg + line.slice(this.col + seg.length)
        this.col += seg.length
        i = j
      }
    }
    if (this.lines.length > this.maxLines) this.lines.splice(0, this.lines.length - this.maxLines)
  }

  text(): string {
    return this.lines.join('\n')
  }
}
