// Turns llama-server's startup log into a progress value and a human-readable stage.

export interface LoadProgress {
  progress: number
  stage: string
}

const STAGES: [RegExp, number, string][] = [
  [/llama_model_loader: loaded meta data/i, 0.05, 'Reading model'],
  [/load_tensors:/i, 0.1, 'Loading weights'],
  [/llama_context:/i, 0.86, 'Allocating context'],
  [/clip_model_loader|mtmd_|loading mmproj/i, 0.9, 'Loading vision projector'],
  [/warming up|warmup/i, 0.94, 'Warming up'],
  [/server is listening|all slots are idle|model loaded/i, 0.98, 'Starting server']
]

/** Weight loading is reported by llama.cpp as up to 100 dots; map them onto 0.1–0.85. */
export function progressFromDots(dots: number): number {
  return 0.1 + 0.75 * Math.min(1, dots / 100)
}

export function nextProgress(current: LoadProgress, line: string): LoadProgress {
  for (const [re, value, stage] of STAGES) {
    if (re.test(line) && value > current.progress) return { progress: value, stage }
  }
  return current
}

/**
 * Splits raw output into complete lines while tracking the in-progress dot line, which llama.cpp
 * prints without newlines while weights load.
 */
export class LogLineSplitter {
  private partial = ''

  /** Returns completed lines and the number of dots on the current dot line (if any). */
  push(chunk: string): { lines: string[]; dots: number | null } {
    this.partial += chunk.replace(/\x1b\[[0-9;]*m/g, '').replace(/\r(?!\n)/g, '\n')
    const parts = this.partial.split(/\r?\n/)
    this.partial = parts.pop() ?? ''
    let dots: number | null = null
    const lines: string[] = []
    for (const l of parts) {
      if (/^\.+$/.test(l)) dots = l.length
      else if (l.trim()) lines.push(l)
    }
    if (/^\.+$/.test(this.partial)) dots = this.partial.length
    return { lines, dots }
  }

  flush(): string[] {
    const rest = this.partial.trim()
    this.partial = ''
    return rest && !/^\.+$/.test(rest) ? [rest] : []
  }
}
