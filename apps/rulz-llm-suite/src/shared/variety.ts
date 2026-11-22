/** Text forms of the list settings behind the DRY and "Avoid words" controls. */

/** One breaker per line; a newline or tab is written as \n or \t so it stays on its line. */
export function breakersToText(breakers: string[]): string {
  return breakers.map((b) => b.replace(/\\/g, '\\\\').replace(/\n/g, '\\n').replace(/\t/g, '\\t')).join('\n')
}

export function textToBreakers(text: string): string[] {
  return text
    .split('\n')
    .filter((l) => l.length > 0)
    .map((l) => l.replace(/\\(\\|n|t)/g, (_m, c: string) => (c === 'n' ? '\n' : c === 't' ? '\t' : '\\')))
}

/** One word per line. */
export function textToWords(text: string): string[] {
  return text.split('\n').map((l) => l.trim()).filter((l) => l.length > 0)
}
