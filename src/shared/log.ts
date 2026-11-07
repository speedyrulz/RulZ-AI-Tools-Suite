import type { LogLine } from './ipc'

/**
 * Adds a batch of log lines to the per-app logs, keeping the newest `limit` lines of each. Returns a new object with
 * new arrays for the apps that changed (React state). Each app's log is copied once per batch, not once per line:
 * a batch of a few hundred pip lines into a full log would otherwise copy the whole log for every line.
 */
export function appendLogLines(
  logs: Record<string, string[]>,
  lines: LogLine[],
  limit: number
): Record<string, string[]> {
  const next = { ...logs }
  const copied = new Set<string>()
  for (const { appId, line, replace } of lines) {
    if (!copied.has(appId)) {
      next[appId] = [...(next[appId] ?? [])]
      copied.add(appId)
    }
    const log = next[appId]
    // A progress counter moving on replaces its previous line.
    if (replace) log.pop()
    log.push(line)
  }
  for (const id of copied) {
    const log = next[id]
    if (log.length > limit) log.splice(0, log.length - limit)
  }
  return next
}
