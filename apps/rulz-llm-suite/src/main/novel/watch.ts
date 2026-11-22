// Folder watcher for an open project (data safety rule 3): fs.watch, recursive, .rulz ignored, 300 ms debounce per path.
import { watch, type FSWatcher } from 'node:fs'

export class ProjectWatcher {
  private watcher: FSWatcher | null = null
  private timers = new Map<string, NodeJS.Timeout>()

  constructor(
    private folder: string,
    private onChange: (rel: string) => void,
    private debounceMs = 300
  ) {}

  start(): void {
    if (this.watcher) return
    try {
      this.watcher = watch(this.folder, { recursive: true, persistent: false }, (_event, name) => {
        if (!name) return
        const rel = String(name).replace(/\\/g, '/')
        if (rel === '.rulz' || rel.toLowerCase().startsWith('.rulz/')) return
        const key = rel.toLowerCase()
        clearTimeout(this.timers.get(key))
        this.timers.set(
          key,
          setTimeout(() => {
            this.timers.delete(key)
            this.onChange(rel)
          }, this.debounceMs)
        )
      })
      // The folder can vanish (drive removed, project deleted outside); there is nothing to do but stop.
      this.watcher.on('error', () => this.stop())
    } catch {
      this.watcher = null
    }
  }

  stop(): void {
    this.watcher?.close()
    this.watcher = null
    for (const t of this.timers.values()) clearTimeout(t)
    this.timers.clear()
  }
}
