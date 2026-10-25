import { mkdir, readdir, rm, rmdir } from 'node:fs/promises'
import { dirname, join } from 'node:path'
import { app, shell } from 'electron'
import { quoteArg } from './process'

export const START_MENU_FOLDER = 'RulZ AI Tools Suite'

function fileName(name: string): string {
  return `${name.replace(/[<>:"/\\|?*]/g, '')}.lnk`
}

export function startMenuDir(): string {
  const appData = process.env.APPDATA ?? join(app.getPath('home'), 'AppData', 'Roaming')
  return join(appData, 'Microsoft', 'Windows', 'Start Menu', 'Programs', START_MENU_FOLDER)
}

export function shortcutPaths(name: string): { desktop: string; startMenu: string } {
  // getPath('desktop') follows the known folder, so OneDrive-redirected desktops work.
  return { desktop: join(app.getPath('desktop'), fileName(name)), startMenu: join(startMenuDir(), fileName(name)) }
}

/** Shortcuts other programs made, e.g. an app's own setup program. */
export function foreignShortcutPaths(name: string): string[] {
  const programs = dirname(startMenuDir())
  return [join(app.getPath('desktop'), fileName(name)), join(programs, fileName(name))]
}

export interface ShortcutTarget {
  target: string
  args: string[]
  cwd: string
  icon?: string
  description: string
}

export async function createShortcut(lnk: string, t: ShortcutTarget): Promise<void> {
  await mkdir(dirname(lnk), { recursive: true })
  // 'create' makes the shortcut or overwrites it ('replace' fails when it doesn't exist yet).
  const ok = shell.writeShortcutLink(lnk, 'create', {
    target: t.target,
    args: t.args.map(quoteArg).join(' '),
    cwd: t.cwd,
    description: t.description,
    ...(t.icon ? { icon: t.icon, iconIndex: 0 } : {})
  })
  if (!ok) throw new Error(`Couldn't create the shortcut ${lnk}`)
}

export async function removeShortcut(lnk: string | null): Promise<void> {
  if (!lnk) return
  await rm(lnk, { force: true })
  // Drop the suite's Start Menu folder once its last shortcut is gone.
  if (dirname(lnk) === startMenuDir()) {
    const left = await readdir(startMenuDir()).catch(() => null)
    if (left && left.length === 0) await rmdir(startMenuDir()).catch(() => undefined)
  }
}
