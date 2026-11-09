import { execFile } from 'node:child_process'

export interface PythonCommand {
  command: string
  args: string[]
  version: string
}

function versionOf(command: string, args: string[]): Promise<string | null> {
  return new Promise((done) => {
    execFile(command, [...args, '--version'], { windowsHide: true, timeout: 8000 }, (err, stdout, stderr) => {
      const m = `${stdout ?? ''}${stderr ?? ''}`.trim().match(/^Python (3\.(\d+)\.\S+)/)
      done(!err && m && Number(m[2]) >= 8 ? m[1] : null)
    })
  })
}

/**
 * Finds a Python 3 to run the RE calculator with: `python`, then the `py -3` launcher. The
 * Microsoft Store alias for `python` fails without printing a version, so it is skipped.
 */
export async function findPython(): Promise<PythonCommand | null> {
  const candidates: [string, string[]][] = [
    ['python', []],
    ['py', ['-3']],
    ['python3', []]
  ]
  for (const [command, args] of candidates) {
    const version = await versionOf(command, args)
    if (version) return { command, args, version }
  }
  return null
}
