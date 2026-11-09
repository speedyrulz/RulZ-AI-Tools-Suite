// Copies the types and helpers shared with RulZ Coding Agent (the desktop app) into src/shared.
// Run after changing them there: npm run sync-shared [path to the desktop app]
import { copyFileSync, existsSync, mkdirSync } from 'node:fs'
import { join, resolve } from 'node:path'

const desktop = resolve(process.argv[2] ?? join(import.meta.dirname, '..', '..', 'RulZ AI Tools Suite', 'apps', 'rulz-coding-agent'))
const from = join(desktop, 'src', 'shared')
if (!existsSync(from)) {
  console.error(`Can't find RulZ Coding Agent's shared code in ${from}. Pass the desktop app's folder.`)
  process.exit(1)
}
const to = join(import.meta.dirname, '..', 'src', 'shared')
mkdirSync(to, { recursive: true })
for (const file of ['ipc.ts', 'types.ts', 'defaults.ts', 'diff.ts', 'runOutput.ts', 'split.ts', 'think.ts']) {
  copyFileSync(join(from, file), join(to, file))
  console.log(`copied ${file}`)
}
