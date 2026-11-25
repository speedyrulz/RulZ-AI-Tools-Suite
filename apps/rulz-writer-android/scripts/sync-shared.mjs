// Copies the contract and helpers shared with RulZ LLM Suite (the desktop app) into src/shared.
// Run after changing them there: npm run sync-shared [path to the desktop app]
// phone.ts is the contract; the other files are what it imports (novel.ts, which imports types.ts) and the
// pure helpers the phone uses (outline ids, tracked-state reading order, check quotes, voice notes).
// All of them are free of node: imports and heavy libraries.
import { copyFileSync, existsSync, mkdirSync } from 'node:fs'
import { join, resolve } from 'node:path'

const desktop = resolve(process.argv[2] ?? join(import.meta.dirname, '..', '..', 'rulz-llm-suite'))
const from = join(desktop, 'src', 'shared')
if (!existsSync(from)) {
  console.error(`Can't find RulZ LLM Suite's shared code in ${from}. Pass the desktop app's folder.`)
  process.exit(1)
}
const to = join(import.meta.dirname, '..', 'src', 'shared')
mkdirSync(to, { recursive: true })
for (const file of ['phone.ts', 'adventure.ts', 'novel.ts', 'types.ts', 'remote.ts', 'novelText.ts', 'novelOutline.ts', 'novelState.ts', 'novelChecks.ts', 'novelVoice.ts']) {
  copyFileSync(join(from, file), join(to, file))
  console.log(`copied ${file}`)
}
