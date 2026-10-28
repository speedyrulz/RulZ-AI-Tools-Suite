import { writeFileSync } from 'node:fs'
import { join, resolve } from 'node:path'

/** Writes an mcp.json (like RulZ LLM Suite's) that starts the fake tool server, plus any extra servers. */
export function fakeToolConfig(dir: string, extra: Record<string, unknown> = {}): string {
  const path = join(dir, `mcp-${Math.random().toString(36).slice(2)}.json`)
  const fake = { command: process.execPath, args: [resolve(__dirname, 'fakeToolServer.mjs')] }
  writeFileSync(path, JSON.stringify({ mcpServers: { fake, ...extra } }))
  return path
}
