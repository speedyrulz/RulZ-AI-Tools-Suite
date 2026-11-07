import { createHash } from 'node:crypto'
import type { McpServerConfig } from '@shared/types'

export const EMPTY_CONFIG = '{\n  "mcpServers": {}\n}\n'

type ReadText = (path: string) => Promise<string>

/**
 * Reads mcp.json. A missing file is an empty config; other failures (a file briefly locked by an
 * installer or antivirus, say) are retried a few times and then reported instead of being hidden.
 */
export async function readMcpConfigFile(
  path: string,
  read: ReadText,
  attempts = 4,
  delayMs = 250
): Promise<{ text: string; error: null } | { text: null; error: string }> {
  let last: unknown
  for (let i = 0; i < attempts; i++) {
    try {
      return { text: await read(path), error: null }
    } catch (e) {
      if ((e as NodeJS.ErrnoException).code === 'ENOENT') return { text: EMPTY_CONFIG, error: null }
      last = e
      if (i < attempts - 1) await new Promise((r) => setTimeout(r, delayMs))
    }
  }
  return { text: null, error: `Couldn't read mcp.json: ${(last as Error)?.message ?? String(last)}` }
}

/** Parses mcp.json text (the Cursor/LM Studio "mcpServers" format). Throws a readable error. */
export function parseMcpConfig(text: string): Record<string, McpServerConfig> {
  let raw: unknown
  try {
    raw = JSON.parse(text)
  } catch (e) {
    throw new Error(`mcp.json is not valid JSON: ${(e as Error).message}`)
  }
  const servers = (raw as { mcpServers?: unknown })?.mcpServers
  if (!servers || typeof servers !== 'object' || Array.isArray(servers)) {
    throw new Error('mcp.json must contain an "mcpServers" object.')
  }
  const out: Record<string, McpServerConfig> = {}
  for (const [name, value] of Object.entries(servers as Record<string, unknown>)) {
    const v = value as McpServerConfig
    if (!v || typeof v !== 'object') throw new Error(`Server "${name}" must be an object.`)
    if (!v.command && !v.url) throw new Error(`Server "${name}" needs either "command" (local) or "url" (remote).`)
    if (v.args !== undefined && (!Array.isArray(v.args) || v.args.some((a) => typeof a !== 'string'))) {
      throw new Error(`Server "${name}": "args" must be a list of strings.`)
    }
    out[name] = v
  }
  return out
}

const cleanName = (s: string): string => s.replace(/[^A-Za-z0-9_-]/g, '_')
const shortHash = (s: string): string => createHash('sha1').update(s).digest('hex').slice(0, 6)

/** Tool name shown to the model; OpenAI-style names allow [A-Za-z0-9_-]{1,64}. */
export function qualifyToolName(server: string, tool: string): string {
  return `${cleanName(server)}__${cleanName(tool)}`.slice(0, 64)
}

/**
 * Qualified names for one server's tools, kept unique when cleaning or shortening would make two
 * look alike (calls are routed by this name, so a clash would run the wrong tool). Of names that
 * clash, the one that needed no change keeps it and the others get a short hash of their real name.
 * Settings remember tools by this name (disabled, always allowed), so the choice never depends on
 * the order servers connect or list their tools in, and names that don't clash stay exactly as
 * qualifyToolName makes them.
 */
export function qualifyToolNames(server: string, tools: string[], allServers: string[]): string[] {
  const mine = cleanName(server)
  const clashes = allServers.some((s) => {
    if (s === server) return false
    const other = cleanName(s)
    // Same cleaned name: a server whose real name is already clean keeps it.
    if (other === mine) return mine !== server
    // "a__b" + "c" and "a" + "b__c" would both read a__b__c; the longer server name gives way.
    return mine.startsWith(`${other}__`)
  })
  const prefix = clashes ? `${mine}_${shortHash(server)}` : mine
  const base = tools.map((t) => `${prefix}__${cleanName(t)}`.slice(0, 64))
  const count = new Map<string, number>()
  for (const q of base) count.set(q, (count.get(q) ?? 0) + 1)
  return tools.map((t, i) => {
    const q = base[i]
    if (count.get(q) === 1) return q
    // Only a tool whose name came through unchanged (clean, not cut short) can keep it, and tool
    // names are unique on a server, so at most one does.
    const unchanged = q === `${prefix}__${t}`
    return unchanged ? q : `${q.slice(0, 57)}_${shortHash(t)}`
  })
}

/** Flattens MCP tool-call content into text for the model. */
export function contentToText(content: unknown): string {
  if (!Array.isArray(content)) return typeof content === 'string' ? content : JSON.stringify(content)
  return content
    .map((c: { type?: string; text?: string; mimeType?: string; resource?: { uri?: string; text?: string } }) => {
      if (c.type === 'text') return c.text ?? ''
      if (c.type === 'image' || c.type === 'audio') return `[${c.type}: ${c.mimeType ?? 'binary'}]`
      if (c.type === 'resource') return c.resource?.text ?? `[resource: ${c.resource?.uri ?? ''}]`
      if (c.type === 'resource_link') return `[resource link: ${(c as { uri?: string }).uri ?? ''}]`
      return JSON.stringify(c)
    })
    .join('\n')
}
