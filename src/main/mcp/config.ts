// Reads the tool servers set up in RulZ LLM Suite (its mcp.json), so they only need setting up once.
// Adapted from RulZ LLM Suite's src/main/mcp/config.ts; keep the two in step.

export interface ToolServerConfig {
  command?: string
  args?: string[]
  env?: Record<string, string>
  cwd?: string
  url?: string
  headers?: Record<string, string>
  disabled?: boolean
}

type ReadText = (path: string) => Promise<string>

/**
 * Reads mcp.json. A missing file means no tool servers; other failures (a file briefly locked by
 * RulZ LLM Suite saving it, say) are retried a few times and then reported.
 */
export async function readToolConfig(
  path: string,
  read: ReadText,
  attempts = 4,
  delayMs = 250
): Promise<{ servers: Record<string, ToolServerConfig>; error: null } | { servers: Record<string, ToolServerConfig>; error: string }> {
  let last: unknown
  for (let i = 0; i < attempts; i++) {
    try {
      return { servers: parseToolConfig(await read(path)), error: null }
    } catch (e) {
      if ((e as NodeJS.ErrnoException).code === 'ENOENT') return { servers: {}, error: null }
      if (!(e as NodeJS.ErrnoException).code) return { servers: {}, error: (e as Error).message }
      last = e
      if (i < attempts - 1) await new Promise((r) => setTimeout(r, delayMs))
    }
  }
  return { servers: {}, error: `Couldn't read ${path}: ${(last as Error)?.message ?? String(last)}` }
}

/** Parses mcp.json text (the "mcpServers" format RulZ LLM Suite, Cursor and LM Studio use). Throws a readable error. */
export function parseToolConfig(text: string): Record<string, ToolServerConfig> {
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
  const out: Record<string, ToolServerConfig> = {}
  for (const [name, value] of Object.entries(servers as Record<string, unknown>)) {
    const v = value as ToolServerConfig
    if (!v || typeof v !== 'object') throw new Error(`Server "${name}" must be an object.`)
    if (!v.command && !v.url) throw new Error(`Server "${name}" needs either "command" (local) or "url" (remote).`)
    if (v.args !== undefined && (!Array.isArray(v.args) || v.args.some((a) => typeof a !== 'string'))) {
      throw new Error(`Server "${name}": "args" must be a list of strings.`)
    }
    out[name] = v
  }
  return out
}

/** Tool name shown to the model; OpenAI-style names allow [A-Za-z0-9_-]{1,64}. */
export function qualifyToolName(server: string, tool: string): string {
  const clean = (s: string): string => s.replace(/[^A-Za-z0-9_-]/g, '_')
  return `${clean(server)}__${clean(tool)}`.slice(0, 64)
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

/** First sentence of a tool description, kept short (for search results). */
export function toolSummary(description: string | undefined, max = 160): string {
  const first = (description ?? '').split(/\n/)[0].split(/(?<=\.)\s/)[0].trim()
  return first.length > max ? `${first.slice(0, max - 1)}…` : first
}

export interface SearchableTool {
  name: string
  description?: string
}

/** Keyword search over tool names and descriptions; name hits weigh more (as in RulZ LLM Suite). */
export function searchTools<T extends SearchableTool>(tools: T[], query: string, limit = 10): T[] {
  const words = query
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter((w) => w.length > 1)
  if (!words.length) return []
  return tools
    .map((t) => {
      const name = t.name.toLowerCase()
      const desc = (t.description ?? '').toLowerCase()
      let score = 0
      for (const w of words) {
        if (name === w) score += 10
        else if (name.includes(w)) score += 4
        if (desc.includes(w)) score += 1
      }
      return { t, score }
    })
    .filter((x) => x.score > 0)
    .sort((a, b) => b.score - a.score || a.t.name.localeCompare(b.t.name))
    .slice(0, limit)
    .map((x) => x.t)
}
