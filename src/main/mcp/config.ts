import type { McpServerConfig } from '@shared/types'

export const EMPTY_CONFIG = '{\n  "mcpServers": {}\n}\n'

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
