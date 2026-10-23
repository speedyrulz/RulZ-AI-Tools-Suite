// On-demand tool loading: built-in tools that let the model find and load MCP tools itself,
// so only the descriptions it needs take up context.
import type { McpToolInfo, ToolSearchHit } from './types'

export const META_SEARCH = 'rulz_search_tools'
export const META_LOAD = 'rulz_load_tools'
export const META_UNLOAD = 'rulz_unload_tools'
export const META_TOOLS = new Set([META_SEARCH, META_LOAD, META_UNLOAD])

const namesParam = {
  type: 'object',
  properties: {
    names: { type: 'array', items: { type: 'string' }, description: 'Tool names, e.g. "open_binary" or "rea__open_binary".' }
  },
  required: ['names']
}

export const META_TOOL_DEFS = [
  {
    type: 'function' as const,
    function: {
      name: META_SEARCH,
      description: 'Search the available tools by keyword. Returns matching tool names with one-line descriptions. Load a tool with rulz_load_tools before calling it.',
      parameters: {
        type: 'object',
        properties: {
          query: { type: 'string', description: 'Keywords describing what you need, e.g. "decompile function".' },
          limit: { type: 'integer', description: 'Maximum results (default 10).' }
        },
        required: ['query']
      }
    }
  },
  {
    type: 'function' as const,
    function: {
      name: META_LOAD,
      description: 'Load tools so you can call them in your next step. Load only what the current step needs.',
      parameters: namesParam
    }
  },
  {
    type: 'function' as const,
    function: {
      name: META_UNLOAD,
      description: 'Unload tools you no longer need, to free context.',
      parameters: namesParam
    }
  }
]

/** First sentence of a tool description, kept short. */
export function toolSummary(description: string | undefined, max = 160): string {
  const first = (description ?? '').split(/\n/)[0].split(/(?<=\.)\s/)[0].trim()
  return first.length > max ? `${first.slice(0, max - 1)}…` : first
}

/** Rough token cost of one tool definition as sent to the model (~3.5 characters per token). */
export function toolTokens(t: Pick<McpToolInfo, 'qualifiedName' | 'description' | 'inputSchema'>): number {
  return Math.ceil(JSON.stringify({ name: t.qualifiedName, description: t.description, parameters: t.inputSchema }).length / 3.5)
}

/** Keyword search over tool names and descriptions; name hits weigh more. */
export function searchTools(tools: McpToolInfo[], query: string, limit = 10): ToolSearchHit[] {
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
    .slice(0, Math.max(1, Math.min(limit, 30)))
    .map(({ t }) => ({ name: t.name, server: t.server, summary: toolSummary(t.description) }))
}

/** Resolves "tool", "server__tool" or "server.tool" to an available tool. */
export function resolveTool(tools: McpToolInfo[], name: string): McpToolInfo | undefined {
  const n = name.trim()
  const exact = tools.find((t) => t.qualifiedName === n)
  if (exact) return exact
  const sep = n.match(/^(.+?)(?:__|\.|\/)(.+)$/)
  if (sep) {
    const hit = tools.find((t) => t.server === sep[1] && t.name === sep[2])
    if (hit) return hit
  }
  const bare = tools.filter((t) => t.name === n)
  return bare.length === 1 ? bare[0] : undefined
}

const CATALOG_CHAR_LIMIT = 6000

/** System-prompt note for on-demand mode: how to load tools, and which exist. */
export function toolCatalogNote(available: McpToolInfo[], active: string[]): string {
  const byServer = new Map<string, string[]>()
  for (const t of available) byServer.set(t.server, [...(byServer.get(t.server) ?? []), t.name])
  let catalog = ''
  let omitted = 0
  for (const [server, names] of byServer) {
    const line = `- ${server}: ${names.join(', ')}\n`
    if (catalog.length + line.length > CATALOG_CHAR_LIMIT) omitted += names.length
    else catalog += line
  }
  const loaded = active.length ? active.join(', ') : 'none'
  return [
    'You can use tools from connected MCP servers, but only loaded tools can be called.',
    `Loaded tools: ${loaded}.`,
    `To use another tool, call ${META_LOAD} with its name (or ${META_SEARCH} to find one by keyword). Load only what the next step needs, and call ${META_UNLOAD} for tools you are finished with.`,
    'Available tools by server:',
    catalog.trimEnd() + (omitted ? `\n(${omitted} more tools not listed; use ${META_SEARCH}.)` : '')
  ].join('\n')
}
