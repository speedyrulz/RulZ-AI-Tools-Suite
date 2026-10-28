// Tool servers (MCP) set up in RulZ LLM Suite, such as REA. They start the first time the agent
// looks for a tool, and their tools are offered to the model only once it loads them (REA alone
// has over 200, far more than fit in the model's context). Adapted from RulZ LLM Suite's McpManager.
import { existsSync, promises as fs } from 'node:fs'
import { homedir } from 'node:os'
import { join } from 'node:path'
import { Client } from '@modelcontextprotocol/sdk/client/index.js'
import { getDefaultEnvironment, StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js'
import { StreamableHTTPClientTransport } from '@modelcontextprotocol/sdk/client/streamableHttp.js'
import type { ToolServerStatus } from '@shared/types'
import { contentToText, qualifyToolName, readToolConfig, searchTools, type ToolServerConfig } from './config'

/** Tool calls that take longer than this are given up (REA's decompiler can take a while). */
const CALL_TIMEOUT_MS = 5 * 60_000
const MAX_LOG = 100

export interface ToolInfo {
  /** Name the model calls, e.g. "rea__open_binary". */
  qualifiedName: string
  server: string
  name: string
  description?: string
  inputSchema: unknown
  /** The server says the tool may change things or run programs (asked about every time). */
  destructive: boolean
}

interface Connection {
  status: ToolServerStatus
  tools: ToolInfo[]
  instructions?: string
  client?: Client
  starting?: Promise<void>
}

/** Folders where npx-style launchers live but that an app started from Explorer may lack on its PATH. */
function launcherDirs(): string[] {
  const home = homedir()
  return [
    'C:\\Program Files\\nodejs',
    join(process.env.APPDATA ?? join(home, 'AppData', 'Roaming'), 'npm'),
    join(home, '.local', 'bin'),
    join(home, '.cargo', 'bin')
  ].filter((d) => existsSync(d))
}

export class ToolServers {
  private connections = new Map<string, Connection>()
  private configError: string | null = null

  constructor(
    readonly configPath: string,
    private readonly onChange: () => void = () => undefined
  ) {}

  /** Names of the servers set up (and not switched off) in mcp.json; read fresh each time. */
  async configured(): Promise<string[]> {
    const read = await readToolConfig(this.configPath, (p) => fs.readFile(p, 'utf8'))
    this.configError = read.error
    return Object.entries(read.servers)
      .filter(([, c]) => !c.disabled)
      .map(([name]) => name)
  }

  status(): { configPath: string; error: string | null; servers: ToolServerStatus[] } {
    return { configPath: this.configPath, error: this.configError, servers: [...this.connections.values()].map((c) => ({ ...c.status, log: c.status.log.slice() })) }
  }

  /** Starts every configured server that isn't running yet, and waits for them. */
  async start(): Promise<void> {
    const read = await readToolConfig(this.configPath, (p) => fs.readFile(p, 'utf8'))
    this.configError = read.error
    const waits: Promise<void>[] = []
    for (const [name, cfg] of Object.entries(read.servers)) {
      if (cfg.disabled) continue
      const conn = this.connections.get(name)
      if (conn?.status.state === 'connected') continue
      if (conn?.starting) {
        waits.push(conn.starting)
        continue
      }
      waits.push(this.connect(name, cfg))
    }
    await Promise.all(waits)
  }

  private connect(name: string, cfg: ToolServerConfig): Promise<void> {
    const status: ToolServerStatus = { name, state: 'starting', tools: 0, log: [] }
    const conn: Connection = { status, tools: [] }
    this.connections.set(name, conn)
    this.onChange()
    const log = (line: string): void => {
      status.log.push(line)
      if (status.log.length > MAX_LOG) status.log.splice(0, status.log.length - MAX_LOG)
    }
    conn.starting = (async () => {
      try {
        const client = new Client({ name: 'RulZ Coding Agent', version: '1.0.0' })
        if (cfg.url) {
          await client.connect(new StreamableHTTPClientTransport(new URL(cfg.url), { requestInit: cfg.headers ? { headers: cfg.headers } : undefined }))
        } else {
          const base = getDefaultEnvironment()
          const pathKey = Object.keys(base).find((k) => k.toLowerCase() === 'path') ?? 'PATH'
          const env = { ...base, [pathKey]: [...launcherDirs(), base[pathKey] ?? ''].join(';'), ...cfg.env }
          const transport = new StdioClientTransport({ command: cfg.command!, args: cfg.args ?? [], env, cwd: cfg.cwd, stderr: 'pipe' })
          transport.stderr?.on('data', (d: Buffer) => {
            for (const line of d.toString().split(/\r?\n/)) if (line.trim()) log(line)
          })
          await client.connect(transport)
        }
        conn.client = client
        client.onclose = () => {
          if (this.connections.get(name) !== conn || status.state !== 'connected') return
          status.state = 'error'
          status.error = 'The tool server closed the connection.'
          conn.client = undefined
          this.onChange()
        }
        const tools: ToolInfo[] = []
        let cursor: string | undefined
        do {
          const page = await client.listTools(cursor ? { cursor } : undefined)
          for (const t of page.tools) {
            tools.push({
              qualifiedName: qualifyToolName(name, t.name),
              server: name,
              name: t.name,
              description: t.description,
              inputSchema: t.inputSchema ?? { type: 'object', properties: {} },
              destructive: t.annotations?.destructiveHint === true
            })
          }
          cursor = page.nextCursor
        } while (cursor)
        conn.tools = tools
        conn.instructions = client.getInstructions()
        status.tools = tools.length
        status.state = 'connected'
        status.error = undefined
      } catch (e) {
        status.state = 'error'
        status.error = (e as Error).message
        log(`[error] ${(e as Error).message}`)
        await conn.client?.close().catch(() => undefined)
        conn.client = undefined
      } finally {
        conn.starting = undefined
        this.onChange()
      }
    })()
    return conn.starting
  }

  /** Every tool of every connected server. */
  tools(): ToolInfo[] {
    return [...this.connections.values()].filter((c) => c.status.state === 'connected').flatMap((c) => c.tools)
  }

  /** A tool by its full name ("rea__open_binary") or, when only one server has it, its own name. */
  find(name: string): ToolInfo | undefined {
    const all = this.tools()
    const exact = all.find((t) => t.qualifiedName === name)
    if (exact) return exact
    const bare = all.filter((t) => t.name === name)
    return bare.length === 1 ? bare[0] : undefined
  }

  search(query: string, limit = 10): ToolInfo[] {
    return searchTools(this.tools(), query, limit)
  }

  /** Each connected server's own advice to the model, if it gives any. */
  instructions(): { server: string; text: string }[] {
    return [...this.connections.values()]
      .filter((c) => c.status.state === 'connected' && c.instructions?.trim())
      .map((c) => ({ server: c.status.name, text: c.instructions!.trim() }))
  }

  async call(tool: ToolInfo, args: Record<string, unknown>, signal?: AbortSignal): Promise<{ text: string; isError: boolean }> {
    const client = this.connections.get(tool.server)?.client
    if (!client) return { text: `The tool server "${tool.server}" is not running.`, isError: true }
    try {
      const res = await client.callTool({ name: tool.name, arguments: args }, undefined, { timeout: CALL_TIMEOUT_MS, resetTimeoutOnProgress: true, signal })
      let text = contentToText(res.content)
      if (!text && res.structuredContent) text = JSON.stringify(res.structuredContent)
      return { text: text || '(no output)', isError: !!res.isError }
    } catch (e) {
      if (signal?.aborted) throw e
      return { text: `The tool failed: ${(e as Error).message}`, isError: true }
    }
  }

  async dispose(): Promise<void> {
    const all = [...this.connections.values()]
    this.connections.clear()
    await Promise.all(all.map((c) => c.client?.close().catch(() => undefined)))
  }
}
