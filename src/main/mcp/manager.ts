import { existsSync, promises as fs } from 'node:fs'
import { homedir } from 'node:os'
import { join } from 'node:path'
import { Client } from '@modelcontextprotocol/sdk/client/index.js'
import { getDefaultEnvironment, StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js'
import { StreamableHTTPClientTransport } from '@modelcontextprotocol/sdk/client/streamableHttp.js'
import type { McpConfigState, McpServerConfig, McpServerStatus, McpToolInfo, ToolResult, WireToolCall } from '@shared/types'
import { contentToText, EMPTY_CONFIG, parseMcpConfig, qualifyToolName, readMcpConfigFile } from './config'

const MAX_LOG = 200
const MAX_RESULT_CHARS = 24000
const CALL_TIMEOUT_MS = 120_000

interface Connection {
  status: McpServerStatus
  client?: Client
  close?: () => Promise<void>
}

/**
 * Folders where npx/uvx-style launchers usually live but that are often missing from the PATH
 * an app started from Explorer receives.
 */
function launcherDirs(): string[] {
  const home = homedir()
  return [
    'C:\\Program Files\\nodejs',
    join(process.env.APPDATA ?? join(home, 'AppData', 'Roaming'), 'npm'),
    join(home, '.local', 'bin'),
    join(home, '.cargo', 'bin')
  ].filter((d) => existsSync(d))
}

/** Connects to the MCP servers in mcp.json and exposes their tools to chats. */
export class McpManager {
  private connections = new Map<string, Connection>()
  private configText = EMPTY_CONFIG
  private configError: string | null = null
  private readFailed = false

  constructor(
    private readonly configPath: string,
    private readonly onChange: (status: McpServerStatus[]) => void,
    private readonly onConfig: (config: McpConfigState) => void = () => undefined
  ) {}

  status(): McpServerStatus[] {
    return [...this.connections.values()].map((c) => ({ ...c.status, log: c.status.log.slice() }))
  }

  config(): McpConfigState {
    return { text: this.configText, path: this.configPath, error: this.configError, readFailed: this.readFailed }
  }

  async init(): Promise<void> {
    await this.reload(null)
  }

  /** Validates and saves mcp.json, then reconnects every server. */
  async saveConfig(text: string): Promise<McpServerStatus[]> {
    const servers = parseMcpConfig(text)
    await fs.writeFile(this.configPath, text, 'utf8')
    this.configText = text
    this.configError = null
    this.readFailed = false
    this.emitConfig()
    await this.apply(servers)
    return this.status()
  }

  /** Re-reads mcp.json from disk, then reconnects one server (by name) or all of them (null). */
  async restart(name: string | null): Promise<void> {
    await this.reload(name)
  }

  /**
   * Reads and parses mcp.json and reconnects. If the file can't be read or parsed, the problem is
   * reported on the MCP page and the servers that are already connected are left alone.
   */
  private async reload(name: string | null): Promise<void> {
    const read = await readMcpConfigFile(this.configPath, (p) => fs.readFile(p, 'utf8'))
    this.readFailed = read.text === null
    if (read.text !== null) this.configText = read.text
    let servers: Record<string, McpServerConfig> | null = null
    this.configError = read.error
    if (read.text !== null) {
      try {
        servers = parseMcpConfig(read.text)
      } catch (e) {
        this.configError = (e as Error).message
      }
    }
    this.emitConfig()
    if (!servers) return
    if (name === null) return this.apply(servers)
    const cfg = servers[name]
    await this.disconnect(name)
    if (cfg) await this.connect(name, cfg)
    else this.emit()
  }

  private async apply(servers: Record<string, McpServerConfig>): Promise<void> {
    await Promise.all([...this.connections.keys()].map((n) => this.disconnect(n)))
    this.connections.clear()
    await Promise.all(Object.entries(servers).map(([name, cfg]) => this.connect(name, cfg)))
  }

  private async connect(name: string, cfg: McpServerConfig): Promise<void> {
    const status: McpServerStatus = {
      name,
      transport: cfg.url ? 'http' : 'stdio',
      state: cfg.disabled ? 'disabled' : 'connecting',
      tools: [],
      log: []
    }
    const conn: Connection = { status }
    this.connections.set(name, conn)
    this.emit()
    if (cfg.disabled) return

    const log = (line: string): void => {
      status.log.push(line)
      if (status.log.length > MAX_LOG) status.log.splice(0, status.log.length - MAX_LOG)
    }

    try {
      const client = new Client({ name: 'RulZ LLM Suite', version: '0.2.0' })
      if (cfg.url) {
        const transport = new StreamableHTTPClientTransport(new URL(cfg.url), {
          requestInit: cfg.headers ? { headers: cfg.headers } : undefined
        })
        await client.connect(transport)
      } else {
        const base = getDefaultEnvironment()
        const pathKey = Object.keys(base).find((k) => k.toLowerCase() === 'path') ?? 'PATH'
        const env = { ...base, [pathKey]: [...launcherDirs(), base[pathKey] ?? ''].join(';'), ...cfg.env }
        const transport = new StdioClientTransport({
          command: cfg.command!,
          args: cfg.args ?? [],
          env,
          cwd: cfg.cwd,
          stderr: 'pipe'
        })
        transport.stderr?.on('data', (d: Buffer) => {
          for (const line of d.toString().split(/\r?\n/)) if (line.trim()) log(line)
        })
        await client.connect(transport)
      }
      conn.client = client
      conn.close = () => client.close()
      client.onclose = () => {
        if (this.connections.get(name) !== conn || status.state !== 'connected') return
        status.state = 'error'
        status.error = 'The server closed the connection.'
        this.emit()
      }

      const tools: McpToolInfo[] = []
      let cursor: string | undefined
      do {
        const page = await client.listTools(cursor ? { cursor } : undefined)
        for (const t of page.tools) {
          tools.push({
            qualifiedName: qualifyToolName(name, t.name),
            server: name,
            name: t.name,
            description: t.description,
            inputSchema: t.inputSchema
          })
        }
        cursor = page.nextCursor
      } while (cursor)

      status.tools = tools
      status.state = 'connected'
      status.error = undefined
    } catch (err) {
      status.state = 'error'
      status.error = (err as Error).message
      log(`[error] ${(err as Error).message}`)
      await conn.close?.().catch(() => undefined)
    }
    this.emit()
  }

  private async disconnect(name: string): Promise<void> {
    const conn = this.connections.get(name)
    if (!conn) return
    this.connections.delete(name)
    await conn.close?.().catch(() => undefined)
  }

  private findTool(qualifiedName: string): { conn: Connection; tool: McpToolInfo } | undefined {
    for (const conn of this.connections.values()) {
      const tool = conn.status.tools.find((t) => t.qualifiedName === qualifiedName)
      if (tool) return { conn, tool }
    }
    return undefined
  }

  /** Connected tools the user has not switched off. */
  availableTools(disabled: Set<string>): McpToolInfo[] {
    return [...this.connections.values()]
      .filter((c) => c.status.state === 'connected')
      .flatMap((c) => c.status.tools)
      .filter((t) => !disabled.has(t.qualifiedName))
  }

  /** OpenAI-style tool definitions for enabled tools, optionally limited to `only`. */
  toolDefinitions(
    disabled: Set<string>,
    only?: Set<string>
  ): { type: 'function'; function: { name: string; description: string; parameters: unknown } }[] {
    return this.availableTools(disabled)
      .filter((t) => !only || only.has(t.qualifiedName))
      .map((t) => ({
        type: 'function' as const,
        function: {
          name: t.qualifiedName,
          description: t.description ?? `${t.name} (from ${t.server})`,
          parameters: t.inputSchema ?? { type: 'object', properties: {} }
        }
      }))
  }

  async callTool(call: Pick<WireToolCall['function'], 'name' | 'arguments'>): Promise<ToolResult> {
    const found = this.findTool(call.name)
    if (!found?.conn.client) return { text: `Tool "${call.name}" is not available.`, isError: true }
    let args: Record<string, unknown> = {}
    if (call.arguments.trim()) {
      try {
        args = JSON.parse(call.arguments)
      } catch {
        return { text: `The tool arguments were not valid JSON: ${call.arguments}`, isError: true }
      }
    }
    try {
      const res = await found.conn.client.callTool({ name: found.tool.name, arguments: args }, undefined, {
        timeout: CALL_TIMEOUT_MS
      })
      let text = contentToText(res.content)
      if (!text && res.structuredContent) text = JSON.stringify(res.structuredContent)
      if (text.length > MAX_RESULT_CHARS) text = `${text.slice(0, MAX_RESULT_CHARS)}\n…[truncated ${text.length - MAX_RESULT_CHARS} characters]`
      return { text, isError: !!res.isError }
    } catch (err) {
      return { text: `Tool call failed: ${(err as Error).message}`, isError: true }
    }
  }

  async dispose(): Promise<void> {
    await Promise.all([...this.connections.keys()].map((n) => this.disconnect(n)))
  }

  private emit(): void {
    this.onChange(this.status())
  }

  private emitConfig(): void {
    this.onConfig(this.config())
  }
}
