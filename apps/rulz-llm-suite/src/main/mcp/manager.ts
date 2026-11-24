import { promises as fs } from 'node:fs'
import { Client } from '@modelcontextprotocol/sdk/client/index.js'
import { getDefaultEnvironment, StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js'
import { StreamableHTTPClientTransport } from '@modelcontextprotocol/sdk/client/streamableHttp.js'
import type { McpConfigState, McpServerConfig, McpServerStatus, McpToolInfo, ToolResult, WireToolCall } from '@shared/types'
import { FILE_TOOL_DEFS, FILE_TOOLS_SERVER, runFileTool } from '../files/fileTools'
import type { FileToolScope } from '@shared/types'
import type { PdfReader } from '../pdf/reader'
import { contentToText, EMPTY_CONFIG, parseMcpConfig, qualifyToolName, qualifyToolNames, readMcpConfigFile } from './config'
import { killTree, killTreesSync, launcherDirs, McpScope, SCOPED_SERVERS, SCOPES_OPEN_MESSAGE, type ScopedProcess } from './scopes'

const MAX_LOG = 200
const MAX_RESULT_CHARS = 24000
const CALL_TIMEOUT_MS = 120_000
/** How long a stdio server gets to exit by itself once its input is closed. */
const EXIT_GRACE_MS = 1500

interface Connection {
  status: McpServerStatus
  client?: Client
  close?: () => Promise<void>
  /** Process id of a stdio server while it runs. */
  pid?: () => number | null
}

/** Connects to the MCP servers in mcp.json and exposes their tools to chats. */
export class McpManager {
  private connections = new Map<string, Connection>()
  private configText = EMPTY_CONFIG
  private configError: string | null = null
  private readFailed = false
  /** Every server name in mcp.json, so tool names can be kept unique across servers. */
  private serverNames: string[] = []
  private disposed = false
  /** Private server processes held by practice lanes; config changes wait until they are closed. */
  private scopes = new Set<McpScope>()
  /** Built-in read-only file tools, always present and handled in-process. */
  private readonly builtin: Connection = {
    status: {
      name: FILE_TOOLS_SERVER,
      transport: 'builtin',
      state: 'connected',
      tools: FILE_TOOL_DEFS.map((t) => ({
        qualifiedName: qualifyToolName(FILE_TOOLS_SERVER, t.name),
        server: FILE_TOOLS_SERVER,
        name: t.name,
        description: t.description,
        inputSchema: t.inputSchema
      })),
      log: []
    }
  }

  constructor(
    private readonly configPath: string,
    private readonly onChange: (status: McpServerStatus[]) => void,
    private readonly onConfig: (config: McpConfigState) => void = () => undefined,
    /** Servers the app provides itself (the RE calculator). A server of the same name in mcp.json wins. */
    private readonly appServers: () => Record<string, McpServerConfig> = () => ({}),
    /** A practice lane that uses the shared REA is opening, or has opened, a program there. */
    private readonly onReaTaken: () => void = () => undefined,
    /** Reads PDFs for the built-in read_pdf tool, off the main thread. Without one the tool answers that it is unavailable. */
    private readonly pdf?: PdfReader
  ) {}

  private withAppServers(servers: Record<string, McpServerConfig>): Record<string, McpServerConfig> {
    return { ...this.appServers(), ...servers }
  }

  private allConnections(): Connection[] {
    return [this.builtin, ...this.connections.values()]
  }

  status(): McpServerStatus[] {
    return this.allConnections().map((c) => ({ ...c.status, log: c.status.log.slice() }))
  }

  config(): McpConfigState {
    return { text: this.configText, path: this.configPath, error: this.configError, readFailed: this.readFailed }
  }

  async init(): Promise<void> {
    await this.reload(null)
  }

  /** Validates and saves mcp.json, then reconnects every server. */
  async saveConfig(text: string): Promise<McpServerStatus[]> {
    this.assertNoScopes()
    const servers = this.withAppServers(parseMcpConfig(text))
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
    this.assertNoScopes()
    await this.reload(name)
  }

  private assertNoScopes(): void {
    if (this.scopes.size) throw new Error(SCOPES_OPEN_MESSAGE)
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
        servers = this.withAppServers(parseMcpConfig(read.text))
      } catch (e) {
        this.configError = (e as Error).message
      }
    }
    this.emitConfig()
    if (!servers) return
    if (name === null) return this.apply(servers)
    this.serverNames = Object.keys(servers)
    const cfg = servers[name]
    await this.disconnect(name)
    if (cfg) await this.connect(name, cfg)
    else this.emit()
  }

  private async apply(servers: Record<string, McpServerConfig>): Promise<void> {
    this.serverNames = Object.keys(servers)
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
    if (name === FILE_TOOLS_SERVER) {
      status.state = 'error'
      status.error = `The name "${FILE_TOOLS_SERVER}" is used by the built-in file tools. Rename this server in mcp.json.`
      this.emit()
      return
    }
    this.emit()
    if (cfg.disabled || this.disposed) return

    const log = (line: string): void => {
      status.log.push(line)
      if (status.log.length > MAX_LOG) status.log.splice(0, status.log.length - MAX_LOG)
    }
    // Restarted, removed or shut down while starting: this connection is no longer the server's.
    const stale = (): boolean => this.disposed || this.connections.get(name) !== conn

    try {
      const client = new Client({ name: 'RulZ LLM Suite', version: '0.2.0' })
      // Known before connecting, so a disconnect while the server is still starting closes it.
      conn.close = () => client.close()
      if (cfg.url) {
        const transport = new StreamableHTTPClientTransport(new URL(cfg.url), {
          requestInit: cfg.headers ? { headers: cfg.headers } : undefined
        })
        await client.connect(transport)
      } else {
        const transport = new StdioClientTransport({
          command: cfg.command!,
          args: cfg.args ?? [],
          env: this.stdioEnv(cfg),
          cwd: cfg.cwd,
          stderr: 'pipe'
        })
        conn.pid = () => transport.pid
        transport.stderr?.on('data', (d: Buffer) => {
          for (const line of d.toString().split(/\r?\n/)) if (line.trim()) log(line)
        })
        await client.connect(transport)
      }
      if (stale()) {
        await this.closeConnection(conn)
        return
      }
      conn.client = client
      client.onclose = () => {
        if (this.connections.get(name) !== conn || status.state !== 'connected') return
        status.state = 'error'
        status.error = 'The server closed the connection.'
        this.emit()
      }

      const listed: { name: string; description?: string; inputSchema: unknown }[] = []
      let cursor: string | undefined
      do {
        const page = await client.listTools(cursor ? { cursor } : undefined)
        listed.push(...page.tools)
        cursor = page.nextCursor
      } while (cursor)
      if (stale()) {
        await this.closeConnection(conn)
        return
      }

      const names = qualifyToolNames(
        name,
        listed.map((t) => t.name),
        [FILE_TOOLS_SERVER, ...this.serverNames]
      )
      const tools: McpToolInfo[] = listed.map((t, i) => ({
        qualifiedName: names[i],
        server: name,
        name: t.name,
        description: t.description,
        inputSchema: t.inputSchema
      }))

      status.tools = tools
      status.state = 'connected'
      status.error = undefined
    } catch (err) {
      if (stale()) {
        await this.closeConnection(conn)
        return
      }
      status.state = 'error'
      status.error = (err as Error).message
      log(`[error] ${(err as Error).message}`)
      await this.closeConnection(conn)
    }
    this.emit()
  }

  /** The environment a stdio server starts with: the SDK's default plus the usual launcher folders on PATH, then the config's own. */
  private stdioEnv(cfg: McpServerConfig): Record<string, string> {
    const base = getDefaultEnvironment()
    const pathKey = Object.keys(base).find((k) => k.toLowerCase() === 'path') ?? 'PATH'
    return { ...base, [pathKey]: [...launcherDirs(), base[pathKey] ?? ''].join(';'), ...cfg.env }
  }

  /**
   * Starts a private copy of a stdio server for a scope: same command, environment and process-tree handling as
   * connect(). It is not a connection of the app (nothing is listed or shown for it).
   */
  private async startScoped(cfg: McpServerConfig): Promise<ScopedProcess> {
    const client = new Client({ name: 'RulZ LLM Suite', version: '0.2.0' })
    const transport = new StdioClientTransport({ command: cfg.command!, args: cfg.args ?? [], env: this.stdioEnv(cfg), cwd: cfg.cwd, stderr: 'pipe' })
    const tail: string[] = []
    transport.stderr?.on('data', (d: Buffer) => {
      for (const line of d.toString().split(/\r?\n/)) if (line.trim()) tail.push(line)
      if (tail.length > 20) tail.splice(0, tail.length - 20)
    })
    let alive = true
    client.onclose = () => {
      alive = false
    }
    try {
      await client.connect(transport)
    } catch (err) {
      alive = false
      const p = transport.pid
      if (process.platform === 'win32' && p) await killTree(p)
      await client.close().catch(() => undefined)
      throw new Error(`${(err as Error).message}${tail.length ? ` (${tail.slice(-2).join(' | ')})` : ''}`)
    }
    const pid = transport.pid
    return {
      callTool: async (toolName, args, opts) =>
        (await client.callTool({ name: toolName, arguments: args }, undefined, opts)) as { content?: unknown; structuredContent?: unknown; isError?: boolean },
      pid: () => (alive ? pid : null),
      alive: () => alive,
      stderrTail: () => tail.slice(),
      close: async () => {
        alive = false
        const closing = client.close().catch(() => undefined)
        if (process.platform === 'win32' && pid) {
          const exited = await Promise.race([closing.then(() => true), new Promise<boolean>((r) => setTimeout(() => r(false), EXIT_GRACE_MS).unref())])
          if (!exited) await killTree(pid)
        }
        await closing
      }
    }
  }

  /**
   * A private set of the servers that keep state per process (REA), for one practice lane. Its tools are the shared
   * ones; calls to them run in processes started for this scope (at the first call), everything else uses the
   * shared connections. While a scope is open, saving or restarting the config is refused. Throws a sentence when
   * REA is not a connected local program.
   */
  openScope(opts: { alone?: boolean } = {}): McpScope {
    if (this.disposed) throw new Error('RulZ LLM Suite is closing.')
    const alone = !!opts.alone
    let configs: Record<string, McpServerConfig> = {}
    try {
      configs = this.withAppServers(parseMcpConfig(this.configText))
    } catch (err) {
      if (!alone) throw new Error(`mcp.json cannot be read: ${(err as Error).message}`)
    }
    const own = new Map<string, McpServerConfig>()
    for (const n of SCOPED_SERVERS) {
      const cfg = configs[n]
      if (!cfg || cfg.disabled) continue
      // A lane that has the program to itself may use the shared connection, whatever kind it is.
      if (cfg.url && alone) continue
      if (cfg.url) throw new Error(`Practice sessions side by side or on a rented instance need "${n}" to be a local program (a "command") in mcp.json; it is an HTTP server there.`)
      own.set(n, cfg)
    }
    if (!alone && (!own.has('rea') || this.connections.get('rea')?.status.state !== 'connected')) {
      throw new Error('REA is not connected. Open the MCP page, fix it, and start the run again.')
    }
    // Alone and REA not connected: the shared connection is used (its tools are simply not there).
    if (alone && this.connections.get('rea')?.status.state !== 'connected') own.clear()
    const scope: McpScope = new McpScope({
      scoped: new Set(own.keys()),
      locate: (qualified) => {
        const found = this.findTool(qualified)
        return found ? { server: found.conn.status.name, tool: found.tool.name } : undefined
      },
      shared: async (call, signal, files) => {
        if (!call.name.startsWith('rea__open_')) return this.callTool(call, signal, files)
        this.onReaTaken()
        try {
          return await this.callTool(call, signal, files)
        } finally {
          this.onReaTaken()
        }
      },
      start: (server) => this.startScoped(own.get(server)!),
      onClose: () => void this.scopes.delete(scope)
    })
    this.scopes.add(scope)
    return scope
  }

  private async disconnect(name: string): Promise<void> {
    const conn = this.connections.get(name)
    if (!conn) return
    this.connections.delete(name)
    await this.closeConnection(conn)
  }

  /**
   * Closes a server. The SDK closes a stdio server's input, waits, then kills only the process it
   * started; on Windows that is often a launcher, so if the server hasn't exited by itself after
   * a short wait, its whole process tree is ended while the launcher still links it together.
   */
  private async closeConnection(conn: Connection): Promise<void> {
    const pid = conn.pid?.() ?? null
    conn.pid = undefined
    const closing = conn.close?.().catch(() => undefined) ?? Promise.resolve()
    if (process.platform === 'win32' && pid) {
      const exited = await Promise.race([
        closing.then(() => true),
        new Promise<boolean>((r) => setTimeout(() => r(false), EXIT_GRACE_MS).unref())
      ])
      if (!exited) await killTree(pid)
    }
    await closing
  }

  private findTool(qualifiedName: string): { conn: Connection; tool: McpToolInfo } | undefined {
    for (const conn of this.allConnections()) {
      const tool = conn.status.tools.find((t) => t.qualifiedName === qualifiedName)
      if (tool) return { conn, tool }
    }
    return undefined
  }

  /** Connected tools the user has not switched off. */
  availableTools(disabled: Set<string>): McpToolInfo[] {
    return this.allConnections()
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

  /** Runs a tool. Aborting `signal` stops the call and tells the server it was cancelled. */
  async callTool(call: Pick<WireToolCall['function'], 'name' | 'arguments'>, signal?: AbortSignal, files?: FileToolScope): Promise<ToolResult> {
    const stopped: ToolResult = { text: 'The tool call was stopped.', isError: true }
    if (signal?.aborted) return stopped
    const found = this.findTool(call.name)
    if (found?.conn === this.builtin) {
      let args: Record<string, unknown> = {}
      try {
        args = call.arguments.trim() ? JSON.parse(call.arguments) : {}
      } catch {
        return { text: `The tool arguments were not valid JSON: ${call.arguments}`, isError: true }
      }
      const res = await runFileTool(found.tool.name, args, signal, files, { pdf: this.pdf })
      if (signal?.aborted) return stopped
      return res.text.length > MAX_RESULT_CHARS
        ? { ...res, text: `${res.text.slice(0, MAX_RESULT_CHARS)}\n…[truncated ${res.text.length - MAX_RESULT_CHARS} characters]` }
        : res
    }
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
        timeout: CALL_TIMEOUT_MS,
        signal
      })
      let text = contentToText(res.content)
      if (!text && res.structuredContent) text = JSON.stringify(res.structuredContent)
      if (text.length > MAX_RESULT_CHARS) text = `${text.slice(0, MAX_RESULT_CHARS)}\n…[truncated ${text.length - MAX_RESULT_CHARS} characters]`
      return { text, isError: !!res.isError }
    } catch (err) {
      if (signal?.aborted) return stopped
      return { text: `Tool call failed: ${(err as Error).message}`, isError: true }
    }
  }

  async dispose(): Promise<void> {
    this.disposed = true
    // The app quits without waiting for this, so a polite close would never finish: end the
    // stdio servers' process trees now, synchronously, so none outlive the app.
    const pids: number[] = []
    for (const scope of this.scopes) scope.closeSync()
    this.scopes.clear()
    for (const conn of this.connections.values()) {
      const pid = conn.pid?.()
      if (pid) pids.push(pid)
      conn.pid = undefined
    }
    killTreesSync(pids)
    await Promise.all([...this.connections.keys()].map((n) => this.disconnect(n)))
  }

  private emit(): void {
    this.onChange(this.status())
  }

  private emitConfig(): void {
    this.onConfig(this.config())
  }
}
