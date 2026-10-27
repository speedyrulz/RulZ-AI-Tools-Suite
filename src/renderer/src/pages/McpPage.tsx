import * as Collapsible from '@radix-ui/react-collapsible'
import { ChevronRight, FileText, FolderOpen, Plus, RefreshCw, ShieldCheck, TriangleAlert } from 'lucide-react'
import { useEffect, useState } from 'react'
import { toast } from 'sonner'
import { toolTokens } from '@shared/tools'
import type { McpServerConfig, McpServerStatus } from '@shared/types'
import { Badge, Button, DropdownMenu, Field, IconButton, Input, NumberInput, Select, Switch, Tooltip } from '../components/ui'
import { errorMessage, invoke } from '../lib/api'
import { cn, formatCompact } from '../lib/format'
import { useApp } from '../stores/app'
import { useMcp } from '../stores/mcp'

const EXAMPLES: { label: string; name: string; config: McpServerConfig }[] = [
  {
    label: 'Filesystem (read/write a folder)',
    name: 'filesystem',
    config: { command: 'npx', args: ['-y', '@modelcontextprotocol/server-filesystem', 'C:\\path\\to\\folder'] }
  },
  { label: 'Memory (knowledge graph)', name: 'memory', config: { command: 'npx', args: ['-y', '@modelcontextprotocol/server-memory'] } },
  { label: 'Fetch web pages (needs uv)', name: 'fetch', config: { command: 'uvx', args: ['mcp-server-fetch'] } },
  { label: 'Remote server (HTTP)', name: 'remote', config: { url: 'https://example.com/mcp', headers: { Authorization: 'Bearer YOUR_TOKEN' } } }
]

const STATE_TONE = { connected: 'ok', connecting: 'accent', error: 'danger', disabled: 'neutral' } as const

/** Above this many tokens of tool descriptions per message, small contexts start to suffer. */
const TOOL_TOKEN_WARNING = 8000

export function McpPage() {
  const servers = useMcp((s) => s.servers)
  const config = useMcp((s) => s.config)
  const settings = useApp((s) => s.settings)
  const update = useApp((s) => s.updateSettings)
  const [text, setText] = useState<string | null>(null)
  const [saved, setSaved] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)
  const [reconnecting, setReconnecting] = useState(false)

  // Follow the file as the app reads it (startup, Reconnect all), unless you have unsaved edits.
  const loadedText = config?.text ?? null
  useEffect(() => {
    if (loadedText === null) return
    setText((cur) => (cur === null || cur === saved ? loadedText : cur))
    setSaved(loadedText)
  }, [loadedText])

  if (!settings || !config || text === null) return null
  const path = config.path
  const dirty = text !== saved

  const reconnectAll = async (): Promise<void> => {
    setReconnecting(true)
    try {
      await invoke('mcp:restart', null)
    } catch (e) {
      toast.error(errorMessage(e))
    } finally {
      setReconnecting(false)
    }
  }
  const disabledSet = new Set(settings.disabledTools)
  const enabled = servers.filter((s) => s.state === 'connected').flatMap((s) => s.tools).filter((t) => !disabledSet.has(t.qualifiedName))
  const toolCount = enabled.length
  const totalTokens = enabled.reduce((sum, t) => sum + toolTokens(t), 0)

  const save = async (): Promise<void> => {
    if (
      config.readFailed &&
      !window.confirm("mcp.json couldn't be read, so this editor may not show what is in the file. Replace the file with the editor's contents?")
    )
      return
    setSaving(true)
    try {
      await invoke('mcp:saveConfig', text)
      setSaved(text)
      setError(null)
      toast.success('Saved mcp.json and reconnected servers')
    } catch (e) {
      setError(errorMessage(e))
    } finally {
      setSaving(false)
    }
  }

  const addExample = (ex: (typeof EXAMPLES)[number]): void => {
    try {
      const parsed = JSON.parse(text) as { mcpServers?: Record<string, McpServerConfig> }
      const servers = parsed.mcpServers ?? {}
      let name = ex.name
      for (let i = 2; servers[name]; i++) name = `${ex.name}-${i}`
      servers[name] = ex.config
      setText(JSON.stringify({ ...parsed, mcpServers: servers }, null, 2) + '\n')
      setError(null)
    } catch {
      setError('Fix the JSON before adding an example.')
    }
  }

  return (
    <div className="flex h-full">
      <div className="flex min-w-0 flex-1 flex-col">
        <div className="flex items-center gap-3 border-b border-border px-6 py-4">
          <div className="min-w-0 flex-1">
            <h1 className="text-base font-semibold">MCP servers</h1>
            <div className="text-xs text-muted">
              {servers.length} {servers.length === 1 ? 'server' : 'servers'} · {toolCount} {toolCount === 1 ? 'tool' : 'tools'} available to chats ·{' '}
              <span className={totalTokens > TOOL_TOKEN_WARNING ? 'text-warn' : undefined}>
                ≈{formatCompact(totalTokens)} tokens of tool descriptions per message
              </span>
            </div>
          </div>
          <Tooltip content="Re-reads mcp.json from disk and reconnects every server">
            <Button size="sm" disabled={reconnecting} onClick={() => void reconnectAll()}>
              <RefreshCw size={14} className={reconnecting ? 'animate-spin' : undefined} /> Reconnect all
            </Button>
          </Tooltip>
        </div>

        <div className="min-h-0 flex-1 space-y-4 overflow-y-auto px-6 py-5">
          <div className="space-y-1 rounded-xl border border-border bg-panel p-4">
            <Field inline label="Use MCP tools in chats" hint="The model can call enabled tools while answering">
              <Switch checked={settings.toolsEnabled} onCheckedChange={(v) => void update({ toolsEnabled: v })} />
            </Field>
            <Field
              inline
              label="Tool loading"
              hint={
                settings.toolMode === 'all'
                  ? 'Every enabled tool is described in every message'
                  : settings.toolMode === 'dynamic'
                    ? 'The model searches for and loads only the tools it needs'
                    : `All tools while they fit the budget, on demand beyond it (now: ${totalTokens > settings.toolBudgetTokens ? 'on demand' : 'all'})`
              }
            >
              <Select
                className="w-40"
                value={settings.toolMode}
                onChange={(e) => void update({ toolMode: e.target.value as typeof settings.toolMode })}
              >
                <option value="auto">Auto</option>
                <option value="all">All enabled tools</option>
                <option value="dynamic">On demand</option>
              </Select>
            </Field>
            <Field inline label="Tool budget" hint="Most tokens of tool descriptions loaded at once; the least recently used are unloaded first">
              <NumberInput
                className="w-28"
                value={settings.toolBudgetTokens}
                min={500}
                step={500}
                onChange={(v) => v !== null && v >= 500 && void update({ toolBudgetTokens: Math.round(v) })}
              />
            </Field>
            <Field
              inline
              label="Shorten older tool results"
              hint="Sends older results as their beginning and end only; the chat still shows them in full"
            >
              <Switch checked={settings.compactToolResults} onCheckedChange={(v) => void update({ compactToolResults: v })} />
            </Field>
            {settings.compactToolResults && (
              <Field
                inline
                label="Keep recent results in full"
                hint="How many of the latest results in the current turn stay complete; earlier turns are always shortened"
              >
                <NumberInput
                  className="w-20"
                  value={settings.keepFullToolResults}
                  min={0}
                  max={100}
                  onChange={(v) => v !== null && v >= 0 && void update({ keepFullToolResults: Math.round(v) })}
                />
              </Field>
            )}
            <Field inline label="Ask before running a tool" hint="Tools you mark “always allow” skip the question">
              <Switch checked={settings.confirmToolCalls} onCheckedChange={(v) => void update({ confirmToolCalls: v })} />
            </Field>
            {settings.allowedTools.length > 0 && (
              <div className="pt-2">
                <div className="mb-1.5 text-[11px] text-faint">Always allowed</div>
                <div className="flex flex-wrap gap-1.5">
                  {settings.allowedTools.map((t) => (
                    <button
                      key={t}
                      onClick={() => void update({ allowedTools: settings.allowedTools.filter((x) => x !== t) })}
                      className="flex items-center gap-1 rounded border border-border bg-panel-2 px-2 py-0.5 font-mono text-[11px] hover:border-danger/50 hover:text-danger"
                      title="Click to require approval again"
                    >
                      <ShieldCheck size={11} /> {t} ✕
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>

          {config.error && (
            <div className="flex gap-3 rounded-xl border border-danger/40 bg-danger/10 p-4 text-sm">
              <TriangleAlert size={16} className="mt-0.5 shrink-0 text-danger" />
              <div className="min-w-0 space-y-1">
                <div className="font-medium text-danger">
                  {config.readFailed ? "Couldn't read mcp.json" : 'mcp.json has a problem'}
                </div>
                <div className="text-xs whitespace-pre-wrap text-muted">{config.error}</div>
                <div className="text-xs text-muted">
                  {config.readFailed
                    ? 'Servers that were already connected keep running. Fix the cause, then press Reconnect all to read the file again.'
                    : 'Fix it in the editor and save, or edit the file and press Reconnect all.'}
                </div>
              </div>
            </div>
          )}
          {servers.length === 0 && !config.error && (
            <div className="rounded-xl border border-dashed border-border p-8 text-center text-sm text-muted">
              No MCP servers yet. Add one to <span className="font-mono">mcp.json</span> on the right, then save.
            </div>
          )}
          {servers.map((s) => (
            <ServerCard key={s.name} server={s} />
          ))}
        </div>
      </div>

      <aside className="flex w-[460px] shrink-0 flex-col border-l border-border bg-panel">
        <div className="flex items-center gap-2 border-b border-border px-4 py-3">
          <FileText size={15} className="text-muted" />
          <span className="text-sm font-medium">mcp.json</span>
          {dirty && <Badge tone="warn">Unsaved</Badge>}
          <div className="flex-1" />
          <DropdownMenu
            trigger={
              <Button size="sm" variant="ghost">
                <Plus size={13} /> Add example
              </Button>
            }
            items={EXAMPLES.map((ex) => ({ label: ex.label, onSelect: () => addExample(ex) }))}
          />
          <IconButton size="sm" label="Show file in Explorer" onClick={() => void invoke('shell:showItem', path)}>
            <FolderOpen size={14} />
          </IconButton>
        </div>
        <textarea
          spellCheck={false}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.ctrlKey && e.key.toLowerCase() === 's') {
              e.preventDefault()
              void save()
            }
          }}
          className="min-h-0 flex-1 resize-none bg-code p-4 font-mono text-[12.5px] leading-relaxed focus:outline-none focus-visible:outline-none"
        />
        {error && <div className="border-t border-danger/40 bg-danger/10 px-4 py-2 text-xs whitespace-pre-wrap text-danger">{error}</div>}
        <div className="flex items-center gap-2 border-t border-border px-4 py-3">
          <span className="text-[11px] text-faint">Same format as LM Studio and Cursor. Ctrl+S saves.</span>
          <div className="flex-1" />
          <Button size="sm" variant="ghost" disabled={!dirty} onClick={() => setText(saved)}>
            Revert
          </Button>
          <Button size="sm" variant="primary" disabled={saving} onClick={() => void save()}>
            Save & reconnect
          </Button>
        </div>
      </aside>
    </div>
  )
}

function ServerCard({ server }: { server: McpServerStatus }) {
  const settings = useApp((s) => s.settings)!
  const update = useApp((s) => s.updateSettings)
  const [showLog, setShowLog] = useState(false)
  const [filter, setFilter] = useState('')
  const disabled = new Set(settings.disabledTools)
  const enabledTools = server.tools.filter((t) => !disabled.has(t.qualifiedName))
  const enabledTokens = enabledTools.reduce((sum, t) => sum + toolTokens(t), 0)
  const q = filter.trim().toLowerCase()
  const visibleTools = q
    ? server.tools.filter((t) => t.name.toLowerCase().includes(q) || t.description?.toLowerCase().includes(q))
    : server.tools

  const toggleTool = (name: string, on: boolean): void => {
    const next = on ? settings.disabledTools.filter((t) => t !== name) : [...settings.disabledTools, name]
    void update({ disabledTools: next })
  }

  const setAll = (on: boolean): void => {
    const mine = new Set(server.tools.map((t) => t.qualifiedName))
    const others = settings.disabledTools.filter((t) => !mine.has(t))
    void update({ disabledTools: on ? others : [...others, ...mine] })
  }

  return (
    <div className="rounded-xl border border-border bg-panel">
      <div className="flex items-center gap-2 px-4 py-3">
        <span className="font-medium">{server.name}</span>
        <Badge>{server.transport}</Badge>
        <Badge tone={STATE_TONE[server.state]}>{server.state}</Badge>
        <span className="text-xs text-faint">{server.state === 'connected' ? `${server.tools.length} tools` : ''}</span>
        <div className="flex-1" />
        {server.log.length > 0 && (
          <Button size="sm" variant="ghost" onClick={() => setShowLog((v) => !v)}>
            {showLog ? 'Hide log' : 'Log'}
          </Button>
        )}
        <IconButton size="sm" label="Reconnect" onClick={() => void invoke('mcp:restart', server.name)}>
          <RefreshCw size={13} />
        </IconButton>
      </div>
      {server.error && <div className="mx-4 mb-3 rounded-md border border-danger/40 bg-danger/10 px-3 py-2 text-xs whitespace-pre-wrap text-danger">{server.error}</div>}
      {showLog && (
        <pre className="mx-4 mb-3 max-h-48 overflow-auto rounded-md bg-code p-2 font-mono text-[11px] whitespace-pre-wrap text-muted">{server.log.join('\n')}</pre>
      )}
      {server.tools.length > 0 && (
        <Collapsible.Root defaultOpen={server.tools.length <= 8} className="border-t border-border">
          <div className="flex items-center gap-2 px-4 py-2">
            <Collapsible.Trigger className="group flex items-center gap-1 text-xs text-muted hover:text-text">
              <ChevronRight size={13} className="transition-transform group-data-[state=open]:rotate-90" />
              Tools · {enabledTools.length} of {server.tools.length} enabled
            </Collapsible.Trigger>
            <Tooltip content="Tool descriptions are sent with every message and use context">
              <span className={cn('text-[11px]', enabledTokens > TOOL_TOKEN_WARNING ? 'text-warn' : 'text-faint')}>
                ≈{formatCompact(enabledTokens)} tokens per message
              </span>
            </Tooltip>
            <div className="flex-1" />
            <Button size="sm" variant="ghost" onClick={() => setAll(true)}>
              Enable all
            </Button>
            <Button size="sm" variant="ghost" onClick={() => setAll(false)}>
              Disable all
            </Button>
          </div>
          <Collapsible.Content className="space-y-1 px-4 pb-3">
            {server.tools.length > 8 && (
              <Input className="mb-2 h-8" placeholder={`Filter ${server.tools.length} tools`} value={filter} onChange={(e) => setFilter(e.target.value)} />
            )}
            {visibleTools.map((t) => (
              <div key={t.qualifiedName} className="flex items-start gap-3 rounded-md px-2 py-1.5 hover:bg-hover/40">
                <div className="min-w-0 flex-1">
                  <div className={cn('font-mono text-[12px]', disabled.has(t.qualifiedName) && 'text-faint line-through')}>
                    {t.name}
                    {settings.allowedTools.includes(t.qualifiedName) && (
                      <Tooltip content="Runs without asking">
                        <ShieldCheck size={12} className="ml-1.5 inline text-ok" />
                      </Tooltip>
                    )}
                  </div>
                  {t.description && <div className="line-clamp-2 text-[11.5px] text-muted">{t.description}</div>}
                </div>
                <Switch checked={!disabled.has(t.qualifiedName)} onCheckedChange={(v) => toggleTool(t.qualifiedName, v)} />
              </div>
            ))}
          </Collapsible.Content>
        </Collapsible.Root>
      )}
    </div>
  )
}
