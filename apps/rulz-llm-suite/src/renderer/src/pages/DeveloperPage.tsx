import { Check, Copy, Eraser, LoaderCircle, Play, RefreshCw, Square } from 'lucide-react'
import { useEffect, useRef, useState, type ReactNode } from 'react'
import type { ApiRequestLog, ApiServerSettings } from '@shared/types'
import { Badge, Button, Field, IconButton, Input, NumberInput, Select, Switch, Tooltip } from '../components/ui'
import { cn, formatDuration, relativeTime } from '../lib/format'
import { useApp } from '../stores/app'
import { useModels } from '../stores/models'
import { useServer } from '../stores/server'

function Card({ title, children, action }: { title: string; children: ReactNode; action?: ReactNode }) {
  return (
    <section className="rounded-xl border border-border bg-panel">
      <div className="flex items-center justify-between border-b border-border px-4 py-2.5">
        <h2 className="text-[11px] font-semibold tracking-wide text-faint uppercase">{title}</h2>
        {action}
      </div>
      <div className="p-4">{children}</div>
    </section>
  )
}

function CopyButton({ text }: { text: string }) {
  const [done, setDone] = useState(false)
  return (
    <IconButton
      size="sm"
      label={done ? 'Copied' : 'Copy'}
      onClick={() => {
        void navigator.clipboard.writeText(text)
        setDone(true)
        setTimeout(() => setDone(false), 1200)
      }}
    >
      {done ? <Check size={13} /> : <Copy size={13} />}
    </IconButton>
  )
}

function Snippet({ title, code }: { title: string; code: string }) {
  return (
    <div className="overflow-hidden rounded-lg border border-border bg-code">
      <div className="flex items-center justify-between border-b border-border px-3 py-1 text-[11px] text-faint">
        {title}
        <CopyButton text={code} />
      </div>
      <pre className="overflow-x-auto p-3 font-mono text-[11.5px] leading-relaxed">{code}</pre>
    </div>
  )
}

function newApiKey(): string {
  const bytes = crypto.getRandomValues(new Uint8Array(24))
  return 'rulz-' + Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('')
}

export function DeveloperPage() {
  const settings = useApp((s) => s.settings)
  const update = useApp((s) => s.updateSettings)
  const models = useModels((s) => s.models)
  const instances = useModels((s) => s.instances)
  const { status, log, busy, start, stop, clearLog } = useServer()
  const [draftPort, setDraftPort] = useState<number | null>(settings?.apiServer.port ?? 1234)

  useEffect(() => {
    setDraftPort(settings?.apiServer.port ?? 1234)
  }, [settings?.apiServer.port])
  if (!settings || !status) return null

  const s = settings.apiServer
  const set = <K extends keyof ApiServerSettings>(key: K, value: ApiServerSettings[K]): void => {
    void update({ apiServer: { ...s, [key]: value } })
  }
  const running = status.running
  const base = `http://${s.bind === 'lan' ? 'localhost' : '127.0.0.1'}:${status.running ? status.port : s.port}`
  const exampleModel = (() => {
    const loaded = instances.find((i) => i.status === 'ready')
    const m = models.find((x) => x.id === (loaded?.modelId ?? s.defaultModelId)) ?? models[0]
    return m?.name ?? 'your-model'
  })()
  const auth = s.apiKey ? ` \\\n  -H "Authorization: Bearer ${s.apiKey}"` : ''
  const curlOpenAI = `curl ${base}/v1/chat/completions \\\n  -H "Content-Type: application/json"${auth} \\\n  -d '{"model":"${exampleModel}","messages":[{"role":"user","content":"Hello!"}],"stream":true}'`
  const curlAnthropic = `curl ${base}/v1/messages \\\n  -H "Content-Type: application/json" -H "anthropic-version: 2023-06-01"${s.apiKey ? ` \\\n  -H "x-api-key: ${s.apiKey}"` : ''} \\\n  -d '{"model":"${exampleModel}","max_tokens":256,"messages":[{"role":"user","content":"Hello!"}]}'`
  const python = `from openai import OpenAI\nclient = OpenAI(base_url="${base}/v1", api_key="${s.apiKey || 'not-needed'}")\nr = client.chat.completions.create(model="${exampleModel}", messages=[{"role": "user", "content": "Hello!"}])\nprint(r.choices[0].message.content)`

  const serverLoaded = new Set(status.jitInstances.map((j) => j.instanceId))

  return (
    <div className="flex h-full">
      <div className="min-h-0 flex-1 space-y-4 overflow-y-auto px-6 py-5">
        <div className="flex items-center gap-3">
          <div className="min-w-0 flex-1">
            <h1 className="text-base font-semibold">Developer</h1>
            <div className="text-xs text-muted">Local OpenAI- and Anthropic-compatible API in front of your models.</div>
          </div>
          <div className="flex items-center gap-2">
            <span className={cn('size-2 rounded-full', running ? 'bg-ok' : 'bg-faint/50')} />
            <span className="text-sm">
              {running ? (
                <>
                  Running at <span className="font-mono">{status.host}:{status.port}</span>
                </>
              ) : (
                'Stopped'
              )}
            </span>
          </div>
          <Button variant={running ? 'secondary' : 'primary'} disabled={busy} onClick={() => void (running ? stop() : start())}>
            {busy ? <LoaderCircle size={14} className="animate-spin" /> : running ? <Square size={13} /> : <Play size={13} />}
            {running ? 'Stop server' : 'Start server'}
          </Button>
        </div>
        {status.error && <div className="rounded-md border border-danger/40 bg-danger/10 px-3 py-2 text-xs text-danger">{status.error}</div>}

        <Card title="Server settings">
          <div className="space-y-1">
            <Field inline label="Start the server when the app starts">
              <Switch checked={s.autoStart} onCheckedChange={(v) => set('autoStart', v)} />
            </Field>
            <Field inline label="Port" hint={running ? 'Restart the server to apply' : 'LM Studio also uses 1234'}>
              <NumberInput
                className="w-24"
                value={draftPort}
                min={1}
                max={65535}
                onChange={setDraftPort}
              />
            </Field>
            {draftPort !== null && draftPort !== s.port && (
              <div className="flex justify-end">
                <Button size="sm" onClick={() => set('port', Math.max(1, Math.min(65535, Math.round(draftPort))))}>
                  Apply port
                </Button>
              </div>
            )}
            <Field inline label="Reachable from" hint={s.bind === 'lan' ? 'Other devices on your network can use it; set an API key' : 'This computer only'}>
              <Select className="w-44" value={s.bind} onChange={(e) => set('bind', e.target.value as 'localhost' | 'lan')}>
                <option value="localhost">This computer only</option>
                <option value="lan">Local network</option>
              </Select>
            </Field>
            <Field label="API key" hint="Empty = no authentication. Sent as Authorization: Bearer or x-api-key.">
              <div className="flex gap-1.5">
                <Input className="font-mono text-xs" value={s.apiKey} placeholder="None" onChange={(e) => set('apiKey', e.target.value.trim())} />
                <Button onClick={() => set('apiKey', newApiKey())}>
                  <RefreshCw size={13} /> Generate
                </Button>
              </div>
            </Field>
            <Field inline label="Allow browser apps (CORS)" hint="Lets web pages on any origin call the API">
              <Switch checked={s.cors} onCheckedChange={(v) => set('cors', v)} />
            </Field>
          </div>
        </Card>

        <Card title="Model loading">
          <div className="space-y-1">
            <Field inline label="Load models on request" hint="A request for a model that isn't loaded loads it with the model's saved settings">
              <Badge tone="ok">Always on</Badge>
            </Field>
            <Field inline label="Unload after idle (minutes)" hint="Applies to models the server loaded itself; 0 = never">
              <NumberInput
                className="w-24"
                value={s.idleUnloadMinutes}
                min={0}
                onChange={(v) => v !== null && v >= 0 && set('idleUnloadMinutes', Math.round(v))}
              />
            </Field>
            <Field label="Default model" hint="Used when a request names no model and nothing is loaded">
              <Select value={s.defaultModelId} onChange={(e) => set('defaultModelId', e.target.value)}>
                <option value="">None (use the loaded model)</option>
                {models.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.name}
                  </option>
                ))}
              </Select>
            </Field>
            {instances.length > 0 && (
              <div className="pt-2">
                <div className="mb-1.5 text-[11px] text-faint">Loaded now</div>
                <div className="space-y-1">
                  {instances.map((i) => {
                    const jit = status.jitInstances.find((j) => j.instanceId === i.instanceId)
                    return (
                      <div key={i.instanceId} className="flex items-center gap-2 text-xs">
                        <span className="truncate font-medium">{i.modelName}</span>
                        <Badge tone={i.status === 'ready' ? 'ok' : 'accent'}>{i.status}</Badge>
                        {serverLoaded.has(i.instanceId) ? (
                          <Tooltip content="Loaded by an API request; unloads when idle">
                            <Badge tone="accent">server</Badge>
                          </Tooltip>
                        ) : (
                          <Tooltip content="Loaded from the app; stays loaded">
                            <Badge>app</Badge>
                          </Tooltip>
                        )}
                        {jit && <span className="text-faint">used {relativeTime(jit.lastUsedAt)}</span>}
                      </div>
                    )
                  })}
                </div>
              </div>
            )}
          </div>
        </Card>

        <Card title="Examples">
          <div className="space-y-3">
            <Snippet title="curl · OpenAI chat completions (streaming)" code={curlOpenAI} />
            <Snippet title="curl · Anthropic messages" code={curlAnthropic} />
            <Snippet title="Python · openai package" code={python} />
            <div className="text-[11px] text-faint">
              Endpoints: /v1/models, /v1/chat/completions, /v1/completions, /v1/embeddings, /v1/responses, /v1/messages, plus llama.cpp's /tokenize, /detokenize,
              /apply-template and /props. The model id is the name shown in My Models.
            </div>
          </div>
        </Card>
      </div>

      <RequestLog log={log} onClear={() => void clearLog()} />
    </div>
  )
}

function RequestLog({ log, onClear }: { log: ApiRequestLog[]; onClear: () => void }) {
  const bottom = useRef<HTMLDivElement>(null)
  const [selected, setSelected] = useState<string | null>(null)
  // Block body: scrollIntoView returns a Promise in current Chromium, which React would treat as a cleanup.
  useEffect(() => {
    bottom.current?.scrollIntoView({ block: 'end' })
  }, [log.length])
  const sel = log.find((e) => e.id === selected)
  return (
    <aside className="flex w-[440px] shrink-0 flex-col border-l border-border bg-panel">
      <div className="flex items-center gap-2 border-b border-border px-4 py-3">
        <span className="text-sm font-medium">Requests</span>
        <span className="text-xs text-faint">{log.length}</span>
        <div className="flex-1" />
        <IconButton size="sm" label="Clear" onClick={onClear}>
          <Eraser size={14} />
        </IconButton>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto font-mono text-[11px]">
        {log.length === 0 && <div className="p-4 text-faint">No requests yet.</div>}
        {log.map((e) => (
          <button
            key={e.id}
            onClick={() => setSelected(e.id === selected ? null : e.id)}
            className={cn('flex w-full items-center gap-2 border-b border-border/60 px-3 py-1.5 text-left hover:bg-hover/50', selected === e.id && 'bg-hover')}
          >
            <span className="w-14 shrink-0 text-faint">{new Date(e.at).toLocaleTimeString([], { hour12: false })}</span>
            <span
              className={cn(
                'w-8 shrink-0',
                e.phase !== 'done' ? 'text-accent' : (e.status ?? 0) >= 400 ? 'text-danger' : 'text-ok'
              )}
            >
              {e.phase === 'loading' ? 'load' : e.phase === 'running' ? '…' : e.status}
            </span>
            <span className="min-w-0 flex-1 truncate">
              {e.method} {e.path}
              {e.model && <span className="text-faint"> · {e.model}</span>}
            </span>
            <span className="shrink-0 text-faint">
              {e.completionTokens !== undefined && `${e.completionTokens} tok `}
              {e.tokensPerSecond !== undefined && `${e.tokensPerSecond.toFixed(1)}/s `}
              {e.durationMs !== undefined && formatDuration(e.durationMs)}
            </span>
          </button>
        ))}
        <div ref={bottom} />
      </div>
      {sel && (
        <div className="max-h-56 space-y-1 overflow-y-auto border-t border-border px-4 py-3 text-xs">
          <div>
            <span className="text-faint">Request</span> {sel.method} {sel.path} {sel.stream ? '(stream)' : ''}
          </div>
          {sel.model && (
            <div>
              <span className="text-faint">Model</span> {sel.model}
            </div>
          )}
          {sel.status !== undefined && (
            <div>
              <span className="text-faint">Status</span> {sel.status} · {formatDuration(sel.durationMs)}
            </div>
          )}
          {sel.promptTokens !== undefined && (
            <div>
              <span className="text-faint">Tokens</span> {sel.promptTokens} prompt · {sel.completionTokens ?? 0} completion
              {sel.tokensPerSecond !== undefined && ` · ${sel.tokensPerSecond.toFixed(1)} tok/s`}
            </div>
          )}
          {sel.error && <div className="whitespace-pre-wrap text-danger">{sel.error}</div>}
        </div>
      )}
    </aside>
  )
}
