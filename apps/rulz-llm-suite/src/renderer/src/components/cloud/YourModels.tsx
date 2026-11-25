import { CircleAlert, Trash2 } from 'lucide-react'
import { useState } from 'react'
import { cloudModelId, type CloudModelDef } from '@shared/cloud'
import { CONTEXT_HINT, badgesFor, clampInt, contextBounds, expiresWarning, priceLine, replyBounds } from '../../lib/cloud'
import { formatTokens } from '../../lib/format'
import { isBusy, isChannelBusy, useCloud } from '../../stores/cloud'
import { useModels } from '../../stores/models'
import { ConfirmDialog } from '../remote/ConfirmDialog'
import { Badge, Button, Field, NumberInput, Switch } from '../ui'

export function YourModels({ models, keySaved }: { models: CloudModelDef[]; keySaved: boolean }) {
  return (
    <section className="space-y-3" aria-label="Your models">
      <h2 className="font-[family-name:var(--font-display)] text-sm font-semibold">
        Your models <span className="font-normal text-faint tabular-nums">· {models.length}</span>
      </h2>
      {models.length === 0 ? (
        <div className="rounded-xl border border-border bg-panel p-5 text-sm text-muted">
          No OpenRouter models added yet. Add one from the list below; it then shows up in the model picker.
        </div>
      ) : (
        models.map((m) => <ModelRow key={m.slug} def={m} keySaved={keySaved} />)
      )}
    </section>
  )
}

function ModelRow({ def, keySaved }: { def: CloudModelDef; keySaved: boolean }) {
  const run = useCloud((s) => s.run)
  const removing = useCloud((s) => isBusy(s, 'cloud:removeModel', def.slug))
  const updating = useCloud((s) => isChannelBusy(s, 'cloud:updateModel'))
  const id = cloudModelId(def.slug)
  const instance = useModels((s) => s.instances.find((i) => i.modelId === id && i.status !== 'stopping'))
  const connecting = useModels((s) => s.instances.some((i) => i.modelId === id && i.status === 'loading'))
  const ensure = useModels((s) => s.ensure)
  const unload = useModels((s) => s.unload)
  const [confirm, setConfirm] = useState(false)
  const info = def.info
  const ctx = contextBounds(info)
  const reply = replyBounds(def)
  const expires = expiresWarning(info)

  return (
    <div className="rounded-xl border border-border bg-panel p-5">
      <div className="flex flex-wrap items-start gap-3">
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <span className="truncate text-sm font-medium">{info.name}</span>
            {instance && <Badge tone={instance.status === 'ready' ? 'ok' : 'accent'}>{instance.status === 'ready' ? 'Connected' : 'Connecting'}</Badge>}
            {badgesFor(info).map((b) => (
              <Badge key={b.label} tone={b.tone}>
                {b.label}
              </Badge>
            ))}
          </div>
          <div className="mt-0.5 truncate font-mono text-[11.5px] text-faint">{def.slug}</div>
          <div className="mt-1 text-xs text-muted tabular-nums">
            {priceLine(info)} · context up to {formatTokens(info.contextLength)}
            {info.maxReply ? ` · reply up to ${formatTokens(info.maxReply)}` : ''}
          </div>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          {instance ? (
            <Button size="sm" disabled={instance.status === 'stopping'} onClick={() => void unload(instance.instanceId)}>
              Disconnect
            </Button>
          ) : (
            <Button
              size="sm"
              disabled={!keySaved || connecting}
              title={keySaved ? undefined : 'Save an OpenRouter key first.'}
              onClick={() => void ensure(id)}
            >
              Use
            </Button>
          )}
          <Button size="sm" variant="ghost" disabled={removing} onClick={() => setConfirm(true)}>
            <Trash2 size={14} /> Remove
          </Button>
        </div>
      </div>

      {expires && (
        <div className="mt-3 flex items-center gap-2 rounded-md border border-warn/40 bg-warn/10 px-3 py-2 text-xs text-warn">
          <CircleAlert size={14} className="shrink-0" /> {expires}
        </div>
      )}

      <div className="mt-4 grid grid-cols-1 gap-x-8 gap-y-3 lg:grid-cols-2">
        <Field label="Context limit" hint={`${ctx.min.toLocaleString('en-US')} to ${ctx.max.toLocaleString('en-US')} tokens`}>
          <NumberInput
            className="w-32"
            value={def.contextLimit}
            min={ctx.min}
            max={ctx.max}
            step={1024}
            onChange={(v) => {
              if (v === null) return
              const next = clampInt(v, ctx.min, ctx.max)
              if (next !== def.contextLimit) void run('cloud:updateModel', def.slug, { contextLimit: next })
            }}
          />
        </Field>
        <Field label="Reply limit" hint={`${reply.min.toLocaleString('en-US')} to ${reply.max.toLocaleString('en-US')} tokens`}>
          <NumberInput
            className="w-32"
            value={def.replyLimit}
            min={reply.min}
            max={reply.max}
            step={256}
            onChange={(v) => {
              if (v === null) return
              const next = clampInt(v, reply.min, reply.max)
              if (next !== def.replyLimit) void run('cloud:updateModel', def.slug, { replyLimit: next })
            }}
          />
        </Field>
      </div>
      <p className="mt-2 text-[11px] text-faint">{CONTEXT_HINT} The reply limit is the most one request may write; the cost reserved for a request grows with it.</p>

      <div className="mt-3 border-t border-border pt-2">
        <Field
          label="Allow other apps"
          hint="The API server, RulZ Coding Agent and the phone apps may then use this model and spend your money on it. Off at first."
          inline
        >
          <Switch checked={def.allowApps} disabled={updating} onCheckedChange={(on) => void run('cloud:updateModel', def.slug, { allowApps: on })} />
        </Field>
      </div>

      <ConfirmDialog
        open={confirm}
        title="Remove this model?"
        lines={[
          `${info.name} is removed from your list and disconnected. Nothing is deleted at OpenRouter, and you can add it again later.`,
          'It is refused while it is answering a request.'
        ]}
        confirmLabel="Remove it"
        variant="danger"
        busy={removing}
        onCancel={() => setConfirm(false)}
        onConfirm={() => {
          setConfirm(false)
          void run('cloud:removeModel', def.slug)
        }}
      />
    </div>
  )
}
