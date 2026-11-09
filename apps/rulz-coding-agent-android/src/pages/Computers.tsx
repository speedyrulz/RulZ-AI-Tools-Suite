import { ChevronDown, Laptop, Plus, QrCode, Trash2 } from 'lucide-react'
import { useState } from 'react'
import { linkProblem } from '../components/LinkStatus'
import { LogoWordmark } from '../components/Logo'
import { Button, Field, IconButton, Input, Sheet, Spinner } from '../components/ui'
import { cleanHost, DEFAULT_PORT, parsePairingLink, validPort, type Computer } from '../lib/computers'
import { cn } from '../lib/format'
import { link } from '../lib/remote'
import { useLink } from '../stores/link'

/** The computers this phone has paired with, and how to add one. */
export function Computers() {
  const computers = useLink((s) => s.computers)
  const select = useLink((s) => s.select)
  const remove = useLink((s) => s.remove)
  const [adding, setAdding] = useState(computers.length === 0)
  const [forget, setForget] = useState<Computer | null>(null)

  return (
    <div className="pt-safe pb-safe px-safe h-full overflow-y-auto">
      <div className="mx-auto max-w-lg space-y-6 px-4 py-8">
        <div className="text-center">
          <LogoWordmark className="mx-auto h-12" />
          <p className="mt-3 text-[15px] text-muted">Watch and control RulZ Coding Agent on your computer from here.</p>
        </div>

        {computers.length > 0 && (
          <section className="rounded-2xl border border-border bg-panel p-2">
            <h2 className="px-3 pt-2 pb-1 text-[13px] font-medium tracking-wide text-faint uppercase">Your computers</h2>
            {computers.map((c) => (
              <div key={c.id} className="flex items-center">
                <button className="flex min-w-0 flex-1 items-center gap-3 rounded-xl px-3 py-3 text-left active:bg-hover" onClick={() => select(c.id)}>
                  <Laptop size={20} className="shrink-0 text-accent" />
                  <span className="min-w-0">
                    <span className="block truncate text-[16px] font-medium">{c.name}</span>
                    <span className="block truncate text-[13px] text-faint">
                      {c.host}:{c.port}
                    </span>
                  </span>
                </button>
                <IconButton label={`Forget ${c.name}`} onClick={() => setForget(c)}>
                  <Trash2 size={17} />
                </IconButton>
              </div>
            ))}
          </section>
        )}

        {adding ? (
          <AddComputer onDone={() => setAdding(false)} />
        ) : (
          <Button size="lg" className="w-full" onClick={() => setAdding(true)}>
            <Plus size={18} /> Add computer
          </Button>
        )}
      </div>

      <Sheet
        open={!!forget}
        onOpenChange={(o) => !o && setForget(null)}
        title={`Forget ${forget?.name}?`}
        footer={
          <>
            <Button onClick={() => setForget(null)}>Cancel</Button>
            <Button
              variant="danger"
              onClick={() => {
                void remove(forget!.id)
                setForget(null)
              }}
            >
              Forget
            </Button>
          </>
        }
      >
        <p className="px-4 pb-4 text-[15px] text-muted">To connect again later, scan the QR code in RulZ Coding Agent's Settings on the computer.</p>
      </Sheet>
    </div>
  )
}

function AddComputer({ onDone }: { onDone: () => void }) {
  const add = useLink((s) => s.add)
  const [manual, setManual] = useState(false)
  const [linkText, setLinkText] = useState('')
  const [host, setHost] = useState('')
  const [port, setPort] = useState(String(DEFAULT_PORT))
  const [token, setToken] = useState('')
  const [name, setName] = useState('')
  const [error, setError] = useState('')

  const fromLink = (): void => {
    const parsed = parsePairingLink(linkText)
    if (!parsed) return setError('That isn’t a pairing link. It starts with rulzagent://pair')
    void add(parsed)
    onDone()
  }
  const fromFields = (): void => {
    const h = cleanHost(host)
    const p = Number(port)
    if (!h) return setError('Type the computer’s Tailscale address, like 100.88.114.32.')
    if (!validPort(p)) return setError('The port is a number, usually 7420.')
    if (!token.trim()) return setError('Type the pairing code shown on the computer.')
    void add({ name: name.trim() || h.split('.')[0], host: h, port: p, token: token.trim() })
    onDone()
  }

  return (
    <section className="space-y-4 rounded-2xl border border-border bg-panel p-4">
      <h2 className="flex items-center gap-2 text-[17px] font-semibold">
        <QrCode size={20} className="text-accent" /> Add a computer
      </h2>
      <ol className="list-decimal space-y-1.5 pl-5 text-[15px] text-muted">
        <li>Make sure Tailscale is on, on this phone and on the computer.</li>
        <li>
          On the computer, open RulZ Coding Agent, then <b className="text-text">Settings</b> and turn on <b className="text-text">Phone access</b>.
        </li>
        <li>Scan the QR code it shows with this phone's camera and tap the link.</li>
      </ol>

      <button className="flex items-center gap-1 py-1 text-[14px] text-muted" onClick={() => setManual(!manual)}>
        <ChevronDown size={16} className={cn('transition-transform', !manual && '-rotate-90')} /> No camera? Type it in instead
      </button>
      {manual && (
        <div className="space-y-4">
          <Field label="Pairing link" hint="If you copied the link from the computer, paste it here">
            <div className="flex gap-2">
              <Input value={linkText} onChange={(e) => setLinkText(e.target.value)} placeholder="rulzagent://pair?…" autoCapitalize="off" autoCorrect="off" />
              <Button disabled={!linkText.trim()} onClick={fromLink}>
                Use
              </Button>
            </div>
          </Field>
          <div className="text-center text-[13px] text-faint">or</div>
          <Field label="Address" hint="The Tailscale address shown on the computer">
            <Input value={host} onChange={(e) => setHost(e.target.value)} placeholder="100.88.114.32" inputMode="url" autoCapitalize="off" autoCorrect="off" />
          </Field>
          <Field label="Port">
            <Input value={port} onChange={(e) => setPort(e.target.value)} inputMode="numeric" />
          </Field>
          <Field label="Pairing code">
            <Input value={token} onChange={(e) => setToken(e.target.value)} autoCapitalize="off" autoCorrect="off" spellCheck={false} />
          </Field>
          <Field label="Name" hint="Optional, e.g. Office PC">
            <Input value={name} onChange={(e) => setName(e.target.value)} />
          </Field>
          {error && <p className="text-[14px] text-danger">{error}</p>}
          <Button variant="primary" size="lg" className="w-full" onClick={fromFields}>
            Connect
          </Button>
        </div>
      )}
    </section>
  )
}

/** Shown until the first connection to the chosen computer works. */
export function Connecting() {
  const state = useLink((s) => s.state)
  const computer = useLink((s) => s.computers.find((c) => c.id === s.currentId))
  const select = useLink((s) => s.select)
  const name = computer?.name ?? 'your computer'
  const problem = linkProblem(state, name)

  return (
    <div className="pt-safe pb-safe px-safe flex h-full items-center justify-center overflow-y-auto">
      <div className="max-w-sm space-y-4 px-6 text-center">
        <LogoWordmark className="mx-auto h-10" />
        {state.kind === 'connecting' && state.attempt === 0 ? (
          <p className="flex items-center justify-center gap-2 text-[16px]">
            <Spinner className="text-accent" /> Connecting to {name}…
          </p>
        ) : (
          <>
            <h1 className="text-[18px] font-semibold">{problem?.title ?? `Connecting to ${name}…`}</h1>
            {problem?.text && <p className="text-[15px] text-muted">{problem.text}</p>}
            {state.kind === 'connecting' && state.error && <p className="text-[13px] text-faint">{state.error}</p>}
          </>
        )}
        <div className="flex flex-col gap-2 pt-2">
          {problem?.retry && (
            <Button
              variant="primary"
              size="lg"
              onClick={() => (state.kind === 'connecting' ? link.retryNow() : computer && select(computer.id))}
            >
              Try again
            </Button>
          )}
          <Button variant="ghost" onClick={() => select(null)}>
            Choose another computer
          </Button>
        </div>
      </div>
    </div>
  )
}
