import { AlertTriangle, CheckCircle2, FolderSearch, Repeat } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import type { AdapterConverterStatus, AdapterConvertRequest, AdapterConvertResult, AdapterEntry } from '@shared/types'
import { defaultConvertName } from '../../lib/adapters'
import { errorMessage, invoke, on } from '../../lib/api'
import { useModels } from '../../stores/models'
import { Button, Dialog, Field, IconButton, Input, Select } from '../ui'

const OUT_TYPES: AdapterConvertRequest['outType'][] = ['f16', 'bf16', 'f32', 'q8_0', 'auto']

/** The base model the converted adapter says it was trained on, from the adapter library; says so when unknown. */
function convertedBase(library: AdapterEntry[], outPath: string | undefined, outName: string): string {
  const norm = (p: string): string => p.replace(/\\/g, '/').toLowerCase()
  const file = (outPath ?? outName).split(/[\\/]/).pop() ?? ''
  const entry = library.find((a) => (outPath && norm(a.path) === norm(outPath)) || norm(a.path).endsWith('/' + norm(file)) || norm(a.name + '.gguf') === norm(file))
  return entry?.info?.baseModelName?.trim() || 'an unknown base model'
}

/** Converts a Hugging Face PEFT LoRA folder to a GGUF adapter in the models folder. */
export function AdapterConvertDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const [status, setStatus] = useState<AdapterConverterStatus | null>(null)
  const [peftDir, setPeftDir] = useState('')
  const [baseDir, setBaseDir] = useState('')
  const [baseModelId, setBaseModelId] = useState('')
  const [outType, setOutType] = useState<AdapterConvertRequest['outType']>('f16')
  const [outName, setOutName] = useState('')
  /** The user typed the name; picking another folder then keeps it. */
  const [nameEdited, setNameEdited] = useState(false)
  const [running, setRunning] = useState(false)
  const [confirmClose, setConfirmClose] = useState(false)
  const [lines, setLines] = useState<string[]>([])
  const [result, setResult] = useState<AdapterConvertResult | null>(null)
  const library = useModels((s) => s.adapters)
  const logRef = useRef<HTMLDivElement>(null)
  const stick = useRef(true)

  useEffect(() => {
    if (!open) return
    setStatus(null)
    setResult(null)
    setLines([])
    setConfirmClose(false)
    let alive = true
    invoke('adapters:converterStatus').then(
      (s) => alive && setStatus(s),
      (e) => alive && setStatus({ available: false, python: null, script: null, error: errorMessage(e) })
    )
    return () => {
      alive = false
    }
  }, [open])

  useEffect(() => {
    if (!open) return
    return on('adapters:convertLog', (line) => setLines((l) => [...l, line]))
  }, [open])

  useEffect(() => {
    const el = logRef.current
    if (el && stick.current) el.scrollTop = el.scrollHeight
  }, [lines])

  const pickPeft = async (): Promise<void> => {
    const dir = await invoke('dialog:pickFolder', { title: 'Choose the PEFT adapter folder (has adapter_config.json)', defaultPath: peftDir || undefined })
    if (!dir) return
    setPeftDir(dir)
    if (!nameEdited) setOutName(defaultConvertName(dir))
  }
  const pickBase = async (): Promise<void> => {
    const dir = await invoke('dialog:pickFolder', { title: 'Choose the base model folder (only config.json is needed)', defaultPath: baseDir || undefined })
    if (dir) setBaseDir(dir)
  }

  const canConvert = !!status?.available && !!peftDir.trim() && !!outName.trim() && !running
  const convert = async (): Promise<void> => {
    setRunning(true)
    setResult(null)
    setLines([])
    stick.current = true
    try {
      setResult(await invoke('adapters:convert', { peftDir: peftDir.trim(), baseDir: baseDir.trim(), baseModelId: baseModelId.trim(), outType, outName: outName.trim() }))
    } catch (e) {
      setResult({ ok: false, error: errorMessage(e), logTail: [] })
    } finally {
      setRunning(false)
    }
  }

  const requestClose = (): void => {
    if (running) setConfirmClose(true)
    else onClose()
  }
  const stopAndClose = (): void => {
    setConfirmClose(false)
    void invoke('adapters:cancelConvert').catch(() => undefined)
    onClose()
  }

  return (
    <>
      <Dialog
        open={open}
        onOpenChange={(o) => !o && requestClose()}
        title="Convert PEFT adapter"
        description="Turns a Hugging Face LoRA folder into a GGUF adapter that llama.cpp can load. The result is written into your models folder."
        className="w-[min(680px,calc(100vw-32px))]"
        footer={
          <>
            {running ? (
              <Button variant="ghost" onClick={() => void invoke('adapters:cancelConvert')}>
                Cancel
              </Button>
            ) : (
              <Button variant="ghost" onClick={onClose}>
                Close
              </Button>
            )}
            <Button variant="primary" disabled={!canConvert} onClick={() => void convert()}>
              <Repeat size={16} /> {running ? 'Converting…' : 'Convert'}
            </Button>
          </>
        }
      >
        <div className="space-y-4 px-5 py-4">
          {status && !status.available && (
            <div className="flex items-start gap-2 rounded-md border border-danger/40 bg-danger/10 p-2 text-xs text-danger">
              <AlertTriangle size={14} className="mt-px shrink-0" />
              <span>Converting is not available: {status.error ?? 'the converter script or Python was not found.'}</span>
            </div>
          )}
          <Field label="PEFT adapter folder" hint="Must contain adapter_config.json">
            <div className="flex gap-1.5">
              <Input value={peftDir} placeholder="Folder with adapter_config.json" disabled={running} onChange={(e) => setPeftDir(e.target.value)} />
              <IconButton label="Browse" disabled={running} onClick={() => void pickPeft()}>
                <FolderSearch size={15} />
              </IconButton>
            </div>
          </Field>
          <Field label="Base model folder" hint="Optional; only config.json is needed">
            <div className="flex gap-1.5">
              <Input value={baseDir} placeholder="None" disabled={running} onChange={(e) => setBaseDir(e.target.value)} />
              <IconButton label="Browse" disabled={running} onClick={() => void pickBase()}>
                <FolderSearch size={15} />
              </IconButton>
            </div>
          </Field>
          <Field label="Or Hugging Face base model id" hint="Optional">
            <Input value={baseModelId} placeholder="org/model-name" disabled={running} onChange={(e) => setBaseModelId(e.target.value)} />
          </Field>
          <p className="-mt-2 text-[11px] text-faint">
            If you leave both empty, the base model named in adapter_config.json is used, and its config may be downloaded from the Hugging Face Hub.
          </p>
          <div className="grid grid-cols-[140px_1fr] gap-3">
            <Field label="Output type">
              <Select value={outType} disabled={running} onChange={(e) => setOutType(e.target.value as AdapterConvertRequest['outType'])}>
                {OUT_TYPES.map((t) => (
                  <option key={t} value={t}>
                    {t === 'auto' ? 'Auto' : t.toUpperCase()}
                  </option>
                ))}
              </Select>
            </Field>
            <Field label="Output file name" hint="Written into the models folder">
              <Input
                value={outName}
                placeholder="my-adapter.gguf"
                disabled={running}
                onChange={(e) => {
                  setOutName(e.target.value)
                  setNameEdited(true)
                }}
              />
            </Field>
          </div>
          {(running || lines.length > 0) && (
            <div
              ref={logRef}
              role="log"
              aria-label="Converter output"
              className="h-48 overflow-auto rounded-md border border-border bg-code px-3 py-2 font-mono text-[11.5px] leading-relaxed whitespace-pre text-muted"
              onScroll={(e) => {
                const el = e.currentTarget
                stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 40
              }}
            >
              {lines.map((l, i) => (
                <div key={i} className={/error|failed/i.test(l) ? 'text-danger' : /warn/i.test(l) ? 'text-warn' : undefined}>
                  {l}
                </div>
              ))}
            </div>
          )}
          {result &&
            (result.ok ? (
              <div className="flex items-start gap-2 text-[13px] text-ok">
                <CheckCircle2 size={16} className="mt-px shrink-0" />
                <div className="min-w-0 space-y-1">
                  <div className="break-all">Converted: {(result.outPath ?? outName).split(/[\\/]/).pop()}</div>
                  <div className="text-muted">
                    Trained on {convertedBase(library, result.outPath, outName)}. Add it to a chat's adapters or a Writing project's <strong className="font-semibold text-body">Adapters</strong>.
                  </div>
                </div>
              </div>
            ) : (
              <div className="flex items-start gap-2 rounded-md border border-danger/40 bg-danger/10 p-2 text-xs text-danger">
                <AlertTriangle size={14} className="mt-px shrink-0" />
                <span className="break-words whitespace-pre-wrap">{result.error ?? 'The conversion failed.'}</span>
              </div>
            ))}
        </div>
      </Dialog>
      <Dialog
        open={confirmClose}
        onOpenChange={(o) => !o && setConfirmClose(false)}
        title="Stop the conversion?"
        description="The converter is still running. Closing this window stops it and no file is written."
        footer={
          <>
            <Button variant="ghost" onClick={() => setConfirmClose(false)}>
              Keep converting
            </Button>
            <Button variant="danger" onClick={stopAndClose}>
              Stop and close
            </Button>
          </>
        }
      />
    </>
  )
}
