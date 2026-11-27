import { RotateCcw } from 'lucide-react'
import { useMemo, useState } from 'react'
import type { MergeCheck, MergeRecipe, UnetDtype } from '@shared/merge'
import type { ServerCapabilities } from '@shared/types'
import { cn } from '../../lib/format'
import { blockValue, formulaText, groupAverage, groupBlockInputs, validOutputName } from '../../lib/merge'
import { useMerge } from '../../stores/merge'
import { Badge, Button, Card, DebouncedInput, Field, Segmented } from '../ui'
import { ModelSelect, SliderField } from './controls'

const SELECT_CLASS =
  'h-9 w-full rounded-lg border border-border bg-panel px-3 text-[13px] focus:border-accent focus:outline-none disabled:cursor-not-allowed disabled:opacity-45'

export function SourcesCard({ caps }: { caps: ServerCapabilities }) {
  const recipe = useMerge((s) => s.recipe)
  const setRecipe = useMerge((s) => s.setRecipe)
  const list = recipe.kind === 'diffusion' ? caps.diffusionModels : caps.checkpoints
  const loaderMissing = caps.nodes[recipe.kind === 'diffusion' ? 'UNETLoader' : 'CheckpointLoaderSimple'] === false

  return (
    <Card title="Sources" description="The files to mix. A is the base. Every file comes from the server's model folders.">
      <div className="space-y-4">
        <div className="flex flex-wrap items-center gap-x-6 gap-y-3">
          <Field label="Type of file">
            <Segmented
              label="Type of file"
              value={recipe.kind}
              onChange={(kind) => setRecipe({ kind })}
              options={[
                { value: 'diffusion', label: 'Diffusion model' },
                { value: 'checkpoint', label: 'Full checkpoint' }
              ]}
            />
          </Field>
          <Field label="How to mix">
            <Segmented
              label="How to mix"
              value={recipe.mode}
              onChange={(mode) => setRecipe({ mode })}
              options={[
                { value: 'weighted', label: 'Weighted' },
                { value: 'addDifference', label: 'Add difference' },
                { value: 'blocks', label: 'Per block' }
              ]}
            />
          </Field>
        </div>
        <p className="text-xs text-faint">
          {recipe.kind === 'diffusion'
            ? 'A diffusion model is the image network alone (models/diffusion_models).'
            : 'A full checkpoint holds the image network, the text encoder and the VAE (models/checkpoints).'}
        </p>
        {loaderMissing && (
          <p className="text-[13px] text-danger-text">This server has no loader node for this type of file, so it cannot be merged here.</p>
        )}
        <div className={cn('grid gap-4', recipe.mode === 'addDifference' ? 'lg:grid-cols-3' : 'lg:grid-cols-2')}>
          <Field label="A (base)">
            <ModelSelect label="Model A" value={recipe.a} options={list} placeholder="Choose model A" onChange={(a) => setRecipe({ a })} />
          </Field>
          <Field label="B">
            <ModelSelect label="Model B" value={recipe.b} options={list} placeholder="Choose model B" onChange={(b) => setRecipe({ b })} />
          </Field>
          {recipe.mode === 'addDifference' && (
            <Field label="C" hint="Subtracted from B. Usually the base model B was trained from.">
              <ModelSelect label="Model C" value={recipe.c ?? ''} options={list} placeholder="Choose model C" onChange={(c) => setRecipe({ c })} />
            </Field>
          )}
        </div>
      </div>
    </Card>
  )
}

const DTYPES: { value: UnetDtype; label: string }[] = [
  { value: 'default', label: 'Same as the files' },
  { value: 'fp8_e4m3fn', label: 'fp8_e4m3fn' },
  { value: 'fp8_e4m3fn_fast', label: 'fp8_e4m3fn_fast' },
  { value: 'fp8_e5m2', label: 'fp8_e5m2' }
]

function Formula({ recipe }: { recipe: MergeRecipe }) {
  return (
    <p className="mt-2 text-[13px] text-muted">
      <span className="sr-only">Formula: </span>
      <span className="font-mono text-xs text-text">{formulaText(recipe)}</span>
    </p>
  )
}

function BlockControls({ check, recipe }: { check: MergeCheck | null; recipe: MergeRecipe }) {
  const setRecipe = useMerge((s) => s.setRecipe)
  const setBlock = useMerge((s) => s.setBlock)
  const resetBlocks = useMerge((s) => s.resetBlocks)
  const [setAll, setSetAll] = useState(0.5)
  const nodes = check?.blockNodes ?? []
  const node = nodes.find((n) => n.node === recipe.blockNode) ?? null
  const groups = useMemo(() => (node ? groupBlockInputs(node.activeInputs) : []), [node])
  const fits = nodes.filter((n) => n.match !== null && n.match >= 0.9)

  if (!check) return <p className="text-[13px] text-muted">Choose A and B to see the blocks of this model.</p>
  if (check.a && !check.a.readable && nodes.length === 0) {
    return <p className="text-[13px] text-muted">Per block needs to read the header of A, which this server does not allow. Use Weighted instead.</p>
  }
  if (nodes.length === 0 || (check.a?.readable && fits.length === 0)) {
    return (
      <div className="rounded-lg border border-warn/25 bg-warn/10 px-3 py-2 text-[13px] text-warn-text">
        Per block is not available for this model: no block node on the server fits it well enough. Use Weighted instead.
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <Field label="Block node" hint="The ComfyUI node that sets a share per block. Pick the one that fits A.">
        <select
          aria-label="Block node"
          className={SELECT_CLASS}
          value={recipe.blockNode ?? ''}
          onChange={(e) => setRecipe({ blockNode: e.target.value || null })}
        >
          {nodes.map((n) => (
            <option key={n.node} value={n.node}>
              {n.node}
              {n.match !== null ? ` (fits A: ${Math.round(n.match * 100)}%)` : ''}
              {n.node === check.suggestedBlockNode ? ' (suggested)' : ''}
            </option>
          ))}
        </select>
      </Field>
      {node && (
        <>
          <div className="flex flex-wrap items-center gap-2">
            <Badge tone={node.match !== null && node.match >= 0.9 ? 'ok' : 'warn'}>
              {node.match === null ? 'Fit with A unknown' : `Fits A: ${Math.round(node.match * 100)}%`}
            </Badge>
            <span className="text-xs text-faint">{node.activeInputs.length} blocks in this model. A block you do not touch uses the share of A below.</span>
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <div className="w-72 max-w-full">
              <SliderField label="Set all blocks" value={setAll} onChange={setSetAll} compact />
            </div>
            <Button size="sm" onClick={() => setBlock(node.activeInputs, setAll)}>
              Set all
            </Button>
            <Button size="sm" variant="ghost" onClick={resetBlocks}>
              <RotateCcw size={14} />
              Reset to share of A
            </Button>
          </div>
          <div className="max-h-[420px] space-y-4 overflow-y-auto rounded-lg border border-border bg-panel-2 p-3">
            {groups.map((g) => (
              <div key={g.name}>
                <div className="mb-1 flex items-center justify-between gap-3">
                  <h4 className="font-mono text-xs font-medium">
                    {g.name} <span className="text-faint">({g.inputs.length})</span>
                  </h4>
                </div>
                <div className="mb-2 max-w-md">
                  <SliderField
                    label={`All of ${g.name}`}
                    value={groupAverage(recipe.blocks, g.inputs, recipe.ratio)}
                    onChange={(v) => setBlock(g.inputs, v)}
                    compact
                  />
                </div>
                {g.inputs.length > 1 && (
                  <div className="space-y-1 border-l border-border pl-3">
                    {g.inputs.map((i) => (
                      <div key={i} className="grid grid-cols-[160px_1fr] items-center gap-3">
                        <span className="truncate font-mono text-xs text-muted" title={i}>
                          {i}
                        </span>
                        <SliderField label={i} value={blockValue(recipe.blocks, i, recipe.ratio)} onChange={(v) => setBlock([i], v)} compact />
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  )
}

export function MixCard({ check }: { check: MergeCheck | null }) {
  const recipe = useMerge((s) => s.recipe)
  const setRecipe = useMerge((s) => s.setRecipe)
  const nameBad = !validOutputName(recipe.outputName)

  return (
    <Card title="Mix" description="How much of each source goes into the result.">
      <div className="space-y-5">
        {recipe.mode === 'weighted' && (
          <div>
            <Field label="Share of A" hint="1 is all A, 0 is all B.">
              <SliderField label="Share of A" value={recipe.ratio} onChange={(ratio) => setRecipe({ ratio })} />
            </Field>
            <Formula recipe={recipe} />
          </div>
        )}
        {recipe.mode === 'addDifference' && (
          <div>
            <Field label="Multiplier" hint="How much of what B changed compared to C is added to A. 1 adds all of it, 0 adds none.">
              <SliderField label="Multiplier" value={recipe.multiplier} max={2} onChange={(multiplier) => setRecipe({ multiplier })} />
            </Field>
            <Formula recipe={recipe} />
          </div>
        )}
        {recipe.mode === 'blocks' && (
          <>
            <div>
              <Field label="Share of A" hint="Used for every block you do not change, and for the text encoder if it is merged.">
                <SliderField label="Share of A" value={recipe.ratio} onChange={(ratio) => setRecipe({ ratio })} />
              </Field>
              <Formula recipe={recipe} />
            </div>
            <BlockControls check={check} recipe={recipe} />
          </>
        )}

        {recipe.kind === 'checkpoint' && (
          <div className="flex flex-wrap items-start gap-x-6 gap-y-4">
            <Field
              label="Text encoder"
              hint={recipe.mode === 'weighted' || recipe.mode === 'blocks' ? 'Merge uses the share of A above.' : 'Merge uses the share of A.'}
            >
              <Segmented
                label="Text encoder"
                value={recipe.clip}
                onChange={(clip) => setRecipe({ clip })}
                options={[
                  { value: 'merge', label: 'Merge' },
                  { value: 'a', label: 'From A' },
                  { value: 'b', label: 'From B' }
                ]}
              />
            </Field>
            <Field label="VAE" hint="A VAE is never mixed. One of the two is saved.">
              <Segmented
                label="VAE"
                value={recipe.vae}
                onChange={(vae) => setRecipe({ vae })}
                options={[
                  { value: 'a', label: 'From A' },
                  { value: 'b', label: 'From B' }
                ]}
              />
            </Field>
          </div>
        )}

        {recipe.kind === 'diffusion' && (
          <div className="max-w-xs">
            <Field label="Precision" hint="fp8 halves the file size and loses some detail.">
              <select
                aria-label="Precision"
                className={SELECT_CLASS}
                value={recipe.weightDtype}
                onChange={(e) => setRecipe({ weightDtype: e.target.value as UnetDtype })}
              >
                {DTYPES.map((d) => (
                  <option key={d.value} value={d.value}>
                    {d.label}
                  </option>
                ))}
              </select>
            </Field>
          </div>
        )}

        <div className="max-w-md">
          <Field
            label="Output name"
            hint={
              nameBad ? (
                <span className="text-danger-text">Use letters, digits, spaces and - _ . only.</span>
              ) : (
                <>Saved as {recipe.outputName.trim()}_00001_.safetensors in the server&apos;s output folder.</>
              )
            }
          >
            <DebouncedInput
              aria-label="Output name"
              value={recipe.outputName}
              delay={250}
              onCommit={(outputName) => setRecipe({ outputName })}
              onDraft={(outputName) => {
                if (outputName !== useMerge.getState().recipe.outputName) setRecipe({ outputName })
              }}
            />
          </Field>
        </div>
      </div>
    </Card>
  )
}
