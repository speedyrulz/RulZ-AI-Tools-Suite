import type { PhoneMode, PhoneModeInfo, PhoneModel } from '@shared/phone'

/** "OpenRouter, Free" or "OpenRouter, Paid"; anything but `free: true` counts as paid, whatever the computer sent. */
export const cloudLabel = (model: { cloud?: unknown }): string =>
  `OpenRouter, ${model.cloud && typeof model.cloud === 'object' && (model.cloud as { free?: unknown }).free === true ? 'Free' : 'Paid'}`
export const rentalLabel = (model: PhoneModel): string => (model.cloud ? cloudLabel(model) : (model.rental?.label ?? 'Not loaded on a rental'))
/** A rental that is not loaded cannot be newly picked. An OpenRouter model is listed only when LLM Suite allows it for other apps, and can always be picked. */
export const canPickModel = (model: PhoneModel, chosen: string | null): boolean => !!model.cloud || !model.rental || model.rental.label !== null || model.id === chosen
/** A chosen model that is no longer listed: never swapped for another one; an OpenRouter one is named as such. */
export const isCloudId = (id: string): boolean => id.startsWith('openrouter/')
export const missingModelLabel = (id: string): string => (isCloudId(id) ? `${id.slice('openrouter/'.length)} (OpenRouter, not available now)` : id)
export function applies(info: PhoneModeInfo): string {
  return [info.ownLoad && 'own load settings', info.mode !== 'writing' && info.ownSampling && 'own sampling', info.hasPrompt && 'own prompt'].filter(Boolean).join(', ')
}

/** A reload that has not come back after this long failed on the computer; the line then stops saying it is loading. */
export const LOADING_LIMIT_MS = 10 * 60_000

// Kept only for this app session, keyed by the actual model, computer and mode.
const kept = new Set<string>()
export class ModeController {
  info: PhoneModeInfo | null = null
  unsupported = false
  busy = false
  loading = false
  private generation = 0
  private pending = false
  private queued = false
  private loadingSince = 0
  constructor(readonly computer: string, readonly mode: PhoneMode, readonly modelId: string | null,
    private ask: () => Promise<PhoneModeInfo>, private reloadRequest: (id: string) => Promise<PhoneModeInfo>, private changed: () => void) {}
  private key(info: PhoneModeInfo): string { return JSON.stringify([this.computer, this.mode, info.modelId]) }
  get isKept(): boolean { return !!this.info && kept.has(this.key(this.info)) }
  keep(): void { if (this.info) { kept.add(this.key(this.info)); this.changed() } }
  invalidate(): void { this.generation++; this.queued = false }
  reconnect(): void { this.invalidate(); this.unsupported = false; this.info = null; this.changed() }
  private accept(info: PhoneModeInfo): void {
    if (!info.loaded || !info.reload) kept.delete(this.key(info))
    this.info = info
    if (info.loaded || Date.now() - this.loadingSince > LOADING_LIMIT_MS) this.loading = false
    this.changed()
  }
  async refresh(): Promise<void> {
    if (this.unsupported) return
    if (this.pending || this.busy) { this.queued = true; return }
    this.pending = true
    const generation = this.generation
    try {
      const info = await this.ask()
      if (generation === this.generation) this.accept(info)
    } catch (e) {
      if (generation === this.generation && e instanceof Error && e.message === 'That is not available from the phone.') {
        this.unsupported = true; this.info = null; this.changed()
      }
    } finally {
      this.pending = false
      if (this.queued) { this.queued = false; void this.refresh() }
    }
  }
  async reload(): Promise<string | null> {
    const info = this.info
    if (this.busy || !info?.modelId || !info.reload || info.reload.blocked || info.rental || info.cloud) return null
    this.busy = true; this.changed()
    // An answer still on its way is from before the reload: it is dropped.
    const generation = ++this.generation
    let error: string | null = null
    try {
      const result = await this.reloadRequest(info.modelId)
      // That it loads now stays true when the link came back or the screen was left meanwhile.
      this.loading = !result.loaded
      this.loadingSince = Date.now()
      if (generation === this.generation) this.accept(result)
    } catch (e) { error = e instanceof Error ? e.message : String(e) }
    finally { this.busy = false; this.changed(); if (error || this.queued) { this.queued = false; void this.refresh() } }
    return error
  }
}
