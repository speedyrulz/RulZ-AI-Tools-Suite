// Pure helpers for the OpenRouter window: prices, badges, spending words, the catalog filter, the top-bar chip.
// Nothing here talks to the main process or reads a store, so all of it is tested (tests/renderer-cloud.test.ts).
import {
  DEFAULT_DAILY_LIMIT_USD,
  MAX_CLOUD_SESSIONS,
  MIN_CONTEXT_LIMIT,
  MIN_DAILY_LIMIT_USD,
  MIN_REPLY_LIMIT,
  formatPerMillion,
  formatUsd,
  isCloudModelId,
  samplerSupport,
  usd,
  type CloudModelDef,
  type CloudModelInfo,
  type CloudState,
  type SamplerSupport
} from '@shared/cloud'
import type { LoadedInstance, ModelEntry } from '@shared/types'

export const OPENROUTER_KEYS_URL = 'https://openrouter.ai/keys'
export const CATALOG_ROW_CAP = 100
/** From this share of the daily limit the spending bar says so. */
export const LIMIT_WARN_SHARE = 0.8

export const GOES_TO_OPENROUTER = 'Text, images and tool results of chats with these models go to OpenRouter and to the provider that answers.'

/** The model entry is an OpenRouter one (the window never trusts the id alone: a file can sit in a folder of that name). */
export const isCloudEntry = (m: { cloud?: unknown } | null | undefined): boolean => !!m?.cloud

/** The model with this id is an OpenRouter model, by its entry when the list has it, else by the id. */
export function isCloudId(id: string | null | undefined, models: { id: string; cloud?: unknown }[]): boolean {
  if (!id) return false
  const m = models.find((x) => x.id === id)
  return m ? !!m.cloud : isCloudModelId(id)
}

export const isCloudInstance = (i: Pick<LoadedInstance, 'cloud'> | null | undefined): boolean => !!i?.cloud

/** "$3.00 in · $15.00 out per million tokens", or "Free". */
export function priceLine(info: Pick<CloudModelInfo, 'prices' | 'free'>): string {
  if (info.free) return 'Free'
  return `${formatPerMillion(info.prices.prompt)} in · ${formatPerMillion(info.prices.completion)} out per million tokens`
}

/** Short form for a list row: "$3.00 / $15.00" or "Free". */
export function priceShort(info: Pick<CloudModelInfo, 'prices' | 'free'>): string {
  return info.free ? 'Free' : `${formatPerMillion(info.prices.prompt)} / ${formatPerMillion(info.prices.completion)}`
}

export interface CloudBadge {
  label: string
  tone: 'neutral' | 'ok' | 'accent'
}

export function badgesFor(info: Pick<CloudModelInfo, 'free' | 'vision' | 'tools' | 'reasoning'>): CloudBadge[] {
  const out: CloudBadge[] = []
  if (info.free) out.push({ label: 'Free', tone: 'ok' })
  if (info.vision) out.push({ label: 'Images', tone: 'neutral' })
  if (info.tools) out.push({ label: 'Tools', tone: 'neutral' })
  if (info.reasoning) out.push({ label: 'Thinking', tone: 'neutral' })
  return out
}

/** Hint under a control the model does not honour. */
export const NOT_SUPPORTED = 'Not supported by this model'

/** The sampling support of an OpenRouter model's entry; undefined for any other model (every control stays live). */
export const supportsOf = (m: { cloud?: { info: CloudModelInfo } } | null | undefined): SamplerSupport | undefined => (m?.cloud ? samplerSupport(m.cloud.info) : undefined)

// ---------- Spending ----------

export const dollars = (v: number): string => formatUsd(Math.round(v * 1_000_000))

export interface SpendView {
  spent: number
  reserved: number
  /** Dollars; null = no limit. */
  limit: number | null
  /** 0..1 of the limit used by settled cost; 0 without a limit. */
  share: number
  tone: 'ok' | 'warn' | 'danger'
  /** The state in words (status is never colour alone). */
  word: string
}

export function spendView(state: Pick<CloudState, 'spend' | 'options'>): SpendView {
  const spent = state.spend.spent
  const reserved = state.spend.reserved
  const limit = state.options.dailyLimitUsd
  if (limit === null) return { spent, reserved, limit, share: 0, tone: 'ok', word: 'No limit' }
  const cap = limit * 1_000_000
  const share = cap > 0 ? Math.min(1, spent / cap) : 1
  const over = spent + reserved >= cap
  if (over) return { spent, reserved, limit, share, tone: 'danger', word: 'Limit reached: requests are refused until tomorrow' }
  if (share >= LIMIT_WARN_SHARE) return { spent, reserved, limit, share, tone: 'warn', word: 'Close to the limit' }
  return { spent, reserved, limit, share, tone: 'ok', word: 'Within the limit' }
}

/** A limit typed in dollars: at least the minimum, cents precision. */
export const cleanLimit = (v: number | null | undefined): number =>
  v === null || v === undefined || !Number.isFinite(v) ? DEFAULT_DAILY_LIMIT_USD : Math.max(MIN_DAILY_LIMIT_USD, Math.round(v * 100) / 100)

/** Per-model spending today, biggest first. */
export function modelSpend(state: Pick<CloudState, 'spend'>): { slug: string; micro: number }[] {
  return Object.entries(state.spend.models)
    .map(([slug, micro]) => ({ slug, micro }))
    .filter((r) => r.micro > 0)
    .sort((a, b) => b.micro - a.micro || a.slug.localeCompare(b.slug))
}

/** "$0.12 of $1.00 today" or "$0.12 today". */
export function spentTodayText(state: Pick<CloudState, 'spend' | 'options'>): string {
  const spent = formatUsd(state.spend.spent)
  const limit = state.options.dailyLimitUsd
  return limit === null ? `${spent} today` : `${spent} of ${dollars(limit)} today`
}

/** The top-bar chip: only with a saved key and something spent, or a connected cloud model. Null = no chip. */
export function chipText(state: Pick<CloudState, 'keySaved' | 'spend'> | null, instances: Pick<LoadedInstance, 'cloud'>[]): string | null {
  if (!state) return null
  const connected = instances.some((i) => i.cloud)
  if (!connected && !(state.keySaved && state.spend.spent > 0)) return null
  return `OpenRouter ${formatUsd(state.spend.spent)} today`
}

/** The cost hint of a batch run on OpenRouter: what may be spent and the limit. */
export function batchCostHint(state: Pick<CloudState, 'spend' | 'options' | 'keySaved'> | null): string {
  if (!state) return 'Every request costs money unless the model is free.'
  const limit = state.options.dailyLimitUsd
  const left = limit === null ? 'No daily limit is set.' : `Today ${formatUsd(state.spend.spent)} of ${dollars(limit)} is spent; the run stops when the limit is reached.`
  return `Cloud requests cost money unless the model is free. ${left}`
}

// ---------- Limits of an added model ----------

export const contextBounds = (info: Pick<CloudModelInfo, 'contextLength'>): { min: number; max: number } => ({
  min: Math.min(MIN_CONTEXT_LIMIT, info.contextLength),
  max: info.contextLength
})

export const replyBounds = (def: Pick<CloudModelDef, 'contextLimit' | 'info'>): { min: number; max: number } => {
  const cap = Math.min(def.info.maxReply ?? def.info.contextLength, def.info.contextLength)
  return { min: Math.min(MIN_REPLY_LIMIT, cap), max: cap }
}

export const clampInt = (v: number, min: number, max: number): number => Math.min(max, Math.max(min, Math.round(v)))

/** Text under the context field. */
export const CONTEXT_HINT =
  'Chat and Writing fill the context they are told about, so a bigger context makes every message dearer. Raise it only when you need it.'

/** A warning when the model is dropped soon; null = none or far away. */
export function expiresWarning(info: Pick<CloudModelInfo, 'expires'>, now = Date.now()): string | null {
  if (!info.expires) return null
  const t = Date.parse(info.expires)
  if (!Number.isFinite(t)) return null
  const days = Math.ceil((t - now) / 86_400_000)
  if (days < 0) return `OpenRouter dropped this model on ${info.expires.slice(0, 10)}.`
  if (days > 60) return null
  return `OpenRouter drops this model on ${info.expires.slice(0, 10)}${days <= 1 ? ' (within a day)' : ` (in ${days} days)`}.`
}

// ---------- The catalog browser ----------

export interface CatalogFilter {
  query: string
  free: boolean
  images: boolean
  tools: boolean
}

export const EMPTY_FILTER: CatalogFilter = { query: '', free: false, images: false, tools: false }

/** Catalog models not yet added that match the search words (in name or slug) and every chip. Newest first, then by name. */
export function filterCatalog(models: CloudModelInfo[], added: Set<string>, f: CatalogFilter): CloudModelInfo[] {
  const words = f.query.toLowerCase().split(/\s+/).filter(Boolean)
  return models
    .filter((m) => !added.has(m.slug))
    .filter((m) => (f.free ? m.free : true) && (f.images ? m.vision : true) && (f.tools ? m.tools : true))
    .filter((m) => {
      const hay = `${m.name} ${m.slug}`.toLowerCase()
      return words.every((w) => hay.includes(w))
    })
    .sort((a, b) => b.created - a.created || a.name.localeCompare(b.name))
}

export function capRows<T>(rows: T[], cap = CATALOG_ROW_CAP): { shown: T[]; hidden: number } {
  return { shown: rows.slice(0, cap), hidden: Math.max(0, rows.length - cap) }
}

export function catalogTime(at: number): string {
  return new Date(at).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })
}

// ---------- Key ----------

/** The dollars left on the key itself, or "No limit". */
export const keyLimitValue = (left: number | null | undefined): string => (left === null || left === undefined ? 'No limit' : dollars(left))

export function freeLeftText(n: number | null | undefined): string {
  return n === null || n === undefined ? 'Not stated' : String(n)
}

// ---------- Where a cloud model runs (batch runs) ----------

/** The one host of a cloud model: OpenRouter, up to MAX_CLOUD_SESSIONS tasks at once. */
export const CLOUD_SESSION_LIMIT = MAX_CLOUD_SESSIONS

/** A model's price class for the picker and lists: the entry's own info. */
export function entryPrice(m: Pick<ModelEntry, 'cloud'>): string {
  return m.cloud ? priceShort(m.cloud.info) : ''
}

/** "$0.0042" for a reply that cost something, null when it cost nothing or is unknown. */
export const costText = (micro: number | undefined): string | null => (micro === undefined || micro <= 0 ? null : formatUsd(micro))

export { usd }
