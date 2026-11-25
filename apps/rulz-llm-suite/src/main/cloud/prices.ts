// OpenRouter prices: USD per token as strings -> micro-dollars per token, and the worst case of one request
// (docs/cloud-design.md, sections 3 and 5).
import type { CloudPrices } from '@shared/cloud'

const MICRO = 1_000_000

/** Micro-dollars from dollars, rounded up to a whole micro-dollar (float noise like 4200.000000000001 is removed first). */
export const toMicro = (usd: number): number => (Number.isFinite(usd) && usd > 0 ? Math.max(0, Math.ceil(Number((usd * MICRO).toPrecision(12)))) : 0)

/** Micro-dollars per token from dollars per token (a fraction is kept). */
export const microPerToken = (usdPerToken: number): number => Number((usdPerToken * MICRO).toPrecision(12))

/** undefined = not given; null = given but unusable (not a number, not finite, negative). */
function readPrice(v: unknown): number | null | undefined {
  if (v === undefined || v === null) return undefined
  let n: number
  if (typeof v === 'number') n = v
  else if (typeof v === 'string' && v.trim() !== '') n = Number(v)
  else return null
  if (!Number.isFinite(n) || n < 0) return null
  return n
}

interface Folded {
  prompt: number
  completion: number
  request: number
  image: number
  promptMax: number
  completionMax: number
}

const PROMPT_KEYS = ['prompt', 'input_cache_write', 'input_cache_write_1h']
const COMPLETION_KEYS = ['completion', 'internal_reasoning']

/** Checks every price field of an object; false = something in it is unparsable or negative. */
function foldInto(obj: Record<string, unknown>, into: Folded, base: boolean): boolean {
  for (const [k, raw] of Object.entries(obj)) {
    if (k === 'overrides' || /^(min|max)_/.test(k)) continue
    const p = readPrice(raw)
    if (p === null) return false
    if (p === undefined) continue
    const micro = microPerToken(p)
    if (PROMPT_KEYS.includes(k)) into.promptMax = Math.max(into.promptMax, micro)
    if (COMPLETION_KEYS.includes(k)) into.completionMax = Math.max(into.completionMax, micro)
    if (k === 'request') into.request = Math.max(into.request, toMicro(p))
    if (k === 'image') into.image = Math.max(into.image, toMicro(p))
    if (base && k === 'prompt') into.prompt = micro
    if (base && k === 'completion') into.completion = micro
  }
  return true
}

/**
 * The prices of a catalog entry, or null when the model cannot be used: a router (-1), any price that is not a number
 * >= 0, or a missing prompt/completion price. The Max values are the highest of the base price and every override.
 */
export function parsePrices(pricing: unknown): CloudPrices | null {
  if (!pricing || typeof pricing !== 'object' || Array.isArray(pricing)) return null
  const p = pricing as Record<string, unknown>
  if (readPrice(p.prompt) === undefined || readPrice(p.completion) === undefined) return null
  const f: Folded = { prompt: 0, completion: 0, request: 0, image: 0, promptMax: 0, completionMax: 0 }
  if (!foldInto(p, f, true)) return null
  if (p.overrides !== undefined && p.overrides !== null) {
    if (!Array.isArray(p.overrides)) return null
    for (const o of p.overrides) {
      if (!o || typeof o !== 'object' || Array.isArray(o)) return null
      if (!foldInto(o as Record<string, unknown>, f, false)) return null
    }
  }
  return { promptMax: f.promptMax, completionMax: f.completionMax, request: f.request, image: f.image, prompt: f.prompt, completion: f.completion }
}

export const isFreePrices = (p: CloudPrices): boolean => p.promptMax === 0 && p.completionMax === 0 && p.request === 0 && p.image === 0 && p.prompt === 0 && p.completion === 0

/** Largest estimated input the request is booked with: the estimate plus a quarter, plus a fixed amount per image. */
export const IMAGE_TOKENS = 2000

/**
 * Worst case of one request in whole micro-dollars (rounded up): (input x 1.25 + 2000 per image) x the dearest prompt
 * price + max_tokens x the dearest completion price + the per-request price + images x the image price.
 */
export function worstCaseMicro(prices: CloudPrices, inputTokens: number, images: number, maxTokens: number): number {
  const prompt = (inputTokens * 1.25 + IMAGE_TOKENS * images) * prices.promptMax
  const completion = maxTokens * prices.completionMax
  const total = prompt + completion + prices.request + images * prices.image
  return Number.isFinite(total) ? Math.max(0, Math.ceil(Number(total.toPrecision(12)))) : Number.MAX_SAFE_INTEGER
}

export interface MaxPrice {
  prompt: number
  completion: number
  request?: number
  image?: number
}

/**
 * provider.max_price: USD per million tokens, which is the same number as micro-dollars per token. Both are the dearest
 * price the model lists (base, cache writes, every override), so no endpoint at its listed price can be refused for being
 * too dear, while a dearer one can never be routed to. A hair is added so float noise cannot turn "equal" into "above".
 */
export function maxPriceFor(prices: CloudPrices): MaxPrice {
  const up = (n: number): number => (n === 0 ? 0 : Number((n * (1 + 1e-9)).toPrecision(12)))
  const out: MaxPrice = { prompt: up(prices.promptMax), completion: up(prices.completionMax) }
  if (prices.request > 0) out.request = prices.request / MICRO
  if (prices.image > 0) out.image = prices.image / MICRO
  return out
}
