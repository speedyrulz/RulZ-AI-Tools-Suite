// Copied from RulZ LLM Suite (src/shared/split.ts) so both apps split GPUs the same way.
/** Reads a --tensor-split string ("3,1") as fractions for `count` GPUs; null when unset or invalid. */
export function parseTensorSplit(text: string, count: number): number[] | null {
  if (!text.trim() || count < 1) return null
  const nums = text.split(',').map((p) => Number(p.trim()))
  if (nums.length > count || nums.some((n) => !Number.isFinite(n) || n < 0)) return null
  // llama.cpp treats missing trailing entries as 0.
  const values = Array.from({ length: count }, (_, i) => nums[i] ?? 0)
  const sum = values.reduce((a, b) => a + b, 0)
  return sum > 0 ? values.map((v) => v / sum) : null
}

/** Whole percentages that add up to exactly 100. */
export function toPercents(fractions: number[]): number[] {
  const out = fractions.map((f) => Math.round(f * 100))
  const drift = 100 - out.reduce((a, b) => a + b, 0)
  if (drift && out.length) {
    const i = out.indexOf(Math.max(...out))
    out[i] += drift
  }
  return out
}

/** Sets GPU `index` to `percent` and rescales the others so the split still adds up to 100. */
export function rebalanceSplit(percents: number[], index: number, percent: number): number[] {
  if (percents.length < 2) return [100]
  const p = Math.max(0, Math.min(100, percent))
  const rest = percents.reduce((sum, v, i) => (i === index ? sum : sum + v), 0)
  const fractions = percents.map((v, i) => {
    if (i === index) return p / 100
    const share = rest > 0 ? v / rest : 1 / (percents.length - 1)
    return (share * (100 - p)) / 100
  })
  const out = toPercents(fractions)
  // Keep the GPU being dragged exactly where the user put it; rounding drift goes to the others.
  const drift = out[index] - Math.round(p)
  if (drift) {
    out[index] -= drift
    const j = out.findIndex((v, i) => i !== index && v + drift >= 0)
    if (j >= 0) out[j] += drift
  }
  return out
}
