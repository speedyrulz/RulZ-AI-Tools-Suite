// A rough token count without a tokenizer: plain ASCII is about 3.5 characters a token, anything else about one token a
// character (the worst case for CJK and emoji). Images are counted, never their data URLs.

/** Tokens of a text as a fraction. */
export function textUnits(s: string): number {
  let ascii = 0
  let other = 0
  for (let i = 0; i < s.length; i++) {
    const c = s.charCodeAt(i)
    if (c < 128) ascii++
    // A low surrogate belongs to the high one before it: one character.
    else if (c >= 0xdc00 && c <= 0xdfff && i > 0) continue
    else other++
  }
  return ascii / 3.5 + other
}

export const estimateText = (s: string): number => Math.ceil(textUnits(s))

const PER_MESSAGE = 4

export interface InputEstimate {
  tokens: number
  images: number
}

function isImagePart(p: unknown): boolean {
  const t = (p as { type?: unknown } | null)?.type
  return t === 'image_url' || t === 'image'
}

/** Estimate of a messages array (and optionally the tool definitions). */
export function estimateInput(messages: unknown, tools?: unknown): InputEstimate {
  let units = 0
  let images = 0
  const list = Array.isArray(messages) ? messages : []
  for (const m of list) {
    if (!m || typeof m !== 'object') continue
    const msg = m as Record<string, unknown>
    units += PER_MESSAGE
    const content = msg.content
    if (typeof content === 'string') units += textUnits(content)
    else if (Array.isArray(content)) {
      for (const part of content) {
        if (isImagePart(part)) images++
        else if (part && typeof part === 'object' && typeof (part as { text?: unknown }).text === 'string') units += textUnits((part as { text: string }).text)
      }
    }
    if (typeof msg.reasoning_content === 'string') units += textUnits(msg.reasoning_content)
    if (typeof msg.name === 'string') units += textUnits(msg.name)
    if (Array.isArray(msg.tool_calls)) {
      for (const c of msg.tool_calls) {
        const f = (c as { function?: { name?: unknown; arguments?: unknown } } | null)?.function
        if (typeof f?.name === 'string') units += textUnits(f.name)
        if (typeof f?.arguments === 'string') units += textUnits(f.arguments)
        units += PER_MESSAGE
      }
    }
    if (Array.isArray(msg.reasoning_details)) units += textUnits(JSON.stringify(msg.reasoning_details))
  }
  if (Array.isArray(tools) && tools.length) units += textUnits(JSON.stringify(tools))
  return { tokens: Math.ceil(units), images }
}
