import { axisLabel, type GridJob } from '@shared/grid'
import { cellAt, gridDims, imageKey, layoutGrid, promptHeader } from '../../lib/grid'

function loadImage(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => resolve(img)
    img.onerror = () => reject(new Error('An image of the grid could not be read.'))
    img.src = url
  })
}

/** Breaks text into lines no wider than maxW (words first, long words by character); at most maxLines, the last one cut with "...". */
function wrap(ctx: CanvasRenderingContext2D, text: string, maxW: number, maxLines: number): string[] {
  const lines: string[] = []
  let line = ''
  for (const word of text.split(/\s+/).filter(Boolean)) {
    const tryLine = line ? `${line} ${word}` : word
    if (ctx.measureText(tryLine).width <= maxW) {
      line = tryLine
      continue
    }
    if (line) lines.push(line)
    line = word
    while (ctx.measureText(line).width > maxW && line.length > 1) {
      let n = line.length - 1
      while (n > 1 && ctx.measureText(line.slice(0, n)).width > maxW) n--
      lines.push(line.slice(0, n))
      line = line.slice(n)
    }
  }
  if (line) lines.push(line)
  if (lines.length > maxLines) {
    const cut = lines.slice(0, maxLines)
    cut[maxLines - 1] = `${cut[maxLines - 1].replace(/.{0,1}$/, '')}…`
    return cut
  }
  return lines
}

/** Draws the grid (headers, row labels, cell images) on an offscreen canvas and returns it as a PNG data: URL. */
export async function composeGridPng(job: GridJob, images: Record<string, string>): Promise<string> {
  const { cols, rows } = gridDims(job)
  const loaded = new Map<string, HTMLImageElement>()
  let natW = 0
  let natH = 0
  for (const c of job.cells) {
    if (c.status !== 'done' || !c.file) continue
    const url = images[imageKey(job.id, c.file)]
    if (!url) continue
    const img = await loadImage(url)
    loaded.set(`${c.col}/${c.row}`, img)
    if (!natW) {
      natW = img.naturalWidth
      natH = img.naturalHeight
    }
  }
  if (!natW) throw new Error('There are no finished images to save yet.')
  const L = layoutGrid(cols, rows, natW, natH)
  const canvas = document.createElement('canvas')
  canvas.width = L.width
  canvas.height = L.height
  const ctx = canvas.getContext('2d')
  if (!ctx) throw new Error('The image could not be drawn.')
  ctx.fillStyle = '#0A0A0C'
  ctx.fillRect(0, 0, L.width, L.height)
  ctx.textBaseline = 'middle'
  const x0 = L.pad + L.labelW + L.gap
  const y0 = L.pad + L.headerH

  // Column headers.
  ctx.fillStyle = '#E8E8EC'
  ctx.font = '600 22px sans-serif'
  ctx.textAlign = 'center'
  job.request.axis.values.forEach((v, c) => {
    const text = axisLabel(job.request.axis, job.request.recipe, v)
    const lines = wrap(ctx, text, L.cellW - 8, 2)
    const lh = 26
    const top = L.pad + L.headerH / 2 - ((lines.length - 1) * lh) / 2
    lines.forEach((ln, i) => ctx.fillText(ln, x0 + c * (L.cellW + L.gap) + L.cellW / 2, top + i * lh))
  })

  // Row labels and cells.
  ctx.textAlign = 'left'
  ctx.font = '400 20px sans-serif'
  for (let r = 0; r < rows; r++) {
    const y = y0 + r * (L.cellH + L.gap)
    ctx.fillStyle = '#B4B4BC'
    const lines = wrap(ctx, promptHeader(job.request.prompts[r] ?? '', 140), L.labelW - 12, Math.max(1, Math.floor(L.cellH / 26)))
    const top = y + L.cellH / 2 - ((lines.length - 1) * 26) / 2
    lines.forEach((ln, i) => ctx.fillText(ln, L.pad, top + i * 26))
    for (let c = 0; c < cols; c++) {
      const x = x0 + c * (L.cellW + L.gap)
      const img = loaded.get(`${c}/${r}`)
      if (img) ctx.drawImage(img, x, y, L.cellW, L.cellH)
      else {
        ctx.fillStyle = '#18181C'
        ctx.fillRect(x, y, L.cellW, L.cellH)
        ctx.fillStyle = '#8A8A94'
        ctx.textAlign = 'center'
        ctx.fillText(cellAt(job, c, r)?.status === 'error' ? 'Failed' : 'No image', x + L.cellW / 2, y + L.cellH / 2)
        ctx.textAlign = 'left'
      }
    }
  }
  return canvas.toDataURL('image/png')
}
