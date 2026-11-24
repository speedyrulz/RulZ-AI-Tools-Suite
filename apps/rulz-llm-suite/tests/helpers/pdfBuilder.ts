// Small hand-written PDFs for the PDF tests: built from code, no files to download or keep.
import type { PdfJob, PdfJobResult } from '../../src/main/pdf/extract'
import { runPdfJob } from '../../src/main/pdf/extract'
import type { PdfReader } from '../../src/main/pdf/reader'

const esc = (s: string): string => s.replace(/\\/g, '\\\\').replace(/\(/g, '\\(').replace(/\)/g, '\\)')

/** A PDF with one Helvetica text page per entry. `encrypted` adds a password the file's check values do not match. */
export function buildPdf(pages: (string[] | null)[], opts: { title?: string; encrypted?: boolean; fontSize?: number } = {}): Buffer {
  const size = opts.fontSize ?? 12
  const objs: string[] = []
  const add = (body: string): number => objs.push(body)
  add('<< /Type /Catalog /Pages 2 0 R >>')
  const kids = pages.map((_, i) => `${4 + i * 2} 0 R`).join(' ')
  add(`<< /Type /Pages /Kids [${kids}] /Count ${pages.length} >>`)
  add('<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>')
  pages.forEach((lines, i) => {
    const content = lines?.length ? `BT /F1 ${size} Tf 72 720 Td ${size + 2} TL ${lines.map((l) => `(${esc(l)}) Tj T*`).join(' ')} ET` : ''
    add(`<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> /Contents ${5 + i * 2} 0 R >>`)
    add(`<< /Length ${content.length} >>\nstream\n${content}\nendstream`)
  })
  let info = ''
  if (opts.title) info = ` /Info ${add(`<< /Title (${esc(opts.title)}) >>`)} 0 R`
  let encrypt = ''
  if (opts.encrypted) {
    const hex = (b: string): string => b.repeat(32)
    encrypt = ` /Encrypt ${add(`<< /Filter /Standard /V 1 /R 2 /O <${hex('ab')}> /U <${hex('cd')}> /P -4 >>`)} 0 R /ID [<00112233445566778899aabbccddeeff> <00112233445566778899aabbccddeeff>]`
  }
  return assemble(objs, info + encrypt)
}

/** One page of Japanese text in a predefined encoding, which can only be read with pdf.js's character maps. */
export function buildJapanesePdf(): Buffer {
  const content = 'BT /F1 24 Tf 72 700 Td <30423044> Tj ET'
  return assemble([
    '<< /Type /Catalog /Pages 2 0 R >>',
    '<< /Type /Pages /Kids [4 0 R] /Count 1 >>',
    '<< /Type /Font /Subtype /Type0 /BaseFont /HeiseiMin-W3 /Encoding /UniJIS-UCS2-H /DescendantFonts [<< /Type /Font /Subtype /CIDFontType0 /BaseFont /HeiseiMin-W3 /CIDSystemInfo << /Registry (Adobe) /Ordering (Japan1) /Supplement 2 >> /DW 1000 >>] >>',
    '<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> /Contents 5 0 R >>',
    `<< /Length ${content.length} >>\nstream\n${content}\nendstream`
  ])
}

function assemble(objs: string[], trailer = ''): Buffer {
  let out = '%PDF-1.4\n'
  const offsets: number[] = []
  objs.forEach((body, i) => {
    offsets.push(out.length)
    out += `${i + 1} 0 obj\n${body}\nendobj\n`
  })
  const xref = out.length
  out += `xref\n0 ${objs.length + 1}\n0000000000 65535 f \n${offsets.map((o) => `${String(o).padStart(10, '0')} 00000 n \n`).join('')}`
  out += `trailer\n<< /Size ${objs.length + 1} /Root 1 0 R${trailer} >>\nstartxref\n${xref}\n%%EOF\n`
  return Buffer.from(out, 'latin1')
}

/** Reads in the test's own thread (the app always uses a worker). */
export class InProcessPdfReader implements PdfReader {
  calls = 0
  run(data: Uint8Array, job: PdfJob): Promise<PdfJobResult> {
    this.calls++
    return runPdfJob(Uint8Array.from(data), job)
  }
}
