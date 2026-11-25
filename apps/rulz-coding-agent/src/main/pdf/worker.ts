// Entry of the PDF worker thread. electron.vite.config.ts builds it as its own main-process file (out/main/pdfWorker.js).
// It does one job and answers once; the reader ends the thread afterwards.
import { parentPort, workerData } from 'node:worker_threads'
import { runPdfJob, type PdfJob, type PdfJobResult } from './extract'
import { PdfProblem } from './text'

export interface PdfWorkerInput {
  data: Uint8Array
  job: PdfJob
  cmapDir?: string
}

export type PdfWorkerOutput = { ok: true; result: PdfJobResult } | { ok: false; code: PdfProblem['code']; message: string }

async function main(): Promise<void> {
  const input = workerData as PdfWorkerInput
  let out: PdfWorkerOutput
  try {
    out = { ok: true, result: await runPdfJob(input.data, input.job, { cmapDir: input.cmapDir }) }
  } catch (e) {
    out =
      e instanceof PdfProblem
        ? { ok: false, code: e.code, message: e.message }
        : { ok: false, code: 'damaged', message: `This PDF could not be read: ${(e as Error)?.message ?? String(e)}` }
  }
  parentPort?.postMessage(out)
}

void main()
