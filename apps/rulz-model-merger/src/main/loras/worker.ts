// Worker thread entry: reads the input LoRAs, merges them on the CPU, writes the result, reads it back and verifies it.
// Started by LoraMergeService; the main thread can terminate it at any time (cancel).
import { parentPort, workerData } from 'node:worker_threads'
import { readSafetensorsFile, writeSafetensors } from '../models/tensors'
import { mergeLoras, verifyMergedLora } from './merge'
import type { LoraWorkerMessage, LoraWorkerRequest } from './workerTypes'

function post(m: LoraWorkerMessage): void {
  parentPort?.postMessage(m)
}

async function run(req: LoraWorkerRequest): Promise<void> {
  const inputs: Array<{ name: string; header: Awaited<ReturnType<typeof readSafetensorsFile>>['header']; data: Buffer; weight: number }> = []
  for (const i of req.inputs) {
    const f = await readSafetensorsFile(i.uncPath)
    inputs.push({ name: i.name, header: f.header, data: f.data, weight: i.weight })
  }
  const merged = await mergeLoras(inputs, req.recipe, {
    appVersion: req.appVersion,
    inputHeaderSha256: req.inputHeaderSha256,
    onProgress: (done, total) => post({ type: 'progress', done, total })
  })
  const written = await writeSafetensors(req.outUncPath, merged.tensors, merged.metadata)
  const back = await readSafetensorsFile(req.outUncPath)
  const verify = verifyMergedLora({ header: back.header, data: back.data }, inputs.map((i) => ({ header: i.header, data: i.data, weight: i.weight })), req.recipe)
  post({
    type: 'done',
    sizeBytes: written.bytes,
    modules: merged.modules,
    rank: merged.rank,
    energyKept: merged.energyKept,
    verify: { ok: verify.ok, message: verify.message },
    metadata: merged.metadata
  })
}

run(workerData as LoraWorkerRequest).catch((e: unknown) => {
  post({ type: 'error', message: e instanceof Error ? e.message : String(e) })
})
