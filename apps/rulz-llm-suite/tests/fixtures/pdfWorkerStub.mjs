// Stands in for the PDF worker in reader tests: what it does depends on the "query" of the job it is given.
import { parentPort, workerData } from 'node:worker_threads'

const mode = workerData.job.query
if (mode === 'ok') parentPort.postMessage({ ok: true, result: { pageCount: 1, title: '', pages: [{ page: 1, text: 'stub' }] } })
else if (mode === 'problem') parentPort.postMessage({ ok: false, code: 'password', message: 'This PDF is password-protected, so it cannot be read.' })
else if (mode === 'slow') setTimeout(() => parentPort.postMessage({ ok: true, result: { pageCount: 1, title: '', pages: [{ page: 1, text: 'slow' }] } }), 300)
else if (mode === 'hang') setInterval(() => undefined, 1000)
else if (mode === 'throw') throw new Error('boom')
else if (mode === 'exit') process.exit(3)
else if (mode === 'buffers') {
  // Memory outside the JavaScript heap, in steps the watcher can see.
  const hold = []
  setInterval(() => {
    if (hold.length < 40) hold.push(new Uint8Array(32 * 1024 * 1024).fill(1))
  }, 50)
}
else if (mode === 'memory') {
  const hold = []
  for (;;) hold.push(new Array(1_000_000).fill(1))
}
