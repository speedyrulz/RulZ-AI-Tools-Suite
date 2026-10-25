// Drive a Chromium page over the DevTools protocol and save screenshots (Node 22+: global fetch/WebSocket).
//
//   node cdp_capture.mjs plan.json
//
// plan.json:
// {
//   "port": 9333,                       // --remote-debugging-port of the Electron app / Edge
//   "match": "index.html",              // substring of the page target's URL to attach to
//   "width": 1600, "height": 960,       // CSS-pixel viewport forced with Emulation.setDeviceMetricsOverride
//   "outDir": "R:\\out\\llm-suite",
//   "startWaitMs": 4000,
//   "shots": [ { "name": "01-x.png", "js": "...click something...", "waitMs": 1500, "check": "js returning a string" } ]
// }
// "check" is evaluated after the wait and printed, so the run log shows what was on screen (e.g. headings).
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs'
import { join } from 'node:path'

const plan = JSON.parse(readFileSync(process.argv[2], 'utf8'))
const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

async function findTarget() {
  for (let i = 0; i < 60; i++) {
    try {
      const list = await (await fetch(`http://127.0.0.1:${plan.port}/json/list`)).json()
      const t = list.find((x) => x.type === 'page' && x.url.includes(plan.match ?? ''))
      if (t) return t
    } catch {}
    await sleep(500)
  }
  throw new Error('no page target found')
}

const target = await findTarget()
console.log('target', target.url)
const ws = new WebSocket(target.webSocketDebuggerUrl)
await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej })
let id = 0
const pending = new Map()
ws.onmessage = (ev) => {
  const msg = JSON.parse(ev.data)
  if (msg.id && pending.has(msg.id)) {
    const { resolve, reject } = pending.get(msg.id)
    pending.delete(msg.id)
    msg.error ? reject(new Error(JSON.stringify(msg.error))) : resolve(msg.result)
  }
}
const send = (method, params = {}) => new Promise((resolve, reject) => {
  const n = ++id
  pending.set(n, { resolve, reject })
  ws.send(JSON.stringify({ id: n, method, params }))
})
const evaluate = async (expr) => {
  const r = await send('Runtime.evaluate', { expression: expr, awaitPromise: true, returnByValue: true })
  if (r.exceptionDetails) throw new Error(r.exceptionDetails.text + ' ' + JSON.stringify(r.exceptionDetails.exception?.description))
  return r.result.value
}

mkdirSync(plan.outDir, { recursive: true })
await send('Page.enable')
await send('Emulation.setDeviceMetricsOverride', {
  width: plan.width, height: plan.height, deviceScaleFactor: 1, mobile: false
})
if (plan.colorScheme) {
  await send('Emulation.setEmulatedMedia', { features: [{ name: 'prefers-color-scheme', value: plan.colorScheme }] })
}
await sleep(plan.startWaitMs ?? 3000)

for (const shot of plan.shots) {
  if (shot.js) console.log(shot.name, 'action ->', await evaluate(shot.js))
  await sleep(shot.waitMs ?? 1500)
  if (shot.check) console.log(shot.name, 'check ->', await evaluate(shot.check))
  if (!shot.name.endsWith('.png')) continue // action-only step
  const { data } = await send('Page.captureScreenshot', { format: 'png', captureBeyondViewport: false })
  writeFileSync(join(plan.outDir, shot.name), Buffer.from(data, 'base64'))
  console.log('saved', join(plan.outDir, shot.name))
}
await send('Emulation.clearDeviceMetricsOverride')
ws.close()
