import net from 'node:net'
import { afterEach, describe, expect, it } from 'vitest'
import { findFreePort, isPortFree, parseSsPorts, scanForFreePort } from '../src/main/server/ports'
import type { WslRunner } from '../src/main/server/wsl'

const SS = `LISTEN 0      128           0.0.0.0:8188 0.0.0.0:*
LISTEN 0      4096    127.0.0.53%lo:53   0.0.0.0:*
LISTEN 0      4096          [::]:8189      [::]:*
`

function wslWith(code: number | null, stdout: string): WslRunner {
  return {
    run: async () => ({ code, stdout, stderr: '' }),
    spawn: () => {
      throw new Error('unused')
    }
  }
}

const servers: net.Server[] = []
afterEach(() => {
  for (const s of servers.splice(0)) s.close()
})

function listen(): Promise<number> {
  return new Promise((resolve) => {
    const s = net.createServer()
    servers.push(s)
    s.listen(0, '127.0.0.1', () => resolve((s.address() as net.AddressInfo).port))
  })
}

/** A port that was just free. */
async function freePort(): Promise<number> {
  const p = await listen()
  await new Promise<void>((r) => servers.pop()!.close(() => r()))
  return p
}

describe('ports', () => {
  it('parses ss output', () => {
    expect([...parseSsPorts(SS)].sort((a, b) => a - b)).toEqual([53, 8188, 8189])
  })
  it('is not free when Node cannot bind it', async () => {
    const p = await listen()
    expect(await isPortFree(p, wslWith(0, ''), 'x')).toBe(false)
  })
  it('is not free when WSL listens on it', async () => {
    const p = await freePort()
    expect(await isPortFree(p, wslWith(0, `LISTEN 0 1 0.0.0.0:${p} 0.0.0.0:*\n`), 'x')).toBe(false)
  })
  it('is free when neither side uses it', async () => {
    expect(await isPortFree(await freePort(), wslWith(0, SS), 'x')).toBe(true)
  })
  it('is NOT free when ss fails or times out', async () => {
    const p = await freePort()
    expect(await isPortFree(p, wslWith(null, ''), 'x')).toBe(false)
    expect(await isPortFree(p, wslWith(1, ''), 'x')).toBe(false)
  })
  it('finds no port and says why when ss fails', async () => {
    const p = await freePort()
    expect(await findFreePort(p, wslWith(null, ''), 'x')).toBeNull()
    expect(await scanForFreePort(p, wslWith(null, ''), 'x')).toMatchObject({ port: null, reason: 'ss-failed' })
    expect(await scanForFreePort(p, wslWith(0, ''), 'x', 0)).toMatchObject({ port: null, reason: 'none-free' })
  })
  it('returns every listening port of WSL', async () => {
    const scan = await scanForFreePort(await freePort(), wslWith(0, SS), 'x')
    expect(scan.reason).toBeNull()
    expect([...scan.listening].sort((a, b) => a - b)).toEqual([53, 8188, 8189])
  })
  it('scans upward for the next free port', async () => {
    const p = await listen()
    const found = await findFreePort(p, wslWith(0, `LISTEN 0 1 0.0.0.0:${p + 1} 0.0.0.0:*\n`), 'x', 10)
    expect(found).not.toBeNull()
    expect(found).toBeGreaterThan(p + 1)
  })
  it('returns null when the span is used up', async () => {
    const p = await listen()
    expect(await findFreePort(p, wslWith(0, ''), 'x', 1)).toBeNull()
  })
})
