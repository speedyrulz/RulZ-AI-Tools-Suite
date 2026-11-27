import { describe, expect, it } from 'vitest'
import { listGpus, parseNvidiaSmi } from '../src/main/server/gpus'

const CSV = `0, NVIDIA GeForce RTX 5060 Ti, 00000000:01:00.0, 13735, 16311, 100, 65
1, NVIDIA GeForce RTX 4060 Ti, 00000000:04:00.0, 2279, 16380, 4, 40
`

describe('parseNvidiaSmi', () => {
  it('parses cards in index order', () => {
    const g = parseNvidiaSmi(CSV)
    expect(g).toHaveLength(2)
    expect(g[0]).toEqual({
      index: 0,
      name: 'NVIDIA GeForce RTX 5060 Ti',
      busId: '00000000:01:00.0',
      memoryUsedMiB: 13735,
      memoryTotalMiB: 16311,
      utilizationPct: 100,
      temperatureC: 65
    })
    expect(g[1].index).toBe(1)
  })
  it('tolerates [N/A], CRLF and junk lines', () => {
    const g = parseNvidiaSmi('1, X, 00:04, [N/A], 16380, [N/A], [N/A]\r\nnonsense\r\n\r\nNo devices were found\r\n')
    expect(g).toHaveLength(1)
    expect(g[0]).toMatchObject({ index: 1, memoryUsedMiB: 0, utilizationPct: 0, temperatureC: null })
  })
  it('returns [] for empty text', () => {
    expect(parseNvidiaSmi('')).toEqual([])
  })
})

describe('listGpus', () => {
  it('calls nvidia-smi.exe with the query', async () => {
    let seen: { file: string; args: string[] } | null = null
    const g = await listGpus(async (file, args) => {
      seen = { file, args }
      return CSV
    })
    expect(g).toHaveLength(2)
    expect(seen!.file).toBe('nvidia-smi.exe')
    expect(seen!.args.join(' ')).toContain('index,name,pci.bus_id,memory.used,memory.total,utilization.gpu,temperature.gpu')
    expect(seen!.args.join(' ')).toContain('--format=csv,noheader,nounits')
  })
  it('gives [] when it fails', async () => {
    expect(
      await listGpus(async () => {
        throw new Error('ENOENT')
      })
    ).toEqual([])
  })
})
