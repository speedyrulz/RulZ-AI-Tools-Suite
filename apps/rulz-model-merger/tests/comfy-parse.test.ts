import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'
import { KNOWN_NODES, type GpuInfo } from '@shared/types'
import { matchGpuIndexes, normalizeServerUrl, parseCapabilities, parseSystemStats } from '../src/main/comfy/parse'

const fixture = (name: string): unknown => JSON.parse(readFileSync(join(__dirname, 'fixtures', 'comfy', name), 'utf8'))

function gpu(index: number, name: string): GpuInfo {
  return { index, name, busId: '', memoryUsedMiB: 0, memoryTotalMiB: 16000, utilizationPct: 0, temperatureC: null }
}

describe('normalizeServerUrl', () => {
  it('adds http:// and drops slash, path, query and fragment', () => {
    expect(normalizeServerUrl('localhost:8188')).toBe('http://localhost:8188')
    expect(normalizeServerUrl('  127.0.0.1:8188/  ')).toBe('http://127.0.0.1:8188')
    expect(normalizeServerUrl('http://10.0.0.5:8188/some/path?x=1#frag')).toBe('http://10.0.0.5:8188')
    expect(normalizeServerUrl('HTTPS://Example.com/')).toBe('https://example.com')
    expect(normalizeServerUrl('http://[::1]:8188/')).toBe('http://[::1]:8188')
    expect(normalizeServerUrl('http://host:80')).toBe('http://host')
  })
  it('rejects other schemes, empty and unparsable values with plain messages', () => {
    expect(() => normalizeServerUrl('ftp://host')).toThrow(/http/)
    expect(() => normalizeServerUrl('file:///etc/passwd')).toThrow(/http/)
    expect(() => normalizeServerUrl('')).toThrow(/address/)
    expect(() => normalizeServerUrl('   ')).toThrow(/address/)
    expect(() => normalizeServerUrl('http://')).toThrow(/valid/)
    expect(() => normalizeServerUrl('http://exa mple.com')).toThrow(/valid/)
  })
})

describe('parseSystemStats', () => {
  it('reads version, devices, argv, cuda device and RAM', () => {
    const info = parseSystemStats(fixture('system_stats.json'))
    expect(info.version).toBe('0.46.0')
    expect(info.devices).toEqual(['cuda:0 NVIDIA GeForce RTX 5060 Ti : cudaMallocAsync', 'cuda:1 NVIDIA GeForce RTX 4060 Ti : cudaMallocAsync'])
    expect(info.cudaDevice).toBe('1')
    expect(info.ramTotalBytes).toBe(64000000000)
    expect(info.ramFreeBytes).toBe(32000000000)
  })
  it('reads --cuda-device=X and tolerates missing fields', () => {
    expect(parseSystemStats({ system: { argv: ['main.py', '--cuda-device=0'] } }).cudaDevice).toBe('0')
    expect(parseSystemStats({ system: { argv: ['--cuda-device'] } }).cudaDevice).toBeNull()
    // repeated option: the last one wins (argparse)
    expect(parseSystemStats({ system: { argv: ['--cuda-device', '0', '--cuda-device=1'] } }).cudaDevice).toBe('1')
    expect(parseSystemStats({ system: { argv: ['--cuda-device=2', '--cuda-device', '3'] } }).cudaDevice).toBe('3')
    for (const bad of [null, undefined, 5, 'x', [], {}, { system: 3, devices: 'no' }]) {
      const i = parseSystemStats(bad)
      expect(i).toEqual({ version: null, devices: [], argv: [], cudaDevice: null, ramTotalBytes: null, ramFreeBytes: null })
    }
    expect(parseSystemStats({ devices: [{ name: 'a' }, { type: 'cpu' }, null, { name: 3 }] }).devices).toEqual(['a'])
  })
})

describe('parseCapabilities', () => {
  it('reads the old combo encoding and finds block merge nodes', () => {
    const c = parseCapabilities(fixture('object_info_old.json'))
    expect(c.checkpoints).toEqual(['sample-a.safetensors', 'folder/sample-b.safetensors'])
    expect(c.diffusionModels).toEqual(['unet-one.safetensors'])
    expect(c.loras).toEqual(['lora-one.safetensors', 'lora-two.safetensors'])
    expect(c.textEncoders).toEqual(['te-one.safetensors'])
    expect(c.vaes).toEqual(['vae-one.safetensors'])
    expect(c.blockMergeNodes).toEqual(['ModelMergeAlpha', 'ModelMergeFluxTest'])
    expect(Object.keys(c.nodes).sort()).toEqual([...KNOWN_NODES].sort())
    expect(c.nodes.ModelMergeSimple).toBe(true)
    expect(c.nodes.ModelMergeBlocks).toBe(true)
    expect(c.nodes.CheckpointSave).toBe(false)
    expect(c.nodes.KSampler).toBe(true)
  })
  it('reads the new COMBO encoding, including optional inputs and empty options', () => {
    const c = parseCapabilities(fixture('object_info_new.json'))
    expect(c.checkpoints).toEqual(['new-a.safetensors'])
    expect(c.diffusionModels).toEqual(['new-unet.safetensors', 'new-unet2.safetensors'])
    expect(c.loras).toEqual([])
    expect(c.textEncoders).toEqual(['new-te.safetensors'])
    expect(c.vaes).toEqual([])
    expect(c.nodes.VAELoader).toBe(false)
  })
  it('survives garbage', () => {
    for (const bad of [null, 3, [], { UNETLoader: null }, { UNETLoader: { input: 'x' } }, { LoraLoader: { input: { required: { lora_name: 'x' } } } }]) {
      const c = parseCapabilities(bad)
      expect(c.checkpoints).toEqual([])
      expect(c.loras).toEqual([])
      expect(c.blockMergeNodes).toEqual([])
    }
  })
})

describe('matchGpuIndexes', () => {
  const gpus = [gpu(0, 'NVIDIA GeForce RTX 4060 Ti'), gpu(1, 'NVIDIA GeForce RTX 5060 Ti'), gpu(2, 'NVIDIA GeForce RTX 4060')]
  it('matches by card name', () => {
    expect(matchGpuIndexes(['cuda:0 NVIDIA GeForce RTX 5060 Ti : cudaMallocAsync'], gpus)).toEqual([1])
  })
  it('does not let a shorter name match a longer card', () => {
    expect(matchGpuIndexes(['cuda:0 NVIDIA GeForce RTX 4060 Ti : cudaMallocAsync'], gpus)).toEqual([0])
    expect(matchGpuIndexes(['cuda:0 NVIDIA GeForce RTX 4060 : cudaMallocAsync'], gpus)).toEqual([2])
  })
  it('returns every card that shares a name, over several devices', () => {
    const twins = [gpu(0, 'NVIDIA RTX A'), gpu(1, 'NVIDIA RTX A'), gpu(2, 'Other')]
    expect(matchGpuIndexes(['cuda:0 NVIDIA RTX A : cudaMallocAsync'], twins)).toEqual([0, 1])
    expect(matchGpuIndexes(['cuda:0 Other : x', 'cuda:1 NVIDIA RTX A : x'], twins)).toEqual([0, 1, 2])
  })
  it('falls back to containment for unusual strings and returns [] for CPU', () => {
    expect(matchGpuIndexes(['weird NVIDIA GeForce RTX 5060 Ti thing'], gpus)).toEqual([1])
    expect(matchGpuIndexes(['cpu'], gpus)).toEqual([])
    expect(matchGpuIndexes(['  CPU  '], gpus)).toEqual([])
    expect(matchGpuIndexes([], gpus)).toEqual([])
    expect(matchGpuIndexes(['cuda:0 NVIDIA GeForce RTX 5060 Ti : x'], [])).toEqual([])
  })
  it('runs the containment fallback for each device that had no exact match', () => {
    // Device 0 matches exactly; device 1 only by containment. Both must count.
    expect(matchGpuIndexes(['cuda:0 NVIDIA GeForce RTX 5060 Ti : cudaMallocAsync', 'weird NVIDIA GeForce RTX 4060 Ti thing'], gpus)).toEqual([0, 1])
  })
})
