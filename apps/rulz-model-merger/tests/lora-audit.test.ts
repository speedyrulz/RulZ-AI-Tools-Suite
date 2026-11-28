import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'
import { parseLoraLayers } from '../src/main/loras/merge'
import { readSafetensorsHeader } from '../src/main/models/safetensors'
import type { SafetensorsHeader } from '../src/main/models/safetensors'

describe('LoRA audit regressions', () => {
  it('refuses unpaired tensors while accepting each supported family and lone alpha values', () => {
    const tensor = { dtype: 'F32', shape: [1, 1], offsets: [0, 4] as [number, number] }
    for (const [down, up] of [
      ['.lora_down.weight', '.lora_up.weight'],
      ['.lora_A.default.weight', '.lora_B.default.weight'],
      ['.lora_A.weight', '.lora_B.weight'],
      ['.lora.down.weight', '.lora.up.weight']
    ]) {
      const header: SafetensorsHeader = {
        tensors: { ['layer' + down]: tensor, ['layer' + up]: tensor, 'unused.alpha': tensor },
        metadata: {}, headerBytes: 0, fileBytes: 4, headerSha256: ''
      }
      expect(parseLoraLayers(header).unsupported).toBeNull()
      for (const extra of ['orphan.lora_down.weight', 'orphan.lora_up.weight', 'orphan.lora_A.weight', 'orphan.lora_B.weight', 'layer.bias']) {
        header.tensors[extra] = tensor
        expect(parseLoraLayers(header).unsupported).toContain(`tensor ${extra} is not part of an up/down pair`)
        delete header.tensors[extra]
      }
      header.tensors['layer.dora_scale'] = tensor
      expect(parseLoraLayers(header).unsupported).toContain('DoRA')
    }
  })

  it('rejects invalid shapes and offsets in safetensors headers', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'rulz-lora-audit-'))
    try {
      const path = join(dir, 'test.safetensors')
      const write = async (shape: number[], offsets: number[]): Promise<void> => {
        const json = Buffer.from(JSON.stringify({ t: { dtype: 'F32', shape, data_offsets: offsets } }))
        const length = Buffer.alloc(8)
        length.writeBigUInt64LE(BigInt(json.length))
        await writeFile(path, Buffer.concat([length, json, Buffer.alloc(4)]))
      }
      await write([1], [0, 4])
      expect((await readSafetensorsHeader(path)).tensors.t.offsets).toEqual([0, 4])
      for (const [shape, offsets] of [
        [[-1], [0, 4]], [[0.5], [0, 4]], [[Number.MAX_SAFE_INTEGER + 1], [0, 4]],
        [[1], [-1, 4]], [[1], [0, 0.5]], [[1], [0, Number.MAX_SAFE_INTEGER + 1]],
        [[1], [3, 2]], [[1], [0, 5]]
      ]) {
        await write(shape, offsets)
        await expect(readSafetensorsHeader(path)).rejects.toThrow(/Bad safetensors header/)
      }
    } finally {
      await rm(dir, { recursive: true, force: true })
    }
  })
})
