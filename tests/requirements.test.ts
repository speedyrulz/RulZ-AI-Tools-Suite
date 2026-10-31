import { describe, expect, it } from 'vitest'
import { parseGpuQuery } from '../src/main/requirements'

describe('parseGpuQuery (CUDA 13 needs driver 580+ and compute 7.5+)', () => {
  it('accepts a current card and driver', () => {
    const r = parseGpuQuery('NVIDIA GeForce RTX 4090, 24564, 581.57, 8.9')
    expect(r.ok).toBe(true)
    expect(r.detail).toBe('RTX 4090 (24 GB) · driver 581.57')
  })

  it('rejects a driver older than 580, saying what to do', () => {
    const r = parseGpuQuery('NVIDIA GeForce RTX 3090, 24576, 566.36, 8.6')
    expect(r.ok).toBe(false)
    expect(r.detail).toMatch(/driver 566\.36 is too old/)
    expect(r.detail).toMatch(/580 or newer/)
  })

  it('rejects cards older than Turing', () => {
    const r = parseGpuQuery('NVIDIA GeForce GTX 1080 Ti, 11264, 581.57, 6.1')
    expect(r.ok).toBe(false)
    expect(r.detail).toMatch(/GTX 1080 Ti is compute 6\.1/)
    expect(r.detail).toMatch(/RTX 20/)
  })

  it('accepts a GTX 16-series card (compute 7.5)', () => {
    expect(parseGpuQuery('NVIDIA GeForce GTX 1660 SUPER, 6144, 580.88, 7.5').ok).toBe(true)
  })

  it('is fine when one of several cards can run CUDA 13, and lists only those', () => {
    const r = parseGpuQuery('NVIDIA GeForce GTX 1070, 8192, 581.57, 6.1\r\nNVIDIA GeForce RTX 5090, 32607, 581.57, 12.0')
    expect(r.ok).toBe(true)
    expect(r.detail).toBe('RTX 5090 (32 GB) · driver 581.57')
  })

  it('handles old drivers that answer without compute_cap', () => {
    const r = parseGpuQuery('NVIDIA GeForce GTX 980, 4096 MiB, 472.12')
    expect(r.ok).toBe(false)
    expect(r.detail).toMatch(/too old/)
  })

  it('reports no GPU for an empty answer', () => {
    expect(parseGpuQuery('').ok).toBe(false)
  })
})
