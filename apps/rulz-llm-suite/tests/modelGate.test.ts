// The model gate: one holder at a time, an idempotent release, and a sentence that says who holds it.
import { describe, expect, it } from 'vitest'
import { ModelGate } from '../src/main/modelGate'

describe('ModelGate', () => {
  it('hands out one release function, then the sentence for everyone else until it is called', () => {
    const gate = new ModelGate()
    expect(gate.busyReason()).toBeNull()
    const release = gate.take('adventure:abc')
    expect(typeof release).toBe('function')
    expect(gate.heldBy()).toBe('adventure:abc')
    const refused = gate.take('writing:def')
    expect(typeof refused).toBe('string')
    expect(refused).toContain('an adventure')
    expect(gate.busyReason()).toBe(refused)
    ;(release as () => void)()
    expect(gate.heldBy()).toBeNull()
    expect(gate.busyReason()).toBeNull()
    expect(typeof gate.take('writing:def')).toBe('function')
  })

  it('names a Writing action from a phone', () => {
    const gate = new ModelGate()
    gate.take('writing:x')
    expect(gate.busyReason()).toContain('Writing action from a phone')
  })

  it('release is idempotent: a second call does not free the model for the next holder', () => {
    const gate = new ModelGate()
    const first = gate.take('a:1') as () => void
    first()
    const second = gate.take('a:2') as () => void
    first()
    first()
    expect(gate.heldBy()).toBe('a:2')
    expect(typeof gate.take('a:3')).toBe('string')
    second()
    expect(gate.heldBy()).toBeNull()
  })
})
