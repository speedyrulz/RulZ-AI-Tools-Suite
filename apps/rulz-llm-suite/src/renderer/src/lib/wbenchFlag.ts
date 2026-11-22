// Whether a writing benchmark run is going. A module with no imports, so the chat, batch and hardware-benchmark
// stores can check it without an import cycle through stores/wbench.ts and lib/aiBusy.ts.
let running = false

export const setWbenchRunning = (value: boolean): void => {
  running = value
}

export const wbenchRunning = (): boolean => running

export const WBENCH_BUSY = 'A writing benchmark is running. Stop it on the Writing page first.'
