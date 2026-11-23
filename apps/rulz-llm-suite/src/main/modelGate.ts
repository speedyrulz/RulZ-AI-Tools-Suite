// One model job at a time, on the whole computer, for the jobs main runs itself (adventure jobs and the phone's
// Writing runs). `chat.busy` is false while a job is preparing (loading the model, counting tokens) and in the gap
// between two requests of one job, so it cannot tell that a job is under way. A job takes the gate when it starts
// and gives it back when it ends; whoever asks while it is held gets a sentence saying who holds it.
// The desktop's own chat and Writing pages do not use the gate (known limit, docs/adventure-mode-design.md).

export interface ModelGateLike {
  /** Returns the release function, or the sentence saying who holds the model. */
  take(holder: string): (() => void) | string
}

/** "adventure:<id>" and "writing:<id>" read as what is running, not as an id. */
function describe(holder: string): string {
  if (holder.startsWith('adventure:')) return 'an adventure'
  if (holder.startsWith('writing:')) return 'a Writing action from a phone'
  return 'another job'
}

export class ModelGate implements ModelGateLike {
  private holder: string | null = null
  private seq = 0

  take(holder: string): (() => void) | string {
    if (this.holder !== null) return this.sentence()
    const mine = ++this.seq
    this.holder = holder
    let released = false
    return () => {
      // Releasing twice (an error path and a finally) must not free the model for the next holder.
      if (released) return
      released = true
      if (this.seq === mine) this.holder = null
    }
  }

  /** Who holds the model, or null. */
  heldBy(): string | null {
    return this.holder
  }

  /** The sentence for a refused request, or null when the gate is free. */
  busyReason(): string | null {
    return this.holder === null ? null : this.sentence()
  }

  private sentence(): string {
    return `LLM Suite is already writing ${describe(this.holder ?? '')} on the computer.`
  }
}
