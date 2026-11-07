import { useEffect, useRef, useState, type DragEvent } from 'react'

const hasFiles = (e: DragEvent): boolean => e.dataTransfer.types.includes('Files')

/** Browsers repeat dragover every few hundred ms while a drag is over us; silence this long means it ended. */
const STALE_MS = 1000

/**
 * A drop target for files, with a flag for the "drop here" overlay. Moving between child
 * elements fires enter and leave for each, so a counter tracks whether the pointer is still
 * inside. A timer also clears the overlay when no leave reaches us at all: a drag cancelled with
 * Esc, or the element under the pointer replaced mid-drag by a streaming reply.
 */
export function useFileDrop(onFiles: (files: File[]) => void): {
  active: boolean
  handlers: {
    onDragEnter: (e: DragEvent) => void
    onDragOver: (e: DragEvent) => void
    onDragLeave: (e: DragEvent) => void
    onDrop: (e: DragEvent) => void
  }
} {
  const [active, setActive] = useState(false)
  const depth = useRef(0)
  const timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)

  const reset = (): void => {
    depth.current = 0
    clearTimeout(timer.current)
    setActive(false)
  }
  useEffect(() => () => clearTimeout(timer.current), [])

  return {
    active,
    handlers: {
      onDragEnter: (e) => {
        if (!hasFiles(e)) return
        depth.current++
        setActive(true)
      },
      onDragOver: (e) => {
        if (!hasFiles(e)) return
        e.preventDefault()
        setActive(true)
        clearTimeout(timer.current)
        timer.current = setTimeout(reset, STALE_MS)
      },
      onDragLeave: (e) => {
        if (!hasFiles(e)) return
        depth.current = Math.max(0, depth.current - 1)
        if (depth.current === 0) reset()
      },
      onDrop: (e) => {
        e.preventDefault()
        reset()
        onFiles([...e.dataTransfer.files])
      }
    }
  }
}
