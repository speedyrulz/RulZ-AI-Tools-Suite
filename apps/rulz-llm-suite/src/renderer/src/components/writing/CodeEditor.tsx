import { useEffect, useRef, useState } from 'react'
import { createEditor, type EditorHandle, type PendingView } from '../../lib/novelEditor'
import { registerEditor } from '../../stores/novel'

/** True while the app is in its dark theme (the root element carries the `dark` class). */
export function useIsDark(): boolean {
  const [dark, setDark] = useState(() => document.documentElement.classList.contains('dark'))
  useEffect(() => {
    const obs = new MutationObserver(() => setDark(document.documentElement.classList.contains('dark')))
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] })
    return () => obs.disconnect()
  }, [])
  return dark
}

export interface PendingProps extends PendingView {
  /** Changes with the pending change and the alternative shown; a new key restarts the view. */
  key: string
}

/**
 * CodeMirror host. The editor owns its document while the user types and reports edits through
 * `onChange`; `text` only matters when it differs from what the editor last reported (a reload, a
 * restore, an accepted change). `path` identifies the file: a new path starts a new editor.
 */
export function CodeEditor({
  path,
  text,
  readOnly,
  prose,
  placeholder,
  register,
  cursorAtEnd,
  pending,
  onChange,
  onSelection,
  className
}: {
  path: string
  text: string
  readOnly: boolean
  prose: boolean
  placeholder?: string
  /** Make the editor reachable by actions (selection and pending changes). */
  register?: boolean
  cursorAtEnd?: boolean
  pending?: PendingProps | null
  onChange: (text: string) => void
  onSelection?: (sel: { from: number; to: number }) => void
  className?: string
}) {
  const host = useRef<HTMLDivElement>(null)
  const handle = useRef<EditorHandle | null>(null)
  const dark = useIsDark()
  const latest = useRef({ text, onChange, onSelection, readOnly, dark })
  latest.current = { text, onChange, onSelection, readOnly, dark }
  const showing = useRef<string | null>(null)

  useEffect(() => {
    const h = createEditor(host.current!, {
      doc: latest.current.text,
      dark: latest.current.dark,
      readOnly: latest.current.readOnly,
      prose,
      placeholder,
      onChange: (t) => latest.current.onChange(t),
      onSelection: (s) => latest.current.onSelection?.({ from: s.from, to: s.to })
    })
    handle.current = h
    showing.current = null
    if (cursorAtEnd) h.view.dispatch({ selection: { anchor: h.view.state.doc.length } })
    if (register) registerEditor(path, h)
    return () => {
      if (register) registerEditor(path, null)
      h.destroy()
      handle.current = null
    }
    // A new path is a new document.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [path])

  // A pending change is shown (or ended) before the text is compared, so an accepted change doesn't flash.
  const pendingKey = pending?.key ?? null
  useEffect(() => {
    const h = handle.current
    if (!h) return
    if (pending && showing.current !== pending.key) {
      showing.current = pending.key
      h.showPending(pending)
    } else if (!pending && showing.current) {
      showing.current = null
      h.clearPending(latest.current.text)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pendingKey, path])

  useEffect(() => {
    const h = handle.current
    if (!h || showing.current) return
    if (text !== h.lastText && (h.view.state.doc.length !== text.length || h.view.state.doc.toString() !== text)) h.setText(text)
  }, [text, path])

  useEffect(() => {
    handle.current?.setReadOnly(readOnly)
  }, [readOnly, path])

  useEffect(() => {
    handle.current?.setDark(dark)
  }, [dark, path])

  return <div ref={host} className={className ?? 'h-full min-h-0'} />
}
