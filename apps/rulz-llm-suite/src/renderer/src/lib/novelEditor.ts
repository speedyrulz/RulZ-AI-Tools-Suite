// CodeMirror 6 setup for the Writing page: a Markdown prose editor coloured from the app's CSS
// variables (so dark and light both work), plus the "pending AI change" view built on the unified
// merge view. The text is never reformatted: what the editor holds is what the store saves.
import { Annotation, Compartment, EditorState, type Extension } from '@codemirror/state'
import { EditorView, drawSelection, keymap, placeholder as placeholderExt } from '@codemirror/view'
import { defaultKeymap, history, historyKeymap } from '@codemirror/commands'
import { markdown } from '@codemirror/lang-markdown'
import { HighlightStyle, syntaxHighlighting } from '@codemirror/language'
import { highlightSelectionMatches, search, searchKeymap } from '@codemirror/search'
import { getChunks, getOriginalDoc, unifiedMergeView } from '@codemirror/merge'
import { tags as t } from '@lezer/highlight'

/** Marks transactions made by the app (loading text, showing a pending change), not by typing. */
const External = Annotation.define<boolean>()

const markdownStyle = HighlightStyle.define([
  { tag: t.heading, fontWeight: '650', color: 'var(--text)' },
  { tag: t.strong, fontWeight: '650' },
  { tag: t.emphasis, fontStyle: 'italic' },
  { tag: t.strikethrough, textDecoration: 'line-through' },
  { tag: t.link, color: 'var(--link)' },
  { tag: t.url, color: 'var(--link)' },
  { tag: t.quote, color: 'var(--muted)', fontStyle: 'italic' },
  { tag: t.monospace, fontFamily: 'var(--font-mono)', fontSize: '0.92em' },
  { tag: [t.processingInstruction, t.contentSeparator, t.meta], color: 'var(--faint)' }
])

function themeFor(dark: boolean, wide: boolean): Extension {
  return EditorView.theme(
    {
      '&': { height: '100%', backgroundColor: 'transparent', color: 'var(--text)', fontSize: '16px' },
      '&.cm-focused': { outline: '2px solid var(--accent)', outlineOffset: '-2px' },
      '.cm-scroller': { fontFamily: 'var(--font-sans)', lineHeight: '1.75', overflow: 'auto' },
      '.cm-content': {
        maxWidth: wide ? '72ch' : 'none',
        margin: '0 auto',
        padding: wide ? '28px 24px 40vh' : '10px 12px',
        caretColor: 'var(--accent)'
      },
      '.cm-line': { padding: '0' },
      '.cm-cursor, .cm-dropCursor': { borderLeftColor: 'var(--accent)', borderLeftWidth: '2px' },
      '.cm-selectionBackground, &.cm-focused > .cm-scroller > .cm-selectionLayer .cm-selectionBackground': {
        backgroundColor: 'color-mix(in srgb, var(--accent) 30%, transparent)'
      },
      '.cm-placeholder': { color: 'var(--faint)' },
      '.cm-selectionMatch': { backgroundColor: 'var(--subtle)' },
      '.cm-searchMatch': { backgroundColor: 'color-mix(in srgb, var(--warn) 30%, transparent)', outline: 'none' },
      '.cm-searchMatch.cm-searchMatch-selected': { backgroundColor: 'color-mix(in srgb, var(--accent) 35%, transparent)' },
      '.cm-panels': { backgroundColor: 'var(--panel)', color: 'var(--text)', borderColor: 'var(--border)' },
      '.cm-panels.cm-panels-bottom': { borderTop: '1px solid var(--border)' },
      '.cm-panels.cm-panels-top': { borderBottom: '1px solid var(--border)' },
      '.cm-panel.cm-search': { padding: '8px 10px', fontSize: '13px' },
      '.cm-panel.cm-search input, .cm-panel.cm-search .cm-textfield': {
        backgroundColor: 'var(--bg)',
        border: '1px solid var(--border)',
        borderRadius: '6px',
        color: 'var(--text)',
        padding: '3px 8px'
      },
      '.cm-panel.cm-search input:focus': { outline: '2px solid var(--accent)', outlineOffset: '1px' },
      '.cm-panel.cm-search .cm-button, .cm-panel.cm-search button': {
        backgroundImage: 'none',
        backgroundColor: 'var(--panel-2)',
        border: '1px solid var(--border)',
        borderRadius: '6px',
        color: 'var(--text)',
        padding: '2px 10px'
      },
      '.cm-panel.cm-search label': { color: 'var(--muted)' },
      '.cm-panel.cm-search [name=close]': { color: 'var(--muted)', backgroundColor: 'transparent', border: 'none' },
      // Pending change (unified merge view)
      '.cm-changedLine': { backgroundColor: 'color-mix(in srgb, var(--ok) 11%, transparent)' },
      '.cm-insertedLine, ins.cm-insertedLine, ins.cm-changedText': {
        backgroundColor: 'color-mix(in srgb, var(--ok) 16%, transparent)',
        textDecoration: 'none'
      },
      '.cm-deletedChunk': {
        backgroundColor: 'color-mix(in srgb, var(--danger) 10%, transparent)',
        padding: '4px 0 4px 6px',
        borderLeft: '3px solid var(--danger)'
      },
      '.cm-deletedChunk .cm-deletedLine, .cm-deletedChunk del': {
        backgroundColor: 'transparent',
        color: 'var(--muted)',
        textDecorationColor: 'var(--danger)'
      },
      '.cm-deletedChunk .cm-deletedText': { backgroundColor: 'color-mix(in srgb, var(--danger) 22%, transparent)' },
      '.cm-chunkButtons': { display: 'flex', gap: '6px', position: 'sticky', top: '4px', justifyContent: 'flex-end', marginBottom: '4px' },
      '.rz-chunk-btn': {
        height: '28px',
        padding: '0 12px',
        borderRadius: '6px',
        border: '1px solid var(--border)',
        backgroundColor: 'var(--panel-2)',
        color: 'var(--text)',
        fontSize: '13px',
        fontFamily: 'var(--font-sans)',
        cursor: 'pointer'
      },
      '.rz-chunk-btn:hover': { backgroundColor: 'var(--hover)', borderColor: 'var(--border-strong)' },
      '.rz-chunk-btn:focus-visible': { outline: '2px solid var(--accent)', outlineOffset: '2px' },
      '.rz-chunk-accept': { borderColor: 'color-mix(in srgb, var(--ok) 55%, transparent)' },
      '.cm-changeGutter': { width: '3px', backgroundColor: 'transparent' },
      '.cm-gutters': { backgroundColor: 'transparent', border: 'none' }
    },
    { dark }
  )
}

export interface PendingView {
  /** Text the editor shows as the baseline; each change is a deviation from it. */
  original: string
  /** The proposed whole document. */
  proposed: string
  /** Called before Accept on a single change; return false to refuse (the caller shows why). */
  canAccept: () => boolean
  onRefuse: () => void
  /** Every change has been accepted or rejected; `text` is the document as it stands. */
  onResolved: (text: string) => void
}

export interface EditorHandle {
  view: EditorView
  /** The text the last change callback delivered (or the last `setText`), to skip echoes. */
  lastText: string
  setText: (text: string) => void
  setReadOnly: (on: boolean) => void
  setDark: (dark: boolean) => void
  /** Shows a pending change: the document becomes `proposed`, compared with `original`. */
  showPending: (p: PendingView) => void
  /** Ends the pending view; the document becomes `text` (what the store holds now). */
  clearPending: (text: string) => void
  /** The document with the open changes accepted. */
  pendingAcceptAll: () => string
  /** The baseline with only the changes accepted so far: what Reject all leaves. */
  pendingRejectAll: () => string
  selection: () => { from: number; to: number }
  focus: () => void
  destroy: () => void
}

export interface EditorOptions {
  doc: string
  dark: boolean
  readOnly: boolean
  /** Wide prose layout (scene editor) or a plain compact one (bible body, style). */
  prose: boolean
  placeholder?: string
  onChange: (text: string) => void
  onSelection?: (sel: { from: number; to: number; text: string }) => void
}

export function createEditor(parent: HTMLElement, opts: EditorOptions): EditorHandle {
  const readOnlyComp = new Compartment()
  const themeComp = new Compartment()
  const mergeComp = new Compartment()
  let pending: PendingView | null = null
  let handle: EditorHandle

  const listener = EditorView.updateListener.of((u) => {
    const external = u.transactions.some((tr) => tr.annotation(External))
    if (u.docChanged && !external && !pending) {
      handle.lastText = u.state.doc.toString()
      opts.onChange(handle.lastText)
    }
    if (opts.onSelection && (u.selectionSet || u.docChanged)) {
      const s = u.state.selection.main
      opts.onSelection({ from: s.from, to: s.to, text: s.empty ? '' : u.state.sliceDoc(s.from, s.to) })
    }
    if (pending && (u.docChanged || u.transactions.some((tr) => tr.effects.length > 0))) {
      const chunks = getChunks(u.state)
      if (chunks && chunks.chunks.length === 0) {
        const done = pending
        const text = u.state.doc.toString()
        // After the update: resolving swaps the view's configuration, which a listener may not do.
        queueMicrotask(() => {
          if (pending === done) done.onResolved(text)
        })
      }
    }
  })

  const state = EditorState.create({
    doc: opts.doc,
    extensions: [
      history(),
      drawSelection(),
      EditorView.lineWrapping,
      EditorState.allowMultipleSelections.of(true),
      markdown({ addKeymap: false }),
      syntaxHighlighting(markdownStyle),
      search({ top: true }),
      highlightSelectionMatches(),
      keymap.of([...searchKeymap, ...historyKeymap, ...defaultKeymap]),
      opts.placeholder ? placeholderExt(opts.placeholder) : [],
      themeComp.of(themeFor(opts.dark, opts.prose)),
      readOnlyComp.of([EditorState.readOnly.of(opts.readOnly), EditorView.editable.of(!opts.readOnly)]),
      mergeComp.of([]),
      EditorView.contentAttributes.of({ 'aria-label': opts.prose ? 'Scene text' : 'Text', spellcheck: 'true' }),
      listener
    ]
  })
  const view = new EditorView({ state, parent })

  let readOnlyBase = opts.readOnly
  const applyReadOnly = (on: boolean): void => {
    view.dispatch({
      effects: readOnlyComp.reconfigure([EditorState.readOnly.of(on), EditorView.editable.of(!on)])
    })
  }

  const mergeExt = (p: PendingView): Extension =>
    unifiedMergeView({
      original: p.original,
      gutter: true,
      highlightChanges: true,
      syntaxHighlightDeletions: false,
      mergeControls: (type, action) => {
        const b = document.createElement('button')
        b.type = 'button'
        b.className = `rz-chunk-btn rz-chunk-${type}`
        b.textContent = type === 'accept' ? 'Accept' : 'Reject'
        b.setAttribute('aria-label', type === 'accept' ? 'Accept this change' : 'Reject this change')
        // mousedown, like the stock buttons, so the editor keeps its selection.
        b.onmousedown = (e) => {
          if (type === 'accept' && !p.canAccept()) {
            e.preventDefault()
            p.onRefuse()
            return
          }
          action(e)
        }
        b.onkeydown = (e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            if (type === 'accept' && !p.canAccept()) {
              e.preventDefault()
              p.onRefuse()
              return
            }
            action(new MouseEvent('mousedown'))
          }
        }
        return b
      }
    })

  handle = {
    view,
    lastText: opts.doc,
    setText(text) {
      handle.lastText = text
      view.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: text }, annotations: External.of(true) })
    },
    setReadOnly(on) {
      readOnlyBase = on
      if (!pending) applyReadOnly(on)
    },
    setDark(dark) {
      view.dispatch({ effects: themeComp.reconfigure(themeFor(dark, opts.prose)) })
    },
    showPending(p) {
      // Off first, so the merge state restarts from this baseline instead of keeping the old one.
      view.dispatch({ effects: mergeComp.reconfigure([]) })
      pending = p
      view.dispatch({
        changes: { from: 0, to: view.state.doc.length, insert: p.proposed },
        annotations: External.of(true),
        effects: [readOnlyComp.reconfigure([EditorState.readOnly.of(true), EditorView.editable.of(false)])]
      })
      view.dispatch({ effects: mergeComp.reconfigure(mergeExt(p)) })
    },
    clearPending(text) {
      pending = null
      view.dispatch({ effects: mergeComp.reconfigure([]) })
      applyReadOnly(readOnlyBase)
      handle.lastText = text
      if (view.state.doc.length !== text.length || view.state.doc.toString() !== text) {
        view.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: text }, annotations: External.of(true) })
      }
    },
    pendingAcceptAll() {
      return view.state.doc.toString()
    },
    pendingRejectAll() {
      return getOriginalDoc(view.state).toString()
    },
    selection() {
      const s = view.state.selection.main
      return { from: s.from, to: s.to }
    },
    focus() {
      view.focus()
    },
    destroy() {
      pending = null
      view.destroy()
    }
  }
  return handle
}
