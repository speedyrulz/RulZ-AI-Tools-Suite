import hljs from 'highlight.js/lib/common'
import { Check, Copy } from 'lucide-react'
import { Children, isValidElement, memo, useMemo, useState, type ReactElement, type ReactNode } from 'react'
import ReactMarkdown, { type Components } from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { cn } from '../lib/format'

/** `live`: the text is still streaming, so skip language detection (it tries every grammar). */
function highlight(code: string, lang: string | undefined, live = false): string {
  try {
    if (lang && hljs.getLanguage(lang)) return hljs.highlight(code, { language: lang, ignoreIllegals: true }).value
    if (!live && code.length < 4000) return hljs.highlightAuto(code).value
  } catch {
    // fall through to plain text
  }
  return code.replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' })[c]!)
}

export function CodeBlock({ code, lang, live = false }: { code: string; lang?: string; live?: boolean }) {
  const [copied, setCopied] = useState(false)
  const html = useMemo(() => highlight(code, lang, live), [code, lang, live])
  return (
    <div className="group/code my-3 overflow-hidden rounded-lg border border-border bg-code">
      <div className="flex items-center justify-between border-b border-border px-3 py-1 text-[11px] text-faint">
        <span className="font-mono">{lang || 'text'}</span>
        <button
          className="flex items-center gap-1 rounded px-1.5 py-0.5 hover:bg-hover hover:text-text"
          onClick={() => {
            void navigator.clipboard.writeText(code)
            setCopied(true)
            setTimeout(() => setCopied(false), 1200)
          }}
        >
          {copied ? <Check size={12} /> : <Copy size={12} />}
          {copied ? 'Copied' : 'Copy'}
        </button>
      </div>
      <pre className="m-0 overflow-x-auto p-3 font-mono text-[12.5px] leading-relaxed">
        <code className="hljs" dangerouslySetInnerHTML={{ __html: html }} />
      </pre>
    </div>
  )
}

function makePre(live: boolean): Components['pre'] {
  return function Pre({ children }) {
    const child = Children.toArray(children)[0]
    if (!isValidElement(child)) return <pre>{children}</pre>
    const props = (child as ReactElement<{ className?: string; children?: ReactNode }>).props
    const lang = /language-([\w+#.-]+)/.exec(props.className ?? '')?.[1]
    const code = String(Children.toArray(props.children).join('')).replace(/\n$/, '')
    return <CodeBlock code={code} lang={lang} live={live} />
  }
}

const components: Components = {
  pre: makePre(false),
  a({ href, children }) {
    return (
      <a href={href} target="_blank" rel="noreferrer">
        {children}
      </a>
    )
  },
  img({ src, alt }) {
    // Remote images are blocked by the CSP; show the link instead.
    return <span className="text-faint">[image: {alt || String(src ?? '')}]</span>
  }
}

const liveComponents: Components = { ...components, pre: makePre(true) }

/** `live`: the text is still streaming in; unlabelled code blocks get highlighted once it is done. */
export const Markdown = memo(function Markdown({ text, className, live = false }: { text: string; className?: string; live?: boolean }) {
  return (
    <div className={cn('md', className)}>
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={live ? liveComponents : components}>
        {text}
      </ReactMarkdown>
    </div>
  )
})
