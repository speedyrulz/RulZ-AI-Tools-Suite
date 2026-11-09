import hljs from 'highlight.js/lib/common'
import { Check, Copy } from 'lucide-react'
import { Children, isValidElement, memo, useMemo, useState, type ReactElement, type ReactNode } from 'react'
import ReactMarkdown, { type Components } from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { cn } from '../lib/format'
import { openInBrowser } from '../lib/native'

function highlight(code: string, lang: string | undefined): string {
  try {
    if (lang && hljs.getLanguage(lang)) return hljs.highlight(code, { language: lang, ignoreIllegals: true }).value
    if (code.length < 4000) return hljs.highlightAuto(code).value
  } catch {
    // fall through to plain text
  }
  return code.replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' })[c]!)
}

export function CodeBlock({ code, lang }: { code: string; lang?: string }) {
  const [copied, setCopied] = useState(false)
  const html = useMemo(() => highlight(code, lang), [code, lang])
  return (
    <div className="group/code my-3 overflow-hidden rounded-lg border border-border bg-code">
      <div className="flex items-center justify-between border-b border-border px-3 py-1 text-[11px] text-faint">
        <span className="font-mono">{lang || 'text'}</span>
        <button
          className="flex items-center gap-1 rounded px-2 py-1 active:bg-hover"
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

const components: Components = {
  pre({ children }) {
    const child = Children.toArray(children)[0]
    if (!isValidElement(child)) return <pre>{children}</pre>
    const props = (child as ReactElement<{ className?: string; children?: ReactNode }>).props
    const lang = /language-([\w+#.-]+)/.exec(props.className ?? '')?.[1]
    const code = String(Children.toArray(props.children).join('')).replace(/\n$/, '')
    return <CodeBlock code={code} lang={lang} />
  },
  a({ href, children }) {
    return (
      <a
        href={href}
        onClick={(e) => {
          e.preventDefault()
          if (href && /^https?:\/\//i.test(href)) void openInBrowser(href)
        }}
      >
        {children}
      </a>
    )
  },
  img({ src, alt }) {
    // Remote images are blocked by the CSP; show the link instead.
    return <span className="text-faint">[image: {alt || String(src ?? '')}]</span>
  }
}

export const Markdown = memo(function Markdown({ text, className }: { text: string; className?: string }) {
  return (
    <div className={cn('md', className)}>
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
        {text}
      </ReactMarkdown>
    </div>
  )
})
