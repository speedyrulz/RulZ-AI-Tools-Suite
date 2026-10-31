import { Github, Loader2 } from 'lucide-react'
import { memo, useEffect, useMemo, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import rehypeRaw from 'rehype-raw'
import rehypeSanitize, { defaultSchema } from 'rehype-sanitize'
import remarkGfm from 'remark-gfm'
import { createSlugger, prepareReadme, resolveReadmeUrl } from '@shared/readme'
import type { ReadmeDoc } from '@shared/types'
import { invoke } from '../lib/api'

// GitHub-like HTML in READMEs (<p align>, <picture>), without scripts or styles.
const schema = {
  ...defaultSchema,
  clobberPrefix: '',
  clobber: [],
  tagNames: [...(defaultSchema.tagNames ?? []), 'picture', 'source'],
  attributes: {
    ...defaultSchema.attributes,
    '*': [...(defaultSchema.attributes?.['*'] ?? []), 'align'],
    img: [...(defaultSchema.attributes?.img ?? []), 'width', 'height', 'align'],
    source: ['srcSet', 'media', 'type']
  }
}

interface HastNode {
  type: string
  tagName?: string
  value?: string
  properties?: Record<string, unknown>
  children?: HastNode[]
}

function textOf(node: HastNode): string {
  return node.type === 'text' ? (node.value ?? '') : (node.children ?? []).map(textOf).join('')
}

/** Points relative image paths at the app's files and gives headings GitHub's anchor ids. */
function rehypeReadme(doc: ReadmeDoc) {
  return () => (tree: HastNode) => {
    const slug = createSlugger()
    const walk = (node: HastNode): void => {
      const p = node.properties
      if (p && node.type === 'element') {
        if (typeof p.src === 'string') p.src = resolveReadmeUrl(p.src, doc.assetBase)
        if (typeof p.srcSet === 'string') p.srcSet = resolveReadmeUrl(p.srcSet.split(/\s+/)[0], doc.assetBase)
        if (Array.isArray(p.srcSet)) p.srcSet = resolveReadmeUrl(String(p.srcSet[0]).split(/\s+/)[0], doc.assetBase)
        if (/^h[1-6]$/.test(node.tagName ?? '')) p.id = slug(textOf(node))
      }
      node.children?.forEach(walk)
    }
    walk(tree)
  }
}

/** Memoized: parsing a long README is slow, and the app page re-renders with every install progress update. */
export const Readme = memo(function Readme({ appId }: { appId: string }) {
  const [doc, setDoc] = useState<ReadmeDoc | null>(null)
  const [error, setError] = useState<string | null>(null)
  const root = useRef<HTMLDivElement>(null)

  useEffect(() => {
    let alive = true
    setDoc(null)
    setError(null)
    invoke('apps:readme', appId).then(
      (d) => alive && setDoc(d),
      (e: Error) => alive && setError(e.message)
    )
    return () => {
      alive = false
    }
  }, [appId])

  const markdown = useMemo(() => (doc ? prepareReadme(doc.markdown) : ''), [doc])
  const rehypePlugins = useMemo(() => (doc ? [rehypeRaw, [rehypeSanitize, schema], rehypeReadme(doc)] : []), [doc])

  if (error) return <p className="text-sm text-danger">Couldn't load the README: {error}</p>
  if (!doc)
    return (
      <div className="flex items-center gap-2 py-10 text-sm text-faint">
        <Loader2 className="size-4 animate-spin" /> Loading README…
      </div>
    )

  const onLink = (href: string | undefined): void => {
    if (!href) return
    if (href.startsWith('#')) {
      root.current?.querySelector(`[id="${CSS.escape(decodeURIComponent(href.slice(1)))}"]`)?.scrollIntoView({ behavior: 'smooth' })
      return
    }
    const url = /^https?:\/\//i.test(href) ? href : doc.linkBase ? resolveReadmeUrl(href, doc.linkBase) : null
    if (url && /^https?:\/\//i.test(url)) void invoke('shell:openExternal', url)
  }

  return (
    <div ref={root}>
      <div className="md">
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          rehypePlugins={rehypePlugins as never}
          components={{
            a: ({ href, children }) => (
              <a
                href={href}
                onClick={(e) => {
                  e.preventDefault()
                  onLink(href)
                }}
              >
                {children}
              </a>
            ),
            img: ({ src, alt, width, height }) => (
              <img src={typeof src === 'string' ? src : undefined} alt={alt ?? ''} width={width} height={height} loading="lazy" />
            )
          }}
        >
          {markdown}
        </ReactMarkdown>
      </div>
      {doc.source === 'github' && (
        <p className="mt-8 flex items-center gap-1.5 border-t border-border pt-3 text-xs text-faint">
          <Github className="size-3.5" /> README from GitHub
        </p>
      )}
    </div>
  )
})
