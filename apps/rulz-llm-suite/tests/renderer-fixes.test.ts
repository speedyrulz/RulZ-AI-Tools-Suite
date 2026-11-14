import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import ReactMarkdown from 'react-markdown'
import rehypeKatex from 'rehype-katex'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import { describe, expect, it } from 'vitest'
import { isStaticReTool, outputInsideTarget, shouldGradePractice } from '@shared/re'
import { toolTokens } from '@shared/tools'
import type { LoadedInstance } from '@shared/types'
import { pickActive } from '../src/renderer/src/lib/activeModel'
import { liveBlocks } from '../src/renderer/src/lib/mdBlocks'

// react-markdown puts "\n" text between top-level blocks (also before raw HTML, which it shows as
// text); .md has no pre-wrap, so they never show.
const html = (md: string): string =>
  renderToStaticMarkup(
    createElement(ReactMarkdown, { remarkPlugins: [remarkGfm, [remarkMath, { singleDollarTextMath: true }]], rehypePlugins: [rehypeKatex] }, md)
  )
    .replace(/>\n+</g, '><')
    .replace(/(<\/[a-z0-9]+>)\n+/g, '$1')
    .replace(/\n+(<[a-z])/g, '$1')

const SAMPLES: Record<string, string> = {
  paragraphs: '# Title\n\nFirst paragraph with **bold** and `code`.\n\nSecond one, $x^2$ inline.\n\n---\n\n> a quote\n\n| a | b |\n|---|---|\n| 1 | 2 |',
  fenceWithBlankLines: 'Look:\n\n```c\nint main(void) {\n\n  return 0;\n}\n```\n\nAfter the code.',
  tildeFence: '~~~\none\n\n```\nstill code\n~~~\n\nout',
  looseList: '1. first\n\n2. second\n\n3. third\n\nThen text.',
  bulletsWithContinuation: '- item one\n\n  more about one\n\n- item two\n\n      indented code in item\n\nEnd.',
  mixedLists: '- a\n- b\n\n1. c\n\n2. d',
  displayMath: 'Energy:\n\n$$\nE = mc^2\n\n+ 1\n$$\n\nDone.',
  // What normalizeMath makes of "Sum: \[ \sum_i x_i \]" and "\(y\)".
  latexBrackets: 'Sum: \n$$\n\\sum_i x_i\n$$\n\n\nand $y$.\n\nNext.',
  indentedCode: 'Para.\n\n    code line\n\n    more code\n\nText.',
  refInQuote: 'See [x] here.\n\n> [x]: https://example.com\n\nAnd [x] again.',
  mathClosedOnlyByDollars: '$$\nx = 1\n\ny = 2 $$\n\nAfter.\n\n$$\n\nEnd.',
  longerMathFence: '$$$\na\n$$\n\nb\n$$$\n\nc',
  htmlComment: 'Text.\n\n<!-- hidden\n\nstill hidden\n-->\n\nShown.\n\n<!-- one line -->\n\nLast.',
  htmlPre: 'A.\n\n<pre>\nkeep\n\nthis\n</pre>\n\nB.\n\n<script>x()</script>\n\nC.',
  doctypeAndCdata: '<!DOCTYPE html>\n\nP.\n\n<![CDATA[\na\n\nb\n]]>\n\nQ.\n\n<?php\n\n?>\n\nR.',
  headingsAndRules: 'Intro\n\n## A\n\ntext\n\n***\n\n## B\n\n- x\n\n  - nested\n\n- y'
}

describe('live Markdown blocks', () => {
  it('render the same block by block as in one piece', () => {
    for (const [name, md] of Object.entries(SAMPLES)) {
      const blocks = liveBlocks(md)
      expect(blocks.map(html).join(''), name).toBe(html(md))
    }
  })

  it('render the same at every point of a stream', () => {
    for (const [name, md] of Object.entries(SAMPLES)) {
      for (let cut = 1; cut <= md.length; cut += 7) {
        const part = md.slice(0, cut)
        expect(liveBlocks(part).map(html).join(''), `${name} @${cut}`).toBe(html(part))
      }
    }
  }, 60_000) // hundreds of renders; slow on a busy machine

  it('split where it is safe and keep everything else together', () => {
    expect(liveBlocks('a\n\nb\n\nc')).toEqual(['a', 'b', 'c'])
    expect(liveBlocks('```\nx\n\ny\n```\n\nz')).toEqual(['```\nx\n\ny\n```', 'z'])
    expect(liveBlocks('- a\n\n- b\n\nc')).toEqual(['- a\n\n- b', 'c'])
    expect(liveBlocks('- a\n\n  more\n\nc')).toEqual(['- a\n\n  more', 'c'])
    expect(liveBlocks('$$\nx\n\ny\n$$\n\nz')).toEqual(['$$\nx\n\ny\n$$', 'z'])
    // An unfinished fence keeps the rest of the stream in its block.
    expect(liveBlocks('a\n\n```\ncode\n\nmore')).toEqual(['a', '```\ncode\n\nmore'])
    // Reference links and footnotes reach across blocks.
    expect(liveBlocks('See [x].\n\n[x]: https://example.com')).toHaveLength(1)
    expect(liveBlocks('Note[^1].\n\n[^1]: the note')).toHaveLength(1)
    // Definitions inside a quote or list item count for the whole text too.
    expect(liveBlocks('See [x].\n\n> [x]: https://example.com')).toHaveLength(1)
    expect(liveBlocks('See [y].\n\n- [y]: https://example.com')).toHaveLength(1)
    expect(liveBlocks('See [z].\n\n> - [z]: https://example.com')).toHaveLength(1)
    expect(liveBlocks('no blank lines\nhere')).toEqual(['no blank lines\nhere'])
  })
})

describe('practice grading', () => {
  const practice = { taskId: 't1' } as never
  it('grades a session the app has not graded yet', () => {
    expect(shouldGradePractice({ practice, outcome: 'unrated' }, 'no answer here')).toBe(true)
    expect(shouldGradePractice({ practice, outcome: 'unrated' }, 'ANSWER: 0x1234')).toBe(true)
  })
  it('keeps an automatic grade until a new ANSWER line comes', () => {
    expect(shouldGradePractice({ practice, outcome: 'solved', autoGraded: true }, 'The hash routine rotates by 5 bits.')).toBe(false)
    expect(shouldGradePractice({ practice, outcome: 'failed', autoGraded: true }, 'ANSWER: 0xbeef')).toBe(true)
  })
  it('never changes a grade set by hand, nor grades ordinary sessions', () => {
    expect(shouldGradePractice({ practice, outcome: 'failed' }, 'ANSWER: 1')).toBe(false)
    expect(shouldGradePractice({ outcome: 'unrated' }, 'ANSWER: 1')).toBe(false)
  })
})

describe('clone output folder', () => {
  it('may not be the target or inside it, whatever the separators or case', () => {
    expect(outputInsideTarget('C:\\app', 'C:\\app')).toBe(true)
    expect(outputInsideTarget('C:/App/', 'c:\\app')).toBe(true)
    expect(outputInsideTarget('C:\\app\\clone', 'C:\\app')).toBe(true)
    expect(outputInsideTarget('C:/app/clone', 'C:\\app\\')).toBe(true)
    expect(outputInsideTarget('C:\\app', '"C:\\app"')).toBe(true)
  })
  it('may be beside the target, or hold a target file', () => {
    expect(outputInsideTarget('C:\\app-clone', 'C:\\app')).toBe(false)
    expect(outputInsideTarget('C:\\clones', 'C:\\app\\tool.exe')).toBe(false)
    expect(outputInsideTarget('C:\\app', 'C:\\app\\tool.exe')).toBe(false)
    expect(outputInsideTarget('', 'C:\\app')).toBe(false)
  })
})

describe('RE auto-approval', () => {
  it('approves only the calculator tool of a server named calc', () => {
    expect(isStaticReTool('calc', 'calc')).toBe(true)
    expect(isStaticReTool('calc', 'run_shell')).toBe(false)
  })
})

describe('active model while a model loads', () => {
  const inst = (modelId: string, status: LoadedInstance['status'], startedAt: number): LoadedInstance =>
    ({ instanceId: `${modelId}-${startedAt}`, modelId, status, startedAt }) as LoadedInstance

  it('stays on a model being reloaded while its old instance stops', () => {
    const stopping = [inst('A', 'stopping', 1), inst('B', 'ready', 2)]
    expect(pickActive(stopping, 'A', 'A')).toBe('A')
    expect(pickActive([inst('B', 'ready', 2), inst('A', 'loading', 3)], 'A', 'A')).toBe('A')
    // Without the load in progress it moves to what is still running, as before.
    expect(pickActive(stopping, 'A', null)).toBe('B')
  })

  it('moves to the newest running model when the active one is gone', () => {
    expect(pickActive([inst('B', 'ready', 2), inst('C', 'ready', 5)], 'A', null)).toBe('C')
    expect(pickActive([], 'A', null)).toBe(null)
  })
})

describe('tool token estimate', () => {
  it('gives the same number from the cache', () => {
    const t = { qualifiedName: 'rea__xrefs', description: 'Cross references', inputSchema: { type: 'object' } }
    const first = toolTokens(t)
    expect(toolTokens(t)).toBe(first)
    expect(toolTokens({ ...t, description: `${t.description} with a much longer description` })).toBeGreaterThan(first)
  })
})
