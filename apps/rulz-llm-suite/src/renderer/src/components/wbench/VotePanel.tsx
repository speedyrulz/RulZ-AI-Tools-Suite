import { ArrowRight, Check, Equal, SkipForward } from 'lucide-react'
import { useEffect } from 'react'
import type { BenchVote } from '@shared/writingBench'
import { cn } from '../../lib/format'
import type { PairSide, PairView } from '../../lib/wbenchRunner'
import { Badge, Button, Kbd } from '../ui'

const VOTE_WORDS: Record<BenchVote, string> = { a: 'You chose A.', b: 'You chose B.', tie: 'You called it a tie.', skip: 'You skipped this pair.' }

function Side({ side, chosen }: { side: PairSide; chosen: boolean }) {
  return (
    <section className={cn('flex min-h-48 min-w-0 flex-col rounded-lg border bg-panel', chosen ? 'border-ok/50' : 'border-border')} aria-label={`Output ${side.letter}`}>
      <header className="flex items-center gap-2 border-b border-border px-3 py-1.5">
        <span className="text-[13px] font-semibold">{side.letter}</span>
        {side.candidate !== null && <span className="min-w-0 truncate text-[12.5px] text-muted">{side.candidate}</span>}
        {chosen && (
          <Badge tone="ok" className="ml-auto">
            <Check size={12} aria-hidden /> Your pick
          </Badge>
        )}
        {side.finish === 'length' && (
          <Badge tone="warn" className={chosen ? '' : 'ml-auto'}>
            Cut off at the token limit
          </Badge>
        )}
      </header>
      <div className="max-h-[28rem] flex-1 overflow-y-auto px-3 py-2.5 text-sm leading-relaxed whitespace-pre-wrap text-body">{side.text.trim() || <span className="text-faint">(empty)</span>}</div>
    </section>
  )
}

/** One blind pair: two outputs side by side, the vote buttons, and the names once voted. */
export function VotePanel({ view, onVote, onNext, hasNext }: { view: PairView; onVote: (vote: BenchVote) => void; onNext: () => void; hasNext: boolean }) {
  const { voted } = view
  useEffect(() => {
    const onKey = (e: KeyboardEvent): void => {
      if (e.ctrlKey || e.metaKey || e.altKey) return
      const el = e.target as HTMLElement | null
      if (el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.tagName === 'SELECT' || el.isContentEditable)) return
      const k = e.key.toLowerCase()
      if (!voted) {
        const vote: BenchVote | null = k === '1' ? 'a' : k === '2' ? 'b' : k === '3' ? 'tie' : k === 's' ? 'skip' : null
        if (vote) {
          e.preventDefault()
          onVote(vote)
        }
      } else if (k === 'n') {
        e.preventDefault()
        onNext()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [voted, onVote, onNext])

  return (
    <div className="space-y-3">
      <div className="rounded-lg border border-border bg-panel-2/50 px-3 py-2">
        <div className="text-[11px] font-semibold tracking-wide text-faint uppercase">Prompt{view.promptTitle ? `: ${view.promptTitle}` : ''}</div>
        <p className="mt-0.5 text-[13px] whitespace-pre-wrap text-body">{view.promptText}</p>
      </div>
      <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
        <Side side={view.a} chosen={view.vote === 'a'} />
        <Side side={view.b} chosen={view.vote === 'b'} />
      </div>
      {!voted ? (
        <div className="flex flex-wrap items-center gap-2">
          <Button onClick={() => onVote('a')}>
            A is better <Kbd>1</Kbd>
          </Button>
          <Button onClick={() => onVote('b')}>
            B is better <Kbd>2</Kbd>
          </Button>
          <Button onClick={() => onVote('tie')}>
            <Equal size={16} aria-hidden /> Tie <Kbd>3</Kbd>
          </Button>
          <Button variant="ghost" onClick={() => onVote('skip')}>
            <SkipForward size={16} aria-hidden /> Skip <Kbd>S</Kbd>
          </Button>
          <span className="text-[12px] text-faint">The two outputs share a prompt and a seed. You see which model wrote them after you vote.</span>
        </div>
      ) : (
        <div className="flex flex-wrap items-center gap-3">
          <span className="text-[13px] text-muted">{view.vote ? VOTE_WORDS[view.vote] : ''}</span>
          <Button variant="primary" onClick={onNext} disabled={!hasNext}>
            Next <ArrowRight size={16} aria-hidden /> <Kbd>N</Kbd>
          </Button>
          {!hasNext && <span className="text-[12px] text-faint">No more pairs to vote on for now.</span>}
        </div>
      )}
    </div>
  )
}
