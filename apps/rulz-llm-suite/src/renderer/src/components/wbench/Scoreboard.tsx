import { useMemo } from 'react'
import { scores, type WritingBenchRun } from '@shared/writingBench'

/** Candidates by rating (Elo from the run's votes in vote order). */
export function Scoreboard({ run }: { run: WritingBenchRun }) {
  const rows = useMemo(
    () => [...scores(run)].sort((a, b) => b.rating - a.rating).map((s) => ({ ...s, label: run.candidates.find((c) => c.id === s.candidateId)?.label ?? 'Unknown' })),
    [run]
  )
  return (
    <div className="overflow-hidden rounded-lg border border-border">
      <table className="w-full text-[13px]">
        <thead className="bg-panel-2 text-[11px] tracking-wide text-faint uppercase">
          <tr>
            <th className="px-3 py-1.5 text-left font-medium">Candidate</th>
            <th className="w-20 px-2 py-1.5 text-right font-medium">Rating</th>
            <th className="w-16 px-2 py-1.5 text-right font-medium">Wins</th>
            <th className="w-16 px-2 py-1.5 text-right font-medium">Losses</th>
            <th className="w-16 px-2 py-1.5 text-right font-medium">Ties</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.candidateId} className="border-t border-border/60">
              <td className="max-w-0 truncate px-3 py-1.5">{r.label}</td>
              <td className="px-2 py-1.5 text-right font-medium tabular-nums">{Math.round(r.rating)}</td>
              <td className="px-2 py-1.5 text-right tabular-nums">{r.wins}</td>
              <td className="px-2 py-1.5 text-right tabular-nums">{r.losses}</td>
              <td className="px-2 py-1.5 text-right tabular-nums">{r.ties}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="border-t border-border/60 px-3 py-1.5 text-[11.5px] text-faint">
        Ratings start at 1000 and move with each vote. Skipped pairs do not count. Only your votes in this run are used.
      </p>
    </div>
  )
}
