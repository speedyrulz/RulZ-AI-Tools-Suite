export function ComingSoonPage({ title, what }: { title: string; what: string }) {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-2 px-8 text-center">
      <div className="text-lg font-semibold">{title}</div>
      <div className="max-w-md text-sm text-muted">{what}</div>
      <div className="mt-2 rounded-full border border-border px-3 py-1 text-xs text-faint">Planned for a later version</div>
    </div>
  )
}
