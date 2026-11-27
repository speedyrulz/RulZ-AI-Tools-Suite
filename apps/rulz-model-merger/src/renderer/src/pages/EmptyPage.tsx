import { PageTitle } from '../components/ui'

export function EmptyPage({ title, text }: { title: string; text: string }) {
  return (
    <div>
      <PageTitle title={title} />
      <div className="rounded-xl border border-dashed border-border-strong bg-panel px-6 py-10 text-center">
        <p className="text-sm text-muted">{text}</p>
      </div>
    </div>
  )
}
