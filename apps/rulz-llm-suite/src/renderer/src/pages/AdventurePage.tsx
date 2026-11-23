import { useEffect } from 'react'
import { AdventureList } from '../components/adventure/AdventureList'
import { PlayView } from '../components/adventure/PlayView'
import { useAdventure } from '../stores/adventure'

export function AdventurePage() {
  const adv = useAdventure((s) => s.adv)
  const opening = useAdventure((s) => s.opening)

  useEffect(() => {
    void useAdventure.getState().loadList()
  }, [])

  if (!adv) {
    if (opening) return <div className="flex h-full items-center justify-center text-sm text-faint">Opening the adventure…</div>
    return <AdventureList />
  }
  return <PlayView key={adv.id} />
}
