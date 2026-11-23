import { useEffect, useState } from 'react'

/** The current time, refreshed every `ms` while the component is shown (countdowns, "so far" spans). */
export function useNow(ms = 1000): number {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), ms)
    return () => clearInterval(t)
  }, [ms])
  return now
}
