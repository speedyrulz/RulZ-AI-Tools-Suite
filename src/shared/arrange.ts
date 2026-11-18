import { CATEGORIES } from './catalog'
import type { AppView, CatalogApp } from './types'

// How the home page lays the apps out: which ones a filter and a search leave, and the sections they go in.

/** Everything, the installed apps, the ones with a newer release, or one category (`cat:<name>`). */
export type HomeFilter = 'all' | 'installed' | 'updates' | `cat:${string}`

export interface HomeSection {
  id: string
  /** Null for a plain list without a heading (search results, the Installed and Updates filters). */
  title: string | null
  blurb?: string
  views: AppView[]
}

export interface HomeCounts {
  all: number
  installed: number
  updates: number
  /** In display order; a category without apps is left out. */
  categories: { name: string; count: number }[]
}

/** The categories in display order: the catalog's list first, then any an app names that the list doesn't have. */
export function categoryOrder(apps: AppView[]): string[] {
  const used = new Set(apps.map((v) => v.app.category))
  const known = CATEGORIES.map((c) => c.name).filter((name) => used.has(name))
  return [...known, ...[...used].filter((name) => !known.includes(name))]
}

function hasUpdate(view: AppView): boolean {
  return view.updateAvailable && !!view.installed
}

export function homeCounts(apps: AppView[]): HomeCounts {
  return {
    all: apps.length,
    installed: apps.filter((v) => v.installed).length,
    updates: apps.filter(hasUpdate).length,
    categories: categoryOrder(apps).map((name) => ({ name, count: apps.filter((v) => v.app.category === name).length }))
  }
}

function searchWords(query: string): string[] {
  return query.toLowerCase().split(/\s+/).filter(Boolean)
}

/** Every word of the search must appear somewhere in the app's name, tagline, description, category or highlights. */
export function matchesQuery(app: CatalogApp, query: string): boolean {
  const words = searchWords(query)
  if (!words.length) return true
  const text = [app.name, app.tagline, app.description, app.category, ...app.highlights].join(' ').toLowerCase()
  return words.every((w) => text.includes(w))
}

function matchesFilter(view: AppView, filter: HomeFilter): boolean {
  if (filter === 'all') return true
  if (filter === 'installed') return !!view.installed
  if (filter === 'updates') return hasUpdate(view)
  return view.app.category === filter.slice(4)
}

/** A saved order read from settings.json: a list of distinct strings, or undefined when it is anything else or empty. */
export function cleanOrder(value: unknown): string[] | undefined {
  if (!Array.isArray(value) || value.some((x) => typeof x !== 'string')) return undefined
  const list = [...new Set(value as string[])]
  return list.length ? list : undefined
}

/** Sorts by the saved order. Apps that aren't in it follow, in the order they came in. */
function byOrder(views: AppView[], order: string[] | undefined): AppView[] {
  const saved = cleanOrder(order)
  if (!saved) return views
  const at = new Map(saved.map((id, i) => [id, i]))
  const rank = (v: AppView): number => at.get(v.app.id) ?? saved.length
  return views.map((v, i) => ({ v, i })).sort((a, b) => rank(a.v) - rank(b.v) || a.i - b.i).map((x) => x.v)
}

/**
 * The new full order to save after `id` is dragged to place `toIndex` among the installed apps (`shownIds` is the
 * order they are shown in now; `toIndex` counts after the app is taken out, so it is where the app ends up).
 * Saved ids that aren't shown (an app that is uninstalled for now) are kept next to the id they followed, so
 * installing it again puts it back where it was.
 */
export function moveInOrder(order: string[] | undefined, shownIds: string[], id: string, toIndex: number): string[] {
  const from = shownIds.indexOf(id)
  if (from < 0) return cleanOrder(order) ?? [...shownIds]
  const next = shownIds.filter((x) => x !== id)
  next.splice(Math.max(0, Math.min(next.length, toIndex)), 0, id)
  const shown = new Set(shownIds)
  const old = cleanOrder(order) ?? []
  const result = [...next]
  old.forEach((h, i) => {
    if (shown.has(h)) return
    // Right after the nearest earlier id of the old order that is in the result, or first.
    let after = -1
    for (let j = i - 1; j >= 0 && after < 0; j--) after = result.indexOf(old[j])
    result.splice(after + 1, 0, h)
  })
  return result
}

/** Keeps the order, with the apps that pass the test moved to the front. */
function first(views: AppView[], test: (v: AppView) => boolean): AppView[] {
  return [...views.filter(test), ...views.filter((v) => !test(v))]
}

/**
 * The sections of the home page. With no filter and no search: the installed apps, then the others by category.
 * The installed apps follow the user's saved order. A category filter lists that whole category, installed apps
 * first. A search lists its matches in one block, the apps whose name matches first.
 */
export function arrangeHome(apps: AppView[], filter: HomeFilter, query: string, installedOrder?: string[]): HomeSection[] {
  const words = searchWords(query)
  const shown = apps.filter((v) => matchesFilter(v, filter) && matchesQuery(v.app, query))
  if (!shown.length) return []

  if (words.length) {
    const inName = (v: AppView): boolean => words.every((w) => v.app.name.toLowerCase().includes(w))
    return [{ id: 'results', title: null, views: first(shown, inName) }]
  }

  if (filter.startsWith('cat:')) {
    const name = filter.slice(4)
    const blurb = CATEGORIES.find((c) => c.name === name)?.blurb
    return [{ id: filter, title: name, blurb, views: [...byOrder(shown.filter((v) => v.installed), installedOrder), ...shown.filter((v) => !v.installed)] }]
  }

  if (filter !== 'all') return [{ id: filter, title: null, views: filter === 'installed' ? byOrder(shown, installedOrder) : shown }]

  const sections: HomeSection[] = []
  const installed = byOrder(shown.filter((v) => v.installed), installedOrder)
  if (installed.length) sections.push({ id: 'installed', title: 'Installed', views: installed })
  for (const name of categoryOrder(shown)) {
    const views = shown.filter((v) => !v.installed && v.app.category === name)
    if (views.length) sections.push({ id: `cat:${name}`, title: name, blurb: CATEGORIES.find((c) => c.name === name)?.blurb, views })
  }
  return sections
}

/** The app ids in the order the unfiltered home page shows them (what Previous / Next on an app's page follow). */
export function homeOrder(apps: AppView[], installedOrder?: string[]): string[] {
  return arrangeHome(apps, 'all', '', installedOrder).flatMap((s) => s.views.map((v) => v.app.id))
}

/** The ids before and after `id` in `order`, going round at the ends. Null when there is nowhere else to go. */
export function neighbors(order: string[], id: string): { previous: string; next: string } | null {
  const at = order.indexOf(id)
  if (at < 0 || order.length < 2) return null
  return { previous: order[(at - 1 + order.length) % order.length], next: order[(at + 1) % order.length] }
}
