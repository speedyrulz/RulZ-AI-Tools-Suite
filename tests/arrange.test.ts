import { describe, expect, it } from 'vitest'
import { arrangeHome, categoryOrder, homeCounts, homeOrder, matchesQuery, neighbors } from '@shared/arrange'
import { CATALOG, CATEGORIES } from '@shared/catalog'
import type { AppView, InstalledInfo } from '@shared/types'

const INSTALLED: InstalledInfo = {
  dir: 'C:\\Apps\\x',
  launchPath: 'C:\\Apps\\x\\x.exe',
  launchArgs: [],
  version: 'v1.0.0',
  method: 'zip',
  managed: true,
  installedAt: '2026-11-01T00:00:00Z',
  shortcuts: { desktop: null, startMenu: null },
  components: [],
  needsSetup: false
}

/** The real catalog, with the named apps installed (and some of them behind their latest release). */
function views(installed: string[] = [], updates: string[] = []): AppView[] {
  return CATALOG.map((app) => ({
    app,
    icon: '',
    screenshots: [],
    installed: installed.includes(app.id) ? INSTALLED : null,
    missing: false,
    latest: null,
    latestError: null,
    updateAvailable: updates.includes(app.id),
    running: false,
    job: null
  }))
}

const ids = (list: AppView[]): string[] => list.map((v) => v.app.id)

describe('categories', () => {
  it('every app is in a category the home page knows, each with a description', () => {
    const names = CATEGORIES.map((c) => c.name)
    expect(new Set(names).size).toBe(names.length)
    for (const app of CATALOG) expect(names, app.id).toContain(app.category)
    for (const c of CATEGORIES) expect(c.blurb).toBeTruthy()
  })

  it('lists them in the catalog order, an unknown one last, an empty one not at all', () => {
    const all = views()
    expect(categoryOrder(all)).toEqual(CATEGORIES.map((c) => c.name))
    const odd = [{ ...all[0], app: { ...all[0].app, category: 'Audio' } }, ...all.filter((v) => v.app.category === 'LLM')]
    expect(categoryOrder(odd)).toEqual(['LLM', 'Audio'])
  })

  it('counts the apps per chip', () => {
    const c = homeCounts(views(['llm-suite', 'image-grid'], ['image-grid', 'rulz-trainer']))
    expect(c.all).toBe(CATALOG.length)
    expect(c.installed).toBe(2)
    // An update only counts for an app that is installed.
    expect(c.updates).toBe(1)
    expect(c.categories.reduce((n, x) => n + x.count, 0)).toBe(CATALOG.length)
    expect(c.categories.find((x) => x.name === 'Video')?.count).toBe(1)
  })
})

describe('arrangeHome', () => {
  it('with nothing installed, shows one section per category and every app once', () => {
    const sections = arrangeHome(views(), 'all', '')
    expect(sections.map((s) => s.title)).toEqual(CATEGORIES.map((c) => c.name))
    expect(sections.every((s) => !!s.blurb)).toBe(true)
    expect(sections.flatMap((s) => ids(s.views)).sort()).toEqual(CATALOG.map((a) => a.id).sort())
  })

  it('puts the installed apps in their own section first and leaves them out of the categories', () => {
    const sections = arrangeHome(views(['coding-agent', 'rulz-trainer']), 'all', '')
    expect(sections[0].title).toBe('Installed')
    // Catalog order inside a section.
    expect(ids(sections[0].views)).toEqual(['rulz-trainer', 'coding-agent'])
    const rest = sections.slice(1).flatMap((s) => ids(s.views))
    expect(rest).not.toContain('coding-agent')
    expect(rest).not.toContain('rulz-trainer')
    expect(rest.length).toBe(CATALOG.length - 2)
  })

  it('leaves out a category whose apps are all installed', () => {
    const sections = arrangeHome(views(['clip-extractor']), 'all', '')
    expect(sections.map((s) => s.title)).not.toContain('Video')
    expect(arrangeHome(views(CATALOG.map((a) => a.id)), 'all', '').map((s) => s.title)).toEqual(['Installed'])
  })

  it('a category filter lists the whole category, installed apps first', () => {
    const [section, ...more] = arrangeHome(views(['image-grid']), 'cat:Images', '')
    expect(more).toEqual([])
    expect(section.title).toBe('Images')
    expect(ids(section.views)[0]).toBe('image-grid')
    expect(section.views.every((v) => v.app.category === 'Images')).toBe(true)
    expect(section.views.length).toBe(CATALOG.filter((a) => a.category === 'Images').length)
  })

  it('the Installed and Updates filters list their apps without a heading', () => {
    const all = views(['llm-suite', 'image-grid'], ['image-grid'])
    expect(arrangeHome(all, 'installed', '')).toMatchObject([{ title: null }])
    expect(ids(arrangeHome(all, 'installed', '')[0].views).sort()).toEqual(['image-grid', 'llm-suite'])
    expect(ids(arrangeHome(all, 'updates', '')[0].views)).toEqual(['image-grid'])
    expect(arrangeHome(views(), 'installed', '')).toEqual([])
    expect(arrangeHome(views(), 'updates', '')).toEqual([])
  })

  it('a search lists its matches in one block, apps whose name matches first', () => {
    const [section, ...more] = arrangeHome(views(['llm-suite']), 'all', 'image')
    expect(more).toEqual([])
    expect(section.title).toBeNull()
    const inName = section.views.map((v) => v.app.name.toLowerCase().includes('image'))
    // Both kinds are there (Rulz Training Captioner only has "image" in its text), and every name match comes first.
    expect(inName).toContain(true)
    expect(inName).toContain(false)
    expect(inName.lastIndexOf(true)).toBeLessThan(inName.indexOf(false))
    expect(ids(section.views)).toContain('image-grid')
    expect(ids(section.views)).toContain('training-captioner')
    expect(arrangeHome(views(), 'all', 'no such app anywhere')).toEqual([])
  })

  it('a search and a filter both apply', () => {
    const found = arrangeHome(views(), 'cat:LLM', 'image').flatMap((s) => s.views)
    expect(found.every((v) => v.app.category === 'LLM')).toBe(true)
    const all = views(['image-grid', 'llm-suite'], ['image-grid'])
    expect(ids(arrangeHome(all, 'installed', 'grid').flatMap((s) => s.views))).toEqual(['image-grid'])
    expect(arrangeHome(all, 'installed', 'tournament')).toEqual([])
    expect(ids(arrangeHome(all, 'updates', 'image').flatMap((s) => s.views))).toEqual(['image-grid'])
    expect(arrangeHome(all, 'updates', 'llm')).toEqual([])
  })

  it('a category nobody is in lists nothing', () => {
    expect(arrangeHome(views(), 'cat:Nope', '')).toEqual([])
    expect(arrangeHome(views(), 'cat:Nope', 'image')).toEqual([])
  })

  it('an update for an app that is not installed is not an update to show', () => {
    const all = views(['llm-suite'], ['llm-suite', 'rulz-trainer'])
    expect(homeCounts(all).updates).toBe(1)
    expect(ids(arrangeHome(all, 'updates', '')[0].views)).toEqual(['llm-suite'])
  })

  it('gives every section its own id, whatever is installed', () => {
    for (const installed of [[], ['llm-suite', 'image-grid'], CATALOG.map((a) => a.id)]) {
      const sectionIds = arrangeHome(views(installed), 'all', '').map((s) => s.id)
      expect(new Set(sectionIds).size).toBe(sectionIds.length)
    }
  })

  it('gives Previous / Next the order of the home page', () => {
    const order = homeOrder(views(['hf-manager']))
    expect(order[0]).toBe('hf-manager')
    expect(new Set(order).size).toBe(order.length)
    expect([...order].sort()).toEqual(CATALOG.map((a) => a.id).sort())
    // With everything installed there is one section, in catalog order.
    expect(homeOrder(views(CATALOG.map((a) => a.id)))).toEqual(CATALOG.map((a) => a.id))
  })
})

describe('neighbors', () => {
  it('steps through the list and goes round at the ends', () => {
    const order = ['a', 'b', 'c']
    expect(neighbors(order, 'b')).toEqual({ previous: 'a', next: 'c' })
    expect(neighbors(order, 'a')).toEqual({ previous: 'c', next: 'b' })
    expect(neighbors(order, 'c')).toEqual({ previous: 'b', next: 'a' })
    expect(neighbors(['a', 'b'], 'a')).toEqual({ previous: 'b', next: 'b' })
  })

  it('has nowhere to go from a single app or an unknown one', () => {
    expect(neighbors(['a'], 'a')).toBeNull()
    expect(neighbors([], 'a')).toBeNull()
    expect(neighbors(['a', 'b'], 'z')).toBeNull()
  })
})

describe('matchesQuery', () => {
  const trainer = CATALOG.find((a) => a.id === 'rulz-trainer')!

  it('needs every word, in any order and any case, anywhere in the app text', () => {
    expect(matchesQuery(trainer, '')).toBe(true)
    expect(matchesQuery(trainer, '   ')).toBe(true)
    expect(matchesQuery(trainer, 'LORA')).toBe(true)
    expect(matchesQuery(trainer, 'wizard lora')).toBe(true)
    expect(matchesQuery(trainer, 'lora spreadsheet')).toBe(false)
    // The category and the highlights count too.
    expect(matchesQuery(trainer, 'ai training')).toBe(true)
    expect(matchesQuery(trainer, 'fleet')).toBe(true)
    // Tabs and runs of spaces separate words like one space does.
    expect(matchesQuery(trainer, '  wizard \t  lora  ')).toBe(true)
    // A word that is only in the tagline.
    expect(trainer.description).not.toContain('guided')
    expect(matchesQuery(trainer, 'guided')).toBe(true)
  })
})

describe('catalog text', () => {
  it('keeps taglines short enough for two lines on a card', () => {
    for (const app of CATALOG) expect(app.tagline.length, `${app.id}: "${app.tagline}"`).toBeLessThanOrEqual(72)
  })
})
