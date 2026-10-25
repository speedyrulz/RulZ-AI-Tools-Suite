import { describe, expect, it } from 'vitest'
import { createSlugger, prepareReadme, resolveReadmeUrl } from '@shared/readme'

describe('prepareReadme', () => {
  it('drops a centered <picture> logo and the leading title', () => {
    const md = [
      '<p align="center">',
      '  <picture>',
      '    <source media="(prefers-color-scheme: dark)" srcset="assets/logo_dark.png">',
      '    <img src="assets/logo.png" alt="Rulz" width="420">',
      '  </picture>',
      '</p>',
      '',
      '# RulzTrainer',
      '',
      'A simple trainer.'
    ].join('\r\n')
    expect(prepareReadme(md)).toBe('A simple trainer.')
  })

  it('drops a markdown logo image above the title', () => {
    const md = '![RulZ](simplecut/assets/logo_light.png)\n\n# Clip Extractor\n\nTrims video.\n\n![shot](docs/a.png)\n'
    expect(prepareReadme(md)).toBe('Trims video.\n\n![shot](docs/a.png)\n')
  })

  it('keeps pictures that are not the logo and titles further down', () => {
    const md = 'Intro\n\n<picture><img src="x.png" alt="diagram"></picture>\n\n# Later\n'
    expect(prepareReadme(md)).toBe(md)
  })
})

describe('resolveReadmeUrl', () => {
  it('resolves relative paths against the base', () => {
    expect(resolveReadmeUrl('docs/screenshot.png', 'https://raw.githubusercontent.com/o/r/main/')).toBe(
      'https://raw.githubusercontent.com/o/r/main/docs/screenshot.png'
    )
    expect(resolveReadmeUrl('./assets/logo.png', 'rulz://catalog/app/repo/')).toBe('rulz://catalog/app/repo/assets/logo.png')
  })

  it('leaves absolute URLs and anchors alone', () => {
    expect(resolveReadmeUrl('https://x.dev/a.png', 'rulz://catalog/app/repo/')).toBe('https://x.dev/a.png')
    expect(resolveReadmeUrl('#running-on-vastai', 'rulz://catalog/app/repo/')).toBe('#running-on-vastai')
  })
})

describe('createSlugger', () => {
  it('matches GitHub heading anchors', () => {
    const slug = createSlugger()
    expect(slug('FIBO 2.0 (Bria): training on dev, using the LoRA on Edit Turbo')).toBe(
      'fibo-20-bria-training-on-dev-using-the-lora-on-edit-turbo'
    )
    expect(slug('Running on vast.ai')).toBe('running-on-vastai')
    expect(slug('Using two GPUs')).toBe('using-two-gpus')
    expect(slug('Using two GPUs')).toBe('using-two-gpus-1')
  })
})
