import { describe, expect, it } from 'vitest'
import { programsInside, programsRunningFrom } from '../src/main/process'

describe('programsInside', () => {
  const dir = 'C:\\Users\\me\\AppData\\Local\\Programs\\RulZ AI Tools\\RulZ Huggingface Manager'
  const listing = [
    'C:\\Windows\\System32\\svchost.exe',
    '',
    'c:\\users\\me\\appdata\\local\\programs\\rulz ai tools\\rulz huggingface manager\\.venv\\Scripts\\python.exe',
    `${dir}\\.venv\\Scripts\\python.exe`,
    `${dir} 2\\other.exe`,
    'C:\\Users\\me\\AppData\\Local\\Programs\\RulZ AI Tools\\RulzTrainer\\venv\\Scripts\\python.exe'
  ].join('\r\n')

  it('finds programs inside the folder, ignoring case and similar folder names', () => {
    expect(programsInside(listing, dir)).toEqual(['python.exe'])
    expect(programsInside(listing, `${dir}\\`)).toEqual(['python.exe'])
  })

  it('finds nothing in an empty or unreadable listing', () => {
    expect(programsInside(null, dir)).toEqual([])
    expect(programsInside('', dir)).toEqual([])
  })
})

describe.runIf(process.platform === 'win32')('programsRunningFrom', () => {
  it('sees the test runner itself', async () => {
    const own = process.execPath
    const folder = own.slice(0, own.lastIndexOf('\\'))
    const name = own.slice(own.lastIndexOf('\\') + 1)
    expect(await programsRunningFrom(folder)).toContain(name)
  }, 30000)
})
