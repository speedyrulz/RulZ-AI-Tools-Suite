// Command-safety cases found by the 2026-11-27 audit: paths outside the project written in ways the
// filter did not see (forward slashes from the root of the drive, the drive root itself, `..` after a
// value, drive-relative paths).
import { readdirSync } from 'node:fs'
import { join, parse } from 'node:path'
import { describe, expect, it } from 'vitest'
import { classifyCommand } from '../src/main/agent/safety'

const ROOT = 'C:\\coding\\My App'
const verdict = (cmd: string, root = ROOT) => classifyCommand(cmd, root)

describe('deleting outside the project, written so the old filter missed it', () => {
  it('refuses forward-slash paths from the root of the drive (B1)', () => {
    for (const cmd of [
      'Remove-Item -Recurse -Force /Users/me/Documents',
      'rm -r -fo /Windows',
      'Remove-Item -Recurse /Users/',
      'Remove-Item -Recurse "/Program Files/x"',
      "Remove-Item -Recurse '/Users/me/My Files'"
    ])
      expect(verdict(cmd), cmd).toMatchObject({ verdict: 'refuse' })
  })

  it('refuses a delete after going to the root of the drive (B1)', () => {
    for (const cmd of ['cd \\; Remove-Item -Recurse Users', 'cd /; rm -r -fo Users', 'Get-ChildItem / -Recurse | Remove-Item -Recurse', 'Set-Location -Path \\; ri -r Windows'])
      expect(verdict(cmd), cmd).toMatchObject({ verdict: 'refuse' })
  })

  it('refuses `..` after a value, which can lead anywhere (B2)', () => {
    for (const cmd of [
      'Remove-Item -Recurse -Force "$pwd\\.."',
      'Remove-Item -Recurse -Force $pwd\\..',
      'Remove-Item -Recurse "${pwd}/.."',
      'Remove-Item -Recurse -Force $(Get-Location)\\..'
    ])
      expect(verdict(cmd), cmd).toMatchObject({ verdict: 'refuse' })
  })

  it('refuses a drive-relative path on another drive (B3)', () => {
    expect(verdict('Remove-Item -Recurse C:Windows', 'D:\\work\\app')).toMatchObject({ verdict: 'refuse' })
    // On the project's own drive it goes from the project folder, where the command runs.
    expect(verdict('Remove-Item -Recurse C:build')).toMatchObject({ verdict: 'ask', kind: 'delete' })
  })

  it('refuses the drive root given with -Path, and a parent looked up at run time', () => {
    for (const cmd of [
      'Get-ChildItem -Path / -Recurse | Remove-Item -Recurse',
      'Remove-Item -Recurse -Force (Split-Path $pwd)',
      '(Get-Item .).Parent | Remove-Item -Recurse',
      'Remove-Item -Recurse ([IO.Path]::GetDirectoryName($pwd))',
      // -Leaf turned off asks for the parent again.
      'Remove-Item -Recurse (Split-Path $pwd -Leaf:$false)'
    ])
      expect(verdict(cmd), cmd).toMatchObject({ verdict: 'refuse' })
    // Only the file's own name: nothing outside.
    expect(verdict('Remove-Item (Split-Path -Leaf $f)')).toMatchObject({ verdict: 'ask', kind: 'delete' })
  })

  it('refuses a folder of its own at the root of the drive, when it is there', () => {
    const drive = parse(process.cwd()).root
    const known = /^(users|windows|programdata|perflogs|recovery|boot|program files.*|\$recycle\.bin)$/i
    const name = readdirSync(drive, { withFileTypes: true }).find(
      (d) => d.isDirectory() && d.name.length >= 2 && !known.test(d.name) && !/[\s'"$]/.test(d.name)
    )?.name
    if (!name) return
    const root = join(drive, 'no-such-project-xyz', 'app')
    expect(verdict(`Remove-Item /${name} -Recurse`, root)).toMatchObject({ verdict: 'refuse' })
    expect(verdict(`Remove-Item -Recurse /${name}/sub`, root)).toMatchObject({ verdict: 'refuse' })
  })

  it('refuses the drive root after switches that take a value, and through Get-Item', () => {
    for (const cmd of [
      'Get-ChildItem -Filter *.log -Path \\ | Remove-Item -Recurse',
      'Remove-Item -Recurse:$true \\',
      'Set-Location -ErrorAction Stop \\; Remove-Item -Recurse Windows',
      'Get-Item / | Remove-Item -Recurse',
      'gi / | ri -Recurse',
      'Remove-Item -Recurse (Get-Item .).Parent.FullName',
      'Remove-Item -Recurse (Split-Path ($pwd))',
      'Remove-Item -Recurse "$pwd\\.."',
      'cd src; Remove-Item -Recurse "$pwd\\dist\\..\\cache"',
      'Remove-Item -Recurse /Users/other',
      'Remove-Item -Recurse "/Program Files/x y"'
    ])
      expect(verdict(cmd, 'C:\\Users\\me\\app'), cmd).toMatchObject({ verdict: 'refuse' })
  })

  it('does not refuse paths that stay inside a project under C:\\Users', () => {
    for (const cmd of [
      'Remove-Item -Recurse "/Users/me/app/dist"',
      'Remove-Item -Recurse /Users/me/app/dist',
      'Remove-Item -Recurse -Force "$pwd\\dist\\..\\cache"',
      'Remove-Item (Split-Path ($f) -Leaf)',
      'Get-ChildItem -Recurse -Directory | Where-Object { $_.Name -eq "bin" -and $_.Parent.Name -eq "src" } | Remove-Item -Recurse -Force',
      'Remove-Item -Recurse dist; cp -r src out',
      'Remove-Item -Recurse dist; dir /s *.js'
    ])
      expect(verdict(cmd, 'C:\\Users\\me\\app'), cmd).toMatchObject({ verdict: 'ask', kind: 'delete' })
  })

  it('takes a drive-relative path as outside once the command changed folder', () => {
    expect(verdict('cd src; Remove-Item -Recurse C:build')).toMatchObject({ verdict: 'refuse' })
  })

  it('asks (not refuses) when such a path is only touched', () => {
    expect(verdict('Copy-Item .env /Users/Public/x.txt')).toMatchObject({ verdict: 'ask', kind: 'outside-project' })
    expect(verdict('Get-ChildItem /')).toMatchObject({ verdict: 'ask', kind: 'outside-project' })
  })
})

describe('ordinary commands stay as they were', () => {
  it('keeps Windows switches, URLs and in-project paths out of it', () => {
    const same: [string, string | null][] = [
      ['Remove-Item -Recurse src', 'delete'],
      ['cmd /c rmdir /s /q build', 'delete'],
      ['robocopy src dist /MIR', 'delete'],
      ['Remove-Item dist -Recurse; npx vite build --base=/app/', 'delete'],
      // A web route in a one-liner is not a folder at the root of the drive.
      [`Remove-Item dist -Recurse; node -e "fetch('/no-such-route-xyz/users')"`, 'delete'],
      [`Remove-Item dist -Recurse; npx vite build --base /no-such-route-xyz/sub/`, 'delete'],
      ['Remove-Item -Recurse "/coding/My App/dist"', 'delete'],
      ['taskkill /F /IM node.exe', 'stop-programs'],
      ['icacls build /grant Users:F', 'system-settings'],
      ['xcopy /E /I src dist', null],
      ['curl http://localhost:3000/api/users', null],
      ['git checkout origin/main -- src/x.ts', null],
      ['npm run build', null]
    ]
    for (const [cmd, kind] of same) {
      if (kind) expect(verdict(cmd), cmd).toMatchObject({ verdict: 'ask', kind })
      else expect(verdict(cmd), cmd).toEqual({ verdict: 'ok' })
    }
  })
})
