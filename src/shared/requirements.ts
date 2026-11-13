import type { RequirementId, RequirementRef, RequirementStatus } from './types'

/** True when version `a` ("3.10.11") is older than `b` ("3.11"). */
function olderThan(a: string, b: string): boolean {
  const x = a.split('.').map(Number)
  const y = b.split('.').map(Number)
  for (let i = 0; i < Math.max(x.length, y.length); i++) {
    const d = (x[i] ?? 0) - (y[i] ?? 0)
    if (d) return d < 0
  }
  return false
}

/** The name of an app's requirement, with the version it needs when that's more than usual ("Python 3.11+"). */
export function requirementRefLabel(ref: RequirementRef): string {
  return ref.minVersion && ref.id === 'python' ? `Python ${ref.minVersion}+` : REQUIREMENT_GUIDES[ref.id].label
}

/** An app's requirement `id`, with the version it asks for (just the id when the app doesn't list it). */
export function refFor(refs: RequirementRef[] | undefined, id: RequirementId): RequirementRef {
  return refs?.find((r) => r.id === id) ?? { id }
}

/**
 * One app's view of a requirement's status: a Python that passes the general check but is older than the app needs
 * counts as missing. winget's Python goes first on the user's PATH, which is enough unless the old one was installed
 * for all users; then the check after the install still finds the old one and the guide takes over.
 */
export function statusFor(ref: RequirementRef, status: RequirementStatus | undefined): RequirementStatus | undefined {
  if (!status || !ref.minVersion) return status
  const label = requirementRefLabel(ref)
  if (!status.ok || !status.version || !olderThan(status.version, ref.minVersion)) return { ...status, label }
  const found = status.version.split('.').slice(0, 2).join('.')
  return { ...status, label, ok: false, detail: `Python ${found} is too old for this app` }
}

// What each prerequisite is and how to install it by hand, for when winget can't do it.

export interface RequirementGuide {
  label: string
  helpUrl: string
  /** Button text for helpUrl. */
  linkLabel: string
  steps: string[]
}

export const REQUIREMENT_GUIDES: Record<RequirementId, RequirementGuide> = {
  python: {
    label: 'Python 3.10+',
    helpUrl: 'https://www.python.org/downloads/windows/',
    linkLabel: 'Open python.org downloads',
    steps: [
      'Open the python.org download page and download the latest Python 3.12 "Windows installer (64-bit)".',
      'Run it. On the first screen, tick "Add python.exe to PATH" at the bottom. This step matters: without it the apps can\'t find Python.',
      'Click "Install Now" and wait for "Setup was successful", then close the installer.',
      'Come back here and click "Check again".'
    ]
  },
  ffmpeg: {
    label: 'FFmpeg',
    helpUrl: 'https://www.gyan.dev/ffmpeg/builds/',
    linkLabel: 'Open the FFmpeg builds page',
    steps: [
      'Open the FFmpeg builds page and download "ffmpeg-release-essentials.zip".',
      'Unzip it to a permanent place, for example C:\\ffmpeg (so the programs end up in C:\\ffmpeg\\bin).',
      'Press Start, type "environment variables" and open "Edit environment variables for your account".',
      'Select "Path", click Edit, then New, and enter the bin folder (C:\\ffmpeg\\bin). Click OK twice.',
      'Come back here and click "Check again".'
    ]
  },
  git: {
    label: 'Git',
    helpUrl: 'https://git-scm.com/download/win',
    linkLabel: 'Open the Git download page',
    steps: [
      'Open the Git download page; the 64-bit installer downloads automatically.',
      'Run it and keep the default choices on every page (they add Git to PATH).',
      'Come back here and click "Check again".'
    ]
  },
  nvidia: {
    label: 'NVIDIA GPU',
    helpUrl: 'https://www.nvidia.com/en-us/drivers/',
    linkLabel: 'Open NVIDIA driver downloads',
    steps: [
      'This needs an NVIDIA graphics card from the GTX 16 or RTX 20 series or newer, with NVIDIA driver 580 or newer (the AI apps use CUDA 13). Older cards such as the GTX 10 series can\'t run them.',
      'Open NVIDIA\'s driver page, pick your card (or use the NVIDIA App to find it) and download the Game Ready or Studio driver.',
      'Run the installer and restart Windows if it asks.',
      'Come back here and click "Check again".'
    ]
  },
  'dotnet-fx': {
    label: '.NET Framework 4.8',
    helpUrl: 'https://dotnet.microsoft.com/download/dotnet-framework/net48',
    linkLabel: 'Open the .NET Framework 4.8 download',
    steps: [
      'Windows 10 and 11 include it, so it is normally already there. If it\'s reported missing, run Windows Update first.',
      'Otherwise download the ".NET Framework 4.8 Runtime" from Microsoft and run it.',
      'Restart Windows if it asks, then click "Check again".'
    ]
  }
}
