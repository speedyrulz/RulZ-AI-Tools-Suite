import type { RequirementId } from './types'

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
