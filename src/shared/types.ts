// ---------- Catalog (what the launcher knows about each app) ----------

export type RequirementId = 'python' | 'ffmpeg' | 'git' | 'nvidia' | 'dotnet-fx'

/** A command run in the install folder. A .bat/.cmd runs through cmd.exe; a relative path resolves in the folder. */
export interface SetupStep {
  label: string
  run: string[]
}

/** How an app is installed. Asset patterns are regular expressions matched against release asset names. */
export type InstallSpec =
  /** A portable zip attached to the latest GitHub release. */
  | { kind: 'zip'; asset: string }
  /** An NSIS setup .exe attached to the latest GitHub release, run silently. */
  | { kind: 'installer'; asset: string; uninstallName: string; shortcutName: string }
  /** The repository itself (git clone, or the source archive without git), then the setup steps. */
  | { kind: 'source'; setup: SetupStep[] }

export interface LaunchSpec {
  /** Program to start, relative to the install folder. */
  target: string
  args?: string[]
  /** Icon for shortcuts, relative to the install folder. Defaults to the target's own icon (or the catalog icon). */
  icon?: string
}

/** An optional download added to the install folder, e.g. a llama.cpp build. */
export interface ExtraComponent {
  id: string
  label: string
  description: string
  sizeHint?: string
  default: boolean
  /** The newest release of this repository (pre-releases included) that has every asset. */
  repo: string
  assets: string[]
  /** Folder (relative to the install folder) the zips are extracted into. */
  dest: string
}

export interface RequirementRef {
  id: RequirementId
  optional?: boolean
  note?: string
}

export interface CatalogApp {
  id: string
  name: string
  tagline: string
  description: string
  category: string
  /** GitHub "owner/name", or null while the app is not published. */
  repo: string | null
  branch: string
  install: InstallSpec
  launch: LaunchSpec
  requirements: RequirementRef[]
  components?: ExtraComponent[]
  /** Folder created under the install root. */
  folderName: string
  /** Program paths of an install made outside the launcher (environment variables allowed). */
  detect?: string[]
  /** Things to know before installing (download size, first-run behaviour). */
  notes?: string[]
  highlights: string[]
}

// ---------- Runtime state ----------

export type InstallMethod = 'zip' | 'installer' | 'git' | 'archive' | 'external'

export interface InstalledInfo {
  dir: string
  /** Absolute path of the program that Launch starts. */
  launchPath: string
  launchArgs: string[]
  /** Release tag or commit SHA; null when unknown (found or located installs). */
  version: string | null
  method: InstallMethod
  /** True when the launcher installed it (and may update or remove the files). */
  managed: boolean
  installedAt: string
  shortcuts: { desktop: string | null; startMenu: string | null }
  components: string[]
  /** The files are in place but a setup step failed or was cancelled; Repair runs the steps again. */
  needsSetup: boolean
}

export interface LatestInfo {
  version: string
  label: string
  date: string | null
  downloadSize: number | null
  url: string | null
}

export type JobPhase =
  | 'resolving'
  | 'downloading'
  | 'extracting'
  | 'installing'
  | 'setup'
  | 'shortcuts'
  | 'removing'
  | 'done'
  | 'error'
  | 'cancelled'

export interface JobState {
  appId: string
  action: 'install' | 'update' | 'uninstall' | 'repair'
  phase: JobPhase
  label: string
  /** 0..1, or null when the step has no measurable progress. */
  progress: number | null
  bytes?: number
  total?: number
  /** Bytes per second while downloading. */
  speed?: number
  error?: string
  /** Set on failure when the setup log has the details. */
  hasLog?: boolean
}

export interface AppView {
  app: CatalogApp
  icon: string
  screenshots: string[]
  installed: InstalledInfo | null
  latest: LatestInfo | null
  latestError: string | null
  updateAvailable: boolean
  running: boolean
  job: JobState | null
}

export interface RequirementStatus {
  id: RequirementId
  label: string
  ok: boolean
  detail: string
  /** winget can install it. */
  canInstall: boolean
  helpUrl: string
}

export interface InstallOptions {
  dir: string
  desktopShortcut: boolean
  startMenuShortcut: boolean
  components: string[]
  launchAfter: boolean
}

export interface ShortcutOptions {
  desktop: boolean
  startMenu: boolean
}

export interface ReadmeDoc {
  markdown: string
  /** Base URL for relative image paths. */
  assetBase: string
  /** Base URL for relative links, or null when they can't be opened. */
  linkBase: string | null
  source: 'github' | 'bundled'
}

export interface Settings {
  /** Apps install into <installRoot>\<folderName> unless another folder is picked. */
  installRoot: string
  checkUpdatesOnStart: boolean
  /** Show the README from GitHub (falls back to the copy shipped with the launcher). */
  liveReadme: boolean
}
