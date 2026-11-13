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
  /** Switches on the app's page that change how the Launch button starts it. Shortcuts don't use them. */
  options?: LaunchOption[]
}

/** A launch switch: while it's on, the Launch button starts the app with these environment variables. */
export interface LaunchOption {
  id: string
  label: string
  hint?: string
  env: Record<string, string>
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
  /** The oldest version the app works with ("3.11"), when it needs more than the check itself asks for. Python only. */
  minVersion?: string
}

export interface CatalogApp {
  id: string
  name: string
  tagline: string
  description: string
  category: string
  /** GitHub "owner/name" (the suite repository), or null while the app is not published. */
  repo: string | null
  /** The app's folder in the repository, e.g. apps/RulzTrainer. */
  path: string
  /** Its release tags are this plus the version, e.g. image-organizer-v1.2.0. */
  tagPrefix: string
  /** The branch a source install clones: its folder split out to the branch root (app/<id>). Otherwise main. */
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
  /** How to show the version when it isn't the tag itself, e.g. "v1.0.0 + 3 commits" for a commit SHA. */
  versionLabel?: string
  method: InstallMethod
  /** True when the launcher installed it (and may update or remove the files). */
  managed: boolean
  installedAt: string
  shortcuts: { desktop: string | null; startMenu: string | null }
  components: string[]
  /** The files are in place but a setup step failed or was cancelled; Repair runs the steps again. */
  needsSetup: boolean
  /**
   * While needsSetup: the shortcuts and optional parts the install was started with, which come after the setup.
   * Repair adds them once the setup has run.
   */
  pending?: { desktop: boolean; startMenu: boolean; components: string[] }
}

export interface LatestInfo {
  version: string
  label: string
  date: string | null
  downloadSize: number | null
  url: string | null
}

export type JobPhase =
  | 'prerequisites'
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
  /** Required prerequisites that are still missing (the install stopped; they need installing by hand). */
  missing?: RequirementId[]
}

export interface AppView {
  app: CatalogApp
  icon: string
  screenshots: string[]
  installed: InstalledInfo | null
  /** Installed, but its program file isn't there (moved, deleted, or on a drive that isn't connected). */
  missing: boolean
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
  /** The version found ("3.12.4"), where an app may need a newer one (Python). */
  version?: string
  /** winget can install it (whether or not it is missing: an app may need a newer Python than the one found). */
  canInstall: boolean
  helpUrl: string
}

export interface InstallOptions {
  dir: string
  desktopShortcut: boolean
  startMenuShortcut: boolean
  components: string[]
  launchAfter: boolean
  /** Install missing required prerequisites (Python, FFmpeg…) with winget first. */
  installPrerequisites: boolean
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

/** A newer release of the launcher itself. */
export interface SelfUpdate {
  version: string
  tag: string
  size: number
  url: string
  notesUrl: string
  /** The release's SHA256SUMS asset, when it has one (the download is checked against it). */
  sumsUrl?: string | null
  /** The setup's file name, as listed in SHA256SUMS. */
  assetName?: string
}

/** The newest RulZ Coding Agent for Android release with an APK, for the launcher's download QR code. */
export interface PhoneApp {
  /** "v0.5.0" */
  version: string
  tag: string
  /** Direct download link of the APK (what the QR code holds). */
  url: string
  size: number
  date: string | null
  releaseUrl: string
}

export type ThemeName = 'dark' | 'light'

export interface Settings {
  /** Unset until the user picks one on first launch (the UI is dark until then). */
  theme?: ThemeName
  /** Apps install into <installRoot>\<folderName> unless another folder is picked. */
  installRoot: string
  checkUpdatesOnStart: boolean
  /** Show the README from GitHub (falls back to the copy shipped with the launcher). */
  liveReadme: boolean
  /** Quit the launcher after it starts an app. */
  closeOnLaunch: boolean
  /** An update the user chose "Skip" for. */
  skippedUpdate?: string
  /** The launch options turned on, per app id (LaunchSpec.options). */
  launchOptions?: Record<string, string[]>
}
