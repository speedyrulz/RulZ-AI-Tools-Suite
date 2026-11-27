// Types shared by the main process and the renderer. Keep this file free of Node and DOM imports.

export type Theme = 'dark' | 'light'

/**
 * Which device the app's own ComfyUI server runs on. GPUs are named by nvidia-smi's index (PCI bus order): the own
 * server is started with CUDA_DEVICE_ORDER=PCI_BUS_ID, so `--cuda-device <index>` means the same card. (Without that
 * variable CUDA uses its own "fastest first" order, which can differ from nvidia-smi's.)
 */
export type ServerDevice = { kind: 'gpu'; index: number; name: string } | { kind: 'cpu' }

/** Where jobs go: the app's own server (default) or a server someone else started. The URL is kept in both modes. */
export interface Connection {
  mode: 'own' | 'url'
  url: string
}

export interface OwnServerSettings {
  /** WSL distro that holds ComfyUI. */
  distro: string
  /** ComfyUI checkout inside WSL; `~` is expanded by bash. */
  comfyDir: string
  /** Python to run, relative to comfyDir or absolute. */
  python: string
  /** Base directory for the own server (its output, input, user and custom_nodes folders). */
  baseDir: string
  /** Models folder the own server reads; empty = `<comfyDir>/models`. */
  modelsDir: string
  /** First port to try; the next free port at or above it is used. */
  portStart: number
  device: ServerDevice
  /** Also load the node packs from `<comfyDir>/custom_nodes` (needed for models with custom loaders). */
  loadCustomNodes: boolean
  /** Extra ComfyUI command-line options, split on whitespace. */
  extraArgs: string
  /** Refuse to start below this much free WSL memory unless the user confirms. */
  minFreeRamGiB: number
  /** Seconds to wait after Ctrl+C before the process is killed. */
  stopTimeoutSec: number
}

export interface AppSettings {
  theme: Theme
  connection: Connection
  ownServer: OwnServerSettings
}

export const DEFAULT_SETTINGS: AppSettings = {
  theme: 'dark',
  connection: { mode: 'own', url: '' },
  ownServer: {
    distro: 'Ubuntu-24.04',
    comfyDir: '~/ComfyUI',
    python: 'venv/bin/python',
    baseDir: '~/.rulz-model-merger/comfy',
    modelsDir: '',
    portStart: 8288,
    device: { kind: 'cpu' },
    loadCustomNodes: false,
    extraArgs: '',
    minFreeRamGiB: 24,
    stopTimeoutSec: 20
  }
}

/** One NVIDIA card, from nvidia-smi on Windows. `index` is nvidia-smi's (PCI bus) order. */
export interface GpuInfo {
  index: number
  name: string
  busId: string
  memoryUsedMiB: number
  memoryTotalMiB: number
  utilizationPct: number
  temperatureC: number | null
}

/** A ComfyUI server found listening on this PC (from GET /system_stats). */
export interface RunningServer {
  url: string
  port: number
  version: string | null
  /** Device names the server reports, e.g. "cuda:0 NVIDIA GeForce RTX 5060 Ti : cudaMallocAsync". */
  devices: string[]
  /** The `--cuda-device` value from its argv, if any. Its meaning depends on that server's CUDA_DEVICE_ORDER. */
  cudaDevice: string | null
  /** Indexes of the GpuInfo cards whose names match the server's devices (empty for CPU or when unknown). */
  gpuIndexes: number[]
  /** True when this app started it. */
  own: boolean
}

export type PreflightCode =
  | 'gpu-in-use-by-server'
  | 'gpu-low-vram'
  | 'low-ram'
  | 'no-free-port'
  | 'wsl-missing'
  | 'comfy-missing'
  | 'python-missing'
  | 'already-running'
  | 'ports-unreadable'
  | 'managed-arg'
  | 'path-not-absolute'
  | 'gpu-unknown'

export interface PreflightIssue {
  code: PreflightCode
  /** 'block' stops the start; 'confirm' needs the user's OK; 'info' is shown only. */
  level: 'block' | 'confirm' | 'info'
  message: string
}

export interface PreflightReport {
  freeRamGiB: number | null
  totalRamGiB: number | null
  gpus: GpuInfo[]
  runningServers: RunningServer[]
  /** Port the server would use, or null when none is free. */
  port: number | null
  issues: PreflightIssue[]
}

export type ServerStatus = 'stopped' | 'starting' | 'running' | 'stopping' | 'error'

export interface ServerState {
  status: ServerStatus
  port: number | null
  url: string | null
  /** PID inside WSL. */
  pid: number | null
  device: ServerDevice | null
  startedAt: number | null
  error: string | null
}

export interface ServerLogEvent {
  /** Text as printed, may contain \r progress updates. */
  text: string
}

/** The ComfyUI that jobs are sent to right now. */
export interface ConnectionStatus {
  url: string | null
  reachable: boolean
  own: boolean
  version: string | null
  devices: string[]
  /** For a server the app did not start: the user agreed to put jobs in its queue this session. */
  sharedQueueAccepted: boolean
  error: string | null
}

/** Merge, save and loader nodes the app relies on, and whether the connected server has them. */
export const KNOWN_NODES = [
  'CheckpointLoaderSimple',
  'UNETLoader',
  'CLIPLoader',
  'DualCLIPLoader',
  'VAELoader',
  'LoraLoader',
  'LoraLoaderModelOnly',
  'ModelMergeSimple',
  'ModelMergeBlocks',
  'ModelMergeAdd',
  'ModelMergeSubtract',
  'CLIPMergeSimple',
  'CheckpointSave',
  'ModelSave',
  'CLIPSave',
  'VAESave',
  'LoraSave',
  'KSampler',
  'SaveImage'
] as const

export interface ServerCapabilities {
  checkpoints: string[]
  diffusionModels: string[]
  loras: string[]
  textEncoders: string[]
  vaes: string[]
  /** Presence of each KNOWN_NODES entry. */
  nodes: Record<string, boolean>
  /** Architecture-specific block merge nodes found (names starting with "ModelMerge" not in KNOWN_NODES). */
  blockMergeNodes: string[]
}
