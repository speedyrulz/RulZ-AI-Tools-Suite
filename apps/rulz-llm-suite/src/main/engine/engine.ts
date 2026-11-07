import { existsSync, statSync } from 'node:fs'
import { dirname, join } from 'node:path'
import type { EngineDevice, EngineInfo } from '@shared/types'
import { describeExitCode, runCapture } from './process'

const EXE_NAME = process.platform === 'win32' ? 'llama-server.exe' : 'llama-server'
const DEFAULT_LLAMA_CPP_DIR = 'C:\\coding\\llama.cpp'

export function candidateDirs(configured: string): string[] {
  const dirs: string[] = []
  const c = configured.trim()
  if (c) dirs.push(/\.exe$/i.test(c) || (existsSync(c) && statSync(c).isFile()) ? dirname(c) : c)
  for (const build of ['build-cuda', 'build']) {
    dirs.push(join(DEFAULT_LLAMA_CPP_DIR, build, 'bin', 'Release'), join(DEFAULT_LLAMA_CPP_DIR, build, 'bin'))
  }
  return [...new Set(dirs)]
}

export function parseVersion(output: string): string | undefined {
  return output.match(/version:\s*([^\r\n]+)/i)?.[1]?.trim()
}

export function parseDevices(output: string): EngineDevice[] {
  const devices: EngineDevice[] = []
  for (const line of output.split(/\r?\n/)) {
    const m = line.match(/^\s*([A-Za-z][\w-]*\d+):\s*(.+?)\s*\((\d+)\s*MiB,\s*(\d+)\s*MiB free\)/)
    if (m) devices.push({ id: m[1], name: m[2], totalMiB: Number(m[3]), freeMiB: Number(m[4]) })
  }
  return devices
}

/** Finds and validates the user's llama-server build. */
export class EngineService {
  private cached: EngineInfo | null = null
  /** The probe running now and the configured path it is for. */
  private pending: { path: string; info: Promise<EngineInfo> } | null = null

  constructor(
    private readonly getConfiguredPath: () => string,
    /** Runs llama-server with arguments and captures its output; injectable for tests. */
    private readonly run: typeof runCapture = runCapture
  ) {}

  /**
   * The engine's status. Each probe starts CUDA on every GPU, so calls made while one is running
   * share it; only a working engine is cached, so a fixed build is picked up on the next call.
   */
  getInfo(refresh = false): Promise<EngineInfo> {
    if (this.cached && !refresh) return Promise.resolve(this.cached)
    const path = this.getConfiguredPath()
    if (this.pending?.path === path) return this.pending.info
    const info = this.probe(path).then((result) => {
      // A probe for a folder that has since been changed must not replace the newer result.
      if (this.pending?.info === info) this.pending = null
      if (path === this.getConfiguredPath()) this.cached = result.ok ? result : null
      return result
    })
    this.pending = { path, info }
    // Clear a failed probe too, so the next call tries again.
    info.catch(() => {
      if (this.pending?.info === info) this.pending = null
    })
    return info
  }

  private async probe(configured: string): Promise<EngineInfo> {
    const searched = candidateDirs(configured)
    const dir = searched.find((d) => existsSync(join(d, EXE_NAME)))
    if (!dir) {
      return {
        ok: false,
        exePath: null,
        devices: [],
        searched,
        error: `${EXE_NAME} was not found. Build llama.cpp with LLAMA_BUILD_SERVER=ON, then set the engine folder in Settings.`
      }
    }
    const exePath = join(dir, EXE_NAME)
    const [ver, dev] = await Promise.all([this.run(exePath, ['--version'], dir), this.run(exePath, ['--list-devices'], dir)])
    const version = parseVersion(ver.output)
    const ok = ver.code === 0 || !!version
    const detail = ver.output.trim().slice(-2000) || describeExitCode(ver.code) || `exit code ${ver.code}`
    return {
      ok,
      exePath,
      version,
      devices: parseDevices(dev.output),
      searched,
      error: ok ? undefined : `llama-server failed to start:\n${detail}`
    }
  }

  async resolveExe(): Promise<string> {
    const info = await this.getInfo()
    if (!info.ok || !info.exePath) throw new Error(info.error ?? 'llama-server is not available')
    return info.exePath
  }
}
