import { existsSync, statSync } from 'node:fs'
import { dirname, join } from 'node:path'
import type { EngineDevice, EngineInfo } from '@shared/types'
import { runCapture } from './process'

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

  constructor(private readonly getConfiguredPath: () => string) {}

  async getInfo(refresh = false): Promise<EngineInfo> {
    if (this.cached && !refresh) return this.cached
    const searched = candidateDirs(this.getConfiguredPath())
    const dir = searched.find((d) => existsSync(join(d, EXE_NAME)))
    if (!dir) {
      this.cached = {
        ok: false,
        exePath: null,
        devices: [],
        searched,
        error: `${EXE_NAME} was not found. Build llama.cpp with LLAMA_BUILD_SERVER=ON, then set the engine folder in Settings.`
      }
      return this.cached
    }
    const exePath = join(dir, EXE_NAME)
    const [ver, dev] = await Promise.all([
      runCapture(exePath, ['--version'], dir),
      runCapture(exePath, ['--list-devices'], dir)
    ])
    const version = parseVersion(ver.output)
    this.cached = {
      ok: ver.code === 0 || !!version,
      exePath,
      version,
      devices: parseDevices(dev.output),
      searched,
      error: ver.code === 0 || version ? undefined : `llama-server failed to start:\n${ver.output.trim().slice(-2000)}`
    }
    return this.cached
  }

  async resolveExe(): Promise<string> {
    const info = await this.getInfo()
    if (!info.ok || !info.exePath) throw new Error(info.error ?? 'llama-server is not available')
    return info.exePath
  }
}
