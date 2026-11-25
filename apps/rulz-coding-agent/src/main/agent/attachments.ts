import { randomBytes } from 'node:crypto'
import { promises as fs } from 'node:fs'
import { basename, join, win32 } from 'node:path'
import { MAX_PDF_ATTACH_BYTES, type PdfAttachment } from '@shared/pdf'
import { projectKey } from '../project/store'
import type { PdfReader } from '../pdf/reader'
import { localFile, pdfInfo } from '../pdf/tool'

const CHUNK = 524288
const ID = /^[0-9a-f]{16}$/
const TIMEOUT = 120000
type Timer = ReturnType<typeof setTimeout>
export interface AttachmentOptions {
  now?: () => number
  setTimeout?: (fn: () => void, ms: number) => Timer
  clearTimeout?: (timer: Timer) => void
}
interface Upload {
  id: string; project: string; name: string; size: number; owner: unknown
  received: number; index: number; temp: string; timer?: Timer; touched: number; busy: boolean
}

function safeName(input: string): string {
  let name = basename(win32.basename(String(input))).replace(/[\x00-\x1f\x7f<>:"/\\|?*]/g, '_').trim().replace(/[. ]+$/, '')
  name = name.replace(/\.pdf$/i, '').slice(0, 116).replace(/[. ]+$/, '')
  if (!name || /^(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?$/i.test(name)) name = 'attachment'
  return `${name}.pdf`
}

export class AttachmentStore {
  private uploads = new Map<string, Upload>()
  private now: () => number
  private schedule: (fn: () => void, ms: number) => Timer
  private unschedule: (timer: Timer) => void
  constructor(private dir: string, private reader: PdfReader, options: AttachmentOptions = {}) {
    this.now = options.now ?? Date.now
    this.schedule = options.setTimeout ?? setTimeout
    this.unschedule = options.clearTimeout ?? clearTimeout
  }
  private folder(project: string): string { return join(this.dir, 'attachments', projectKey(project)) }
  private async discard(u: Upload): Promise<void> {
    if (this.uploads.get(u.id) !== u) return
    this.uploads.delete(u.id)
    if (u.timer) this.unschedule(u.timer)
    await fs.rm(u.temp, { force: true })
  }
  private arm(u: Upload): void {
    if (u.timer) this.unschedule(u.timer)
    u.touched = this.now()
    u.timer = this.schedule(() => { void this.discard(u) }, TIMEOUT)
  }
  private get(id: string, owner: unknown): Upload {
    const u = this.uploads.get(id)
    if (!u || u.owner !== owner) throw new Error('Upload not found.')
    if (this.now() - u.touched >= TIMEOUT) { void this.discard(u); throw new Error('Upload expired.') }
    if (u.busy) throw new Error('Upload is busy.')
    return u
  }
  async begin(projectPath: string, name: string, bytes: number, owner: unknown): Promise<{ uploadId: string; chunkBytes: number }> {
    if (!Number.isSafeInteger(bytes) || bytes < 1 || bytes > MAX_PDF_ATTACH_BYTES) throw new Error('PDF must be between 1 byte and 50 MB.')
    // A new upload takes the place of the same owner's earlier one (cancelled on the phone, which just stops sending).
    await this.cancelOwner(owner)
    if (this.uploads.size >= 2) throw new Error('Too many uploads are in progress.')
    const id = randomBytes(8).toString('hex')
    const folder = this.folder(projectPath)
    const u: Upload = { id, project: projectKey(projectPath), name: safeName(name), size: bytes, owner, received: 0, index: 0, temp: join(folder, `.${id}.part`), touched: this.now(), busy: true }
    this.uploads.set(id, u)
    try {
      await fs.mkdir(folder, { recursive: true })
      await fs.writeFile(u.temp, '', { flag: 'wx' })
      if (this.uploads.get(id) !== u) { await fs.rm(u.temp, { force: true }); throw new Error('Upload cancelled.') }
      u.busy = false
      this.arm(u)
      return { uploadId: id, chunkBytes: CHUNK }
    } catch (e) { await this.discard(u); throw e }
  }
  async chunk(id: string, index: number, base64: string, owner: unknown): Promise<void> {
    const u = this.get(id, owner); u.busy = true
    try {
      if (index !== u.index || typeof base64 !== 'string' || base64.length > Math.ceil(CHUNK / 3) * 4 || !/^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/.test(base64)) throw new Error('Invalid upload chunk.')
      const data = Buffer.from(base64, 'base64')
      if (!data.length || data.length > CHUNK || data.toString('base64') !== base64 || u.received + data.length > u.size) throw new Error('Invalid upload chunk size.')
      await fs.appendFile(u.temp, data)
      if (this.uploads.get(id) !== u) { await fs.rm(u.temp, { force: true }); throw new Error('Upload cancelled.') }
      u.received += data.length; u.index++; u.busy = false; this.arm(u)
    } catch (e) { await this.discard(u); throw e }
  }
  async end(id: string, owner: unknown): Promise<PdfAttachment> {
    const u = this.get(id, owner); u.busy = true
    // Everything has arrived: counting the pages may wait its turn behind other reads, so the timeout no longer applies.
    if (u.timer) this.unschedule(u.timer)
    let target = ''
    try {
      if (u.received !== u.size) throw new Error('Upload is incomplete.')
      const data = await fs.readFile(u.temp)
      const info = await pdfInfo(data, this.reader)
      if (this.uploads.get(id) !== u) throw new Error('Upload cancelled.')
      target = join(this.folderForKey(u.project), u.id)
      await fs.mkdir(target)
      await fs.rename(u.temp, join(target, u.name))
      const item: PdfAttachment = { id: u.id, name: u.name, pages: info.pages, bytes: u.size }
      await fs.writeFile(join(target, 'attachment.json'), JSON.stringify(item), { flag: 'wx' })
      if (this.uploads.get(id) !== u) throw new Error('Upload cancelled.')
      await this.discard(u)
      return item
    } catch (e) { if (target) await fs.rm(target, { recursive: true, force: true }); await this.discard(u); throw e }
  }
  private folderForKey(key: string): string { return join(this.dir, 'attachments', key) }
  async cancel(id: string, owner: unknown): Promise<void> { await this.discard(this.get(id, owner)) }
  async dropProject(projectPath: string): Promise<void> {
    await Promise.all([...this.uploads.values()].filter((u) => u.project === projectKey(projectPath)).map((u) => this.discard(u)))
  }
  async attachPath(projectPath: string, path: string): Promise<PdfAttachment> {
    const real = await localFile(path)
    const st = await fs.stat(real)
    if (!st.isFile() || st.size < 1 || st.size > MAX_PDF_ATTACH_BYTES) throw new Error('Choose a regular PDF file of up to 50 MB.')
    const owner = Symbol('attachPath')
    const { uploadId } = await this.begin(projectPath, path, st.size, owner)
    try {
      const data = await fs.readFile(real)
      for (let i = 0, index = 0; i < data.length; i += CHUNK, index++) await this.chunk(uploadId, index, data.subarray(i, i + CHUNK).toString('base64'), owner)
      return await this.end(uploadId, owner)
    } catch (e) { const u = this.uploads.get(uploadId); if (u) await this.discard(u); throw e }
  }
  private async metadata(project: string, id: string): Promise<PdfAttachment | null> {
    if (!ID.test(id)) return null
    try {
      const folder = join(this.folder(project), id)
      const stat = await fs.lstat(folder)
      if (!stat.isDirectory() || stat.isSymbolicLink()) return null
      const meta = join(folder, 'attachment.json')
      const ms = await fs.lstat(meta)
      if (!ms.isFile() || ms.isSymbolicLink() || ms.size > 4096) return null
      const item = JSON.parse(await fs.readFile(meta, 'utf8')) as PdfAttachment
      if (item.id !== id || typeof item.name !== 'string' || safeName(item.name) !== item.name || !Number.isSafeInteger(item.pages) || item.pages < 1 || !Number.isSafeInteger(item.bytes) || item.bytes < 1 || item.bytes > MAX_PDF_ATTACH_BYTES) return null
      const file = await fs.lstat(join(folder, item.name))
      if (!file.isFile() || file.isSymbolicLink() || file.size !== item.bytes) return null
      return item
    } catch { return null }
  }
  async resolve(projectPath: string, id: string): Promise<{ path: string; name: string; pages: number } | null> {
    const item = await this.metadata(projectPath, id)
    return item ? { path: join(this.folder(projectPath), id, item.name), name: item.name, pages: item.pages } : null
  }
  async list(projectPath: string): Promise<PdfAttachment[]> {
    const names = await fs.readdir(this.folder(projectPath)).catch(() => [] as string[])
    const items = await Promise.all(names.filter((id) => ID.test(id)).map((id) => this.metadata(projectPath, id)))
    return items.filter((item): item is PdfAttachment => item !== null)
  }
  /** Stops every upload of one owner (a phone that disconnected), removing its partial files. */
  async cancelOwner(owner: unknown): Promise<void> {
    await Promise.all([...this.uploads.values()].filter((u) => u.owner === owner).map((u) => this.discard(u)))
  }
  /** Stops every upload (the app is closing). */
  async cancelAll(): Promise<void> {
    await Promise.all([...this.uploads.values()].map((u) => this.discard(u)))
  }
  /**
   * Removes what no message uses: attachments whose id is not in `referenced` and that are older than `olderThanMs`,
   * and ".<id>.part" files that do not belong to a running upload. Only ever touches names of that exact shape
   * inside this project's folder. Returns the ids removed.
   */
  async sweep(projectPath: string, referenced: ReadonlySet<string>, olderThanMs: number): Promise<string[]> {
    const folder = this.folder(projectPath)
    const names = await fs.readdir(folder).catch(() => [] as string[])
    const running = new Set([...this.uploads.values()].map((u) => u.temp))
    const removed: string[] = []
    for (const name of names) {
      const path = join(folder, name)
      try {
        if (/^\.[0-9a-f]{16}\.part$/.test(name)) {
          if (!running.has(path)) await fs.rm(path, { force: true })
        } else if (ID.test(name) && !referenced.has(name)) {
          const st = await fs.lstat(path)
          if (this.now() - st.mtimeMs >= olderThanMs) {
            await fs.rm(path, { recursive: true, force: true })
            removed.push(name)
          }
        }
      } catch {
        /* left for the next sweep */
      }
    }
    return removed
  }
  async remove(projectPath: string, id: string): Promise<void> {
    if (!ID.test(id)) return
    await fs.rm(join(this.folder(projectPath), id), { recursive: true, force: true })
  }
}
