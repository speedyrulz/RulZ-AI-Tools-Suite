// Puts the parts of src/main/cloud together for index.ts (docs/cloud-design.md). Files in userData: cloud.json (options and
// the user's models), cloud-spend.json (the ledger), cloud-secrets.json (the key, encrypted), cloud-catalog.json (the cache).
import { join } from 'node:path'
import {
  CLOUD_ID_PREFIX,
  cloudModelId,
  isCloudModelId,
  type CloudCatalog,
  type CloudKeyInfo,
  type CloudModelDef,
  type CloudState,
  type CloudModelPatch,
  type CloudOptions
} from '@shared/cloud'
import type { ModelEntry } from '@shared/types'
import { CatalogStore, OPENROUTER_BASE } from './catalog'
import type { CloudCaller, CloudFetchInit, CloudService, CloudServiceDeps } from './contract'
import { CloudGatewayCore, abortError } from './gateway'
import { CloudKeyStore } from './keyStore'
import { CloudLedger } from './ledger'
import { CloudModelStore, toModelEntry } from './models'
import { toMicro } from './prices'
import { ReasoningCache } from './reasoningCache'

export const defaultSleep = (ms: number, signal?: AbortSignal): Promise<void> =>
  new Promise<void>((resolve, reject) => {
    if (signal?.aborted) return reject(abortError())
    const onAbort = (): void => {
      clearTimeout(timer)
      reject(abortError())
    }
    const timer = setTimeout(() => {
      signal?.removeEventListener('abort', onAbort)
      resolve()
    }, ms)
    timer.unref?.()
    signal?.addEventListener('abort', onAbort, { once: true })
  })

const PUSH_THROTTLE_MS = 250

export type CloudServiceHandle = CloudService & {
  /** Resolves when background settlements (generation lookups) have finished. */
  settled(): Promise<void>
}

export function createCloudService(deps: CloudServiceDeps): CloudServiceHandle {
  const doFetch = deps.fetch ?? ((input, init) => globalThis.fetch(input, init))
  const now = deps.now ?? Date.now
  const sleep = deps.sleep ?? defaultSleep

  const keys = new CloudKeyStore({ path: join(deps.userData, 'cloud-secrets.json'), safeStorage: deps.safeStorage })
  const scrubText = (s: string): string => keys.scrub(s)
  const store = new CloudModelStore({ path: join(deps.userData, 'cloud.json'), now })
  const catalog = new CatalogStore({ path: join(deps.userData, 'cloud-catalog.json'), fetch: doFetch, now, scrub: scrubText })
  const ledger = new CloudLedger({
    path: join(deps.userData, 'cloud-spend.json'),
    now,
    sleep,
    limitMicro: () => {
      const l = store.options().dailyLimitUsd
      return l === null ? null : toMicro(l)
    }
  })
  const reasoning = new ReasoningCache()
  const gateway = new CloudGatewayCore({
    fetch: doFetch,
    now,
    sleep,
    key: () => keys.get(),
    model: (id) => (isCloudModelId(id) ? store.get(id.slice(CLOUD_ID_PREFIX.length)) : undefined),
    options: () => store.options(),
    ledger,
    reasoning,
    scrub: scrubText
  })

  let rev = 0
  let keyInfo: CloudKeyInfo | null = null
  let lastError: string | null = null
  let timer: ReturnType<typeof setTimeout> | undefined

  const errText = (err: unknown): string => scrubText(err instanceof Error ? err.message : String(err))

  const snapshot = (): CloudState => ({
    rev,
    keySaved: keys.get() !== null,
    key: keyInfo,
    error: lastError,
    options: store.options(),
    models: store.list(),
    spend: ledger.snapshot(),
    spendError: ledger.spendError ? scrubText(ledger.spendError) : null
  })

  const push = (): CloudState => {
    if (timer) clearTimeout(timer)
    timer = undefined
    rev++
    const s = snapshot()
    deps.onChange(s)
    return s
  }
  // Spending changes with every request: pushed at most every PUSH_THROTTLE_MS.
  const schedulePush = (): void => {
    if (timer) return
    timer = setTimeout(() => {
      timer = undefined
      push()
    }, PUSH_THROTTLE_MS)
    timer.unref?.()
  }
  ledger.onChange(schedulePush)

  const defOf = (modelId: string): CloudModelDef | undefined => (isCloudModelId(modelId) ? store.get(modelId.slice(CLOUD_ID_PREFIX.length)) : undefined)

  /** GET /key with this key; throws a sentence when OpenRouter does not accept it. */
  async function readKey(key: string): Promise<CloudKeyInfo> {
    keys.remember(key)
    let res: Response
    try {
      res = await doFetch(`${OPENROUTER_BASE}/key`, { headers: { Authorization: `Bearer ${key}`, 'X-OpenRouter-Title': 'RulZ LLM Suite' }, redirect: 'error', signal: AbortSignal.timeout(15_000) })
    } catch (err) {
      throw new Error(`Could not reach OpenRouter to check the key: ${errText(err)}`)
    }
    if (res.status === 401 || res.status === 403) throw new Error('OpenRouter did not accept this key. Check that it was copied whole and has not been deleted.')
    if (!res.ok) throw new Error(`OpenRouter answered ${res.status} when checking the key.`)
    const d = ((await res.json().catch(() => ({}))) as { data?: Record<string, unknown> }).data ?? {}
    const n = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null)
    return {
      label: typeof d.label === 'string' ? d.label : '',
      limitRemaining: n(d.limit_remaining),
      usageToday: n(d.usage_daily) ?? 0,
      freeTier: d.is_free_tier === true,
      freeRequestsLeft: n(d.free_requests_left)
    }
  }

  // The window is trusted, but a wrong type must fail here, before anything is read or sent.
  const text = (v: unknown, what: string): string => {
    if (typeof v !== 'string' || v.length > 2000) throw new Error(`The ${what} must be a text.`)
    return v
  }
  const slugOf = (v: unknown): string => {
    const s = text(v, 'model').trim()
    if (!s) throw new Error('No model was given.')
    return s
  }
  const patchOf = <T extends object>(v: T): T => {
    if (!v || typeof v !== 'object' || Array.isArray(v)) throw new Error('The change must be an object.')
    return v
  }

  async function ejectAll(): Promise<void> {
    for (const d of store.list()) await deps.eject(cloudModelId(d.slug)).catch(() => undefined)
  }

  const handlers: CloudService['handlers'] = {
    'cloud:state': async () => snapshot(),
    'cloud:setKey': async (key) => {
      const k = text(key, 'key').trim()
      try {
        if (!k) {
          await keys.clear()
          await ejectAll()
          keyInfo = null
          lastError = null
          return push()
        }
        // Checked first: a key OpenRouter refuses is not saved.
        const info = await readKey(k)
        await keys.save(k)
        keyInfo = info
        lastError = null
        return push()
      } catch (err) {
        throw new Error(errText(err))
      }
    },
    'cloud:checkKey': async () => {
      const k = keys.get()
      if (!k) {
        keyInfo = null
        return push()
      }
      try {
        keyInfo = await readKey(k)
        lastError = null
      } catch (err) {
        lastError = errText(err)
      }
      return push()
    },
    'cloud:catalog': async (refresh): Promise<CloudCatalog> => {
      const r = await catalog.get(refresh)
      lastError = r.error
      if (!r.error && (await store.refreshInfo(r.catalog.models))) deps.onModels()
      push()
      return r.catalog
    },
    'cloud:addModel': async (raw) => {
      const slug = slugOf(raw)
      let r = await catalog.get(false)
      let info = r.catalog.models.find((m) => m.slug === slug)
      if (!info) {
        r = await catalog.get(true)
        info = r.catalog.models.find((m) => m.slug === slug)
      }
      if (!info) throw new Error(`${slug} is not in OpenRouter's list of models this app can use.${r.error ? ' ' + r.error : ''}`)
      await store.add(info)
      deps.onModels()
      return push()
    },
    'cloud:updateModel': async (slug, patch: CloudModelPatch) => {
      await store.update(slugOf(slug), patchOf(patch))
      deps.onModels()
      return push()
    },
    'cloud:removeModel': async (raw) => {
      const slug = slugOf(raw)
      const id = cloudModelId(slug)
      if (deps.usersOf(id) > 0) throw new Error('This model is answering right now. Stop that first, then remove it.')
      await deps.eject(id)
      await store.remove(slug)
      await deps.forgetDefaults(id)
      deps.onModels()
      return push()
    },
    'cloud:setOptions': async (patch: Partial<CloudOptions>) => {
      await store.setOptions(patchOf(patch))
      return push()
    }
  }

  const service: CloudServiceHandle = {
    models: (): ModelEntry[] => store.list().map(toModelEntry),
    hasKey: () => keys.get() !== null,
    allowedForApps: (modelId) => !!defOf(modelId)?.allowApps && keys.get() !== null,
    check(modelId) {
      const d = defOf(modelId)
      if (!d) throw new Error('That OpenRouter model is not in your list.')
      if (!keys.get()) throw new Error('No OpenRouter key is saved. Add one on the OpenRouter page first.')
    },
    props(modelId) {
      const d = defOf(modelId)
      if (!d) throw new Error('That OpenRouter model is not in your list.')
      return { contextLength: d.contextLimit, hasVision: d.info.vision, free: d.info.free }
    },
    fetch: (modelId: string, path: string, init: CloudFetchInit, by: CloudCaller) => gateway.fetch(modelId, path, init, by),
    handlers,
    async load() {
      await keys.load()
      try {
        await store.load()
      } catch (err) {
        lastError = `The model list could not be read: ${errText(err)}`
      }
      await ledger.load()
      await catalog.cached()
      await gateway.settleLeftovers()
      push()
    },
    dispose() {
      gateway.dispose()
      if (timer) clearTimeout(timer)
      timer = undefined
    },
    scrub: errText,
    settled: () => gateway.settled()
  }
  return service
}
