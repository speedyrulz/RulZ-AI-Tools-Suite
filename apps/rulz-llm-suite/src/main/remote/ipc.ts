// The 'remote:*' channels the rental manager answers, with their input checked. No electron import: index.ts
// registers this table, tests call it directly. 'remote:load', 'remote:estimate', 'remote:saveModel' and
// 'remote:deleteModel' belong to other parts. None of these is registered for the phone or the API server.
import type { RemoteIpcInvoke } from '@shared/remote'
import { cleanAutoStop, type RentalManager } from './rentals'
import { cleanOfferFilter } from './vastApi'

type Handled = 'remote:load' | 'remote:estimate' | 'remote:saveModel' | 'remote:deleteModel'
export type RemoteChannel = Exclude<keyof RemoteIpcInvoke, Handled>

/** The same functions as the contract, each possibly async. */
export type RemoteHandlers = {
  [K in RemoteChannel]: (...args: Parameters<RemoteIpcInvoke[K]>) => ReturnType<RemoteIpcInvoke[K]> | Promise<ReturnType<RemoteIpcInvoke[K]>>
}

const MAX_TEXT = 512

function id(v: unknown, what = 'rental'): number {
  if (typeof v !== 'number' || !Number.isSafeInteger(v) || v <= 0) throw new Error(`That ${what} is not valid.`)
  return v
}
function text(v: unknown, max = MAX_TEXT, allowEmpty = true): string {
  if (typeof v !== 'string' || v.length > max || (!allowEmpty && !v)) throw new Error('That text is not valid.')
  return v
}
function bool(v: unknown): boolean {
  if (typeof v !== 'boolean') throw new Error('Expected yes or no.')
  return v
}
function obj(v: unknown, what: string): Record<string, unknown> {
  if (!v || typeof v !== 'object' || Array.isArray(v)) throw new Error(`The ${what} is not valid.`)
  return v as Record<string, unknown>
}

export function remoteHandlers(m: RentalManager): RemoteHandlers {
  return {
    'remote:state': () => m.state(),
    'remote:refresh': () => m.refresh(),
    // A key is a few dozen characters; a bound keeps a pasted file out.
    'remote:setVastKey': (key) => m.setVastKey(text(key, 400)),
    'remote:setHfToken': (token) => m.setHfToken(text(token, 400)),
    'remote:saveSettings': (s) => {
      const o = obj(s, 'settings')
      return m.saveSettings({ engineTag: text(o.engineTag, 20), image: text(o.image, 200) })
    },
    'remote:addSshKey': () => m.addSshKey(),

    'remote:searchOffers': (filter) => m.searchOffers(cleanOfferFilter(obj(filter, 'filter'))),
    'remote:instances': () => m.instances(),

    'remote:rent': (req) => {
      const r = obj(req, 'rent request')
      if (typeof r.shownDollarsPerHour !== 'number' || !Number.isFinite(r.shownDollarsPerHour) || r.shownDollarsPerHour <= 0 || r.shownDollarsPerHour > 1000) {
        throw new Error('The price shown is not valid.')
      }
      return m.rent({
        offerId: id(r.offerId, 'offer'),
        shownDollarsPerHour: r.shownDollarsPerHour,
        filter: cleanOfferFilter(obj(r.filter, 'filter')),
        autoStop: cleanAutoStop(obj(r.autoStop, 'auto-stop setting'), { enabled: true, idleMinutes: 30, capHours: 12, apiCountsAsUse: false }),
        modelId: text(r.modelId, 80)
      })
    },
    'remote:start': (rentalId) => m.start(id(rentalId)),
    'remote:stop': (rentalId, force) => m.stop(id(rentalId), bool(force)),
    'remote:destroy': (rentalId, label, force) => m.destroy(id(rentalId), text(label, 120, false), bool(force)),

    'remote:checkIntent': (label) => m.checkIntent(text(label, 80, false)),
    'remote:dismissIntent': (label) => m.dismissIntent(text(label, 80, false)),

    'remote:attach': (instanceId) => m.attach(id(instanceId, 'instance')),
    'remote:detach': (rentalId) => m.detach(id(rentalId)),
    'remote:setAutoStop': (rentalId, patch) => {
      const p = obj(patch, 'auto-stop setting')
      // Only the four known keys go on; the manager's check stays the one that decides the ranges.
      const clean: Record<string, unknown> = {}
      for (const k of ['enabled', 'idleMinutes', 'capHours', 'apiCountsAsUse']) if (k in p) clean[k] = p[k]
      return m.setAutoStop(id(rentalId), clean)
    },
    'remote:keepRunning': (rentalId) => m.keepRunning(id(rentalId)),
    'remote:extendCap': (rentalId, hours) => m.extendCap(id(rentalId), hours),

    'remote:setup': (rentalId, modelId) => m.setup(id(rentalId), modelId === null ? null : text(modelId, 80, false)),
    'remote:cancelSetup': (rentalId) => m.cancelSetup(id(rentalId)),
    'remote:removeModelFiles': (rentalId, modelId) => m.removeModelFiles(id(rentalId), text(modelId, 80, false)),
    'remote:log': (rentalId) => m.log(id(rentalId))
  }
}
