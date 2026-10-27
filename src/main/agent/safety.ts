// Decides which commands are refused outright, which always need the owner's OK, and which follow
// the approval level. Regexes are deliberately broad: a needless question is cheap, a missed one isn't.
import { resolve } from 'node:path'
import type { ApprovalLevel, RiskKind } from '@shared/types'
import { isInside } from './pathGuard'

export type CommandVerdict =
  | { verdict: 'refuse'; reason: string }
  | { verdict: 'ask'; kind: RiskKind; reason: string }
  | { verdict: 'ok' }

interface Rule {
  re: RegExp
  reason: string
}

const REFUSE: Rule[] = [
  {
    re: /\bformat(\.com)?\s+[a-z]:|\b(format-volume|clear-disk|initialize-disk|remove-partition|new-partition|resize-partition|set-partition|diskpart|mkfs)\b|\bcipher(\.exe)?\s+\/w\b|\b(vssadmin|wbadmin)(\.exe)?\s+delete\b|\b(bcdedit|bootrec|bcdboot)\b/i,
    reason: 'It would format or repartition a disk, which can erase everything on it.'
  },
  {
    re: /\breg(\.exe)?\s+(add|delete|import|load|unload|restore|copy)\b|\bregedit\b|\b(set|new|remove|rename|clear)-item(property)?\b[^\n]*\b(hk(lm|cu|cr|u|cc)\b|registry::)|\[microsoft\.win32\.registry/i,
    reason: 'It would edit the Windows registry, which can break Windows or other programs.'
  },
  {
    re: /\b(set|add|remove)-mppreference\b|\bnetsh(\.exe)?\s+(adv)?firewall\b[^\n]*\b(off|disable)|\bset-netfirewallprofile\b[^\n]*-enabled\s+(\$?false|0)|\b(sc(\.exe)?\s+(stop|config|delete)|stop-service|set-service)\b[^\n]*\b(windefend|wscsvc|mpssvc|sense|wdnissvc|securityhealthservice)\b/i,
    reason: 'It would turn off security protection on this computer.'
  },
  {
    re: /(^|[\s;&|(])(shutdown(\.exe)?|restart-computer|stop-computer|logoff(\.exe)?)(?=$|[\s;&|)])/i,
    reason: 'It would shut down, restart or sign out of the computer.'
  },
  {
    re: /\b(net(\.exe)?\s+(user|localgroup)\b[^\n]*\/(add|delete)|new-localuser|remove-localuser|add-localgroupmember)\b/i,
    reason: 'It would change the user accounts on this computer.'
  },
  {
    re: /\b(iwr|irm|curl|wget|invoke-webrequest|invoke-restmethod)\b[^\n]*\|\s*(iex|invoke-expression)\b|\b(iex|invoke-expression)\b[^\n]*\b(downloadstring|iwr|irm|invoke-webrequest|invoke-restmethod)\b/i,
    reason: 'It would download a script from the internet and run it without anyone checking it.'
  }
]

const ASK: (Rule & { kind: RiskKind })[] = [
  {
    kind: 'admin',
    re: /-verb\s+['"]?runas\b|\brunas(\.exe)?\s|\b(g?sudo)\b/i,
    reason: 'This asks for administrator rights.'
  },
  {
    kind: 'install-global',
    re: /\b(npm|pnpm)(\.cmd)?\s+(i|install|add|un|uninstall|remove|rm|link|update|up)\b[^\n]*\s(-g|--global)\b|\byarn\s+global\b|\b(winget|choco|scoop)(\.exe)?\s+(install|upgrade|uninstall|update)\b|\b(install|uninstall|update)-(module|package|script)\b|\bmsiexec\b|\bpipx\s+install\b|\bcargo\s+install\b|\bcorepack\s+enable\b|\.(msi|exe)['"]?\s+\/(s|silent|quiet|q)\b/i,
    reason: 'This installs software for your whole computer, not just this project.'
  },
  {
    kind: 'system-settings',
    re: /\bsetx(\.exe)?\b|\[(system\.)?environment\]::setenvironmentvariable\b[^\n]*['"](machine|user)['"]|\bset-executionpolicy\b|\bnetsh(\.exe)?\b|\bsc(\.exe)?\s+(config|create|delete|stop|start)\b|\b(new|set|remove|stop|start|restart|suspend)-service\b|\bschtasks(\.exe)?\b|\b(register|unregister|set)-scheduledtask\b|\bpowercfg\b|\bset-(date|timezone)\b|\btakeown\b|\bicacls\b|\b(new|set|remove)-netfirewallrule\b|\b(enable|disable)-windowsoptionalfeature\b|\bdism(\.exe)?\b/i,
    reason: 'This changes a Windows setting.'
  },
  {
    kind: 'internet',
    re: /\bgit\s+push\b|\b(npm|yarn|pnpm)(\.cmd)?\s+publish\b|\btwine\s+upload\b|\bgh(\.exe)?\s+\w|\bdocker\s+push\b|\b(vercel|netlify|surge|firebase)(\.cmd)?\s+(deploy|publish|login|--prod)\b|\bnpx\s+(vercel|netlify|surge)\b|\bgh-pages\b|\b(scp|sftp|ftp|rsync|ssh)(\.exe)?\s|\bcurl(\.exe)?\b[^\n]*\s(-d|--data\S*|-F|--form|-T|--upload-file|-X\s*['"]?(POST|PUT|PATCH|DELETE))\b|\binvoke-(webrequest|restmethod)\b[^\n]*-method\s+['"]?(post|put|patch|delete)\b|\bsend-mailmessage\b/i,
    reason: 'This sends something to the internet.'
  },
  {
    kind: 'stop-programs',
    re: /\b(taskkill|stop-process|pkill|killall)\b|(^|[\s;&|])(kill|spps)\s/i,
    reason: 'This closes programs that are running on your computer.'
  }
]

const DELETE =
  /(^|[\s;&|(])(remove-item|ri|rm|rmdir|rd|del|erase|rimraf)(?=$|[\s;&|)])|\bnpx\s+rimraf\b|\bgit\s+clean\b|\bshutil\.rmtree\b|\bos\.(remove|unlink|rmdir)\b|\bfs\.(rm|rmdir|unlink)(sync)?\b|\[(system\.)?io\.(file|directory)\]::delete\b/i

const ENV_PATH = /\$env:(userprofile|appdata|localappdata|programfiles|programdata|windir|systemroot|temp|tmp|homepath|homedrive|public|onedrive)\b|%(userprofile|appdata|localappdata|programfiles|programdata|windir|systemroot|temp|tmp|homepath|public)%|\$home\b|(^|[\s'"(=])~[\\/]/i
const PARENT = /(^|[\s'"(=,;])\.\.([\\/]|$|[\s'";)])/
const QUOTED_PATH = /(["'])([a-z]:[\\/][^"'\n]*|\\\\[^"'\n]+)\1/gi
const ABS_PATH = /(?:^|[\s'"(=,;])([a-z]:[\\/][^\s'"`;|&)<>,]*|\\\\[^\s'"`;|&)<>,]+)/gi
// Drive-only references like "C:" also count.
const BARE_DRIVE = /(?:^|[\s'"(=,;])([a-z]:)(?=$|[\s'"`;|&)])/gi

/** Absolute paths in a command that are not inside the project (plus "..", ~ and system variables). */
export function outsidePaths(command: string, root: string): string[] {
  const found: string[] = []
  // Quoted paths may contain spaces; check them whole, then look at the rest.
  const rest = command.replace(QUOTED_PATH, (_m, _q: string, p: string) => {
    if (!isInside(root, resolve(p))) found.push(p)
    return ' '
  })
  for (const m of rest.matchAll(ABS_PATH)) {
    const p = m[1].replace(/[.:]+$/, '')
    if (!isInside(root, resolve(p))) found.push(p)
  }
  for (const m of command.matchAll(BARE_DRIVE)) found.push(m[1])
  if (PARENT.test(command)) found.push('..')
  const env = command.match(ENV_PATH)
  if (env) found.push(env[0].trim())
  return found
}

export function isDeleteCommand(command: string): boolean {
  return DELETE.test(command)
}

export function classifyCommand(command: string, root: string): CommandVerdict {
  for (const r of REFUSE) if (r.re.test(command)) return { verdict: 'refuse', reason: r.reason }
  const outside = outsidePaths(command, root)
  const deletes = isDeleteCommand(command)
  if (deletes && outside.length) {
    return { verdict: 'refuse', reason: `It would delete something outside the project folder (${outside[0]}).` }
  }
  for (const r of ASK) if (r.re.test(command)) return { verdict: 'ask', kind: r.kind, reason: r.reason }
  if (/\b(py|python3?)(\.exe)?\s+-m\s+pip\s+install\b|(^|[\s;&|])pip3?(\.exe)?\s+install\b/i.test(command) && !/venv/i.test(command)) {
    return {
      verdict: 'ask',
      kind: 'install-global',
      reason: 'This installs Python packages for your whole computer, not just this project.'
    }
  }
  if (outside.length) return { verdict: 'ask', kind: 'outside-project', reason: `This touches something outside the project folder (${outside[0]}).` }
  if (deletes) return { verdict: 'ask', kind: 'delete', reason: 'This deletes files.' }
  return { verdict: 'ok' }
}

export type ActionKind = { type: 'read' } | { type: 'write' } | { type: 'delete' } | { type: 'command'; risk?: RiskKind }

/** The "Allow for this task" key an action falls under. */
export function approvalKey(action: ActionKind): string {
  if (action.type === 'command') return action.risk ? `risk:${action.risk}` : 'commands'
  return action.type === 'delete' ? 'delete' : 'files'
}

/** Whether the owner must be asked, given the approval level and what they allowed for this task. */
export function needsApproval(level: ApprovalLevel, action: ActionKind, allowedForTask: ReadonlySet<string>): boolean {
  if (action.type === 'read') return false
  if (allowedForTask.has(approvalKey(action))) return false
  switch (action.type) {
    case 'delete':
      return true
    case 'write':
      return level === 'careful'
    case 'command':
      return action.risk !== undefined || level !== 'handsoff'
  }
}
