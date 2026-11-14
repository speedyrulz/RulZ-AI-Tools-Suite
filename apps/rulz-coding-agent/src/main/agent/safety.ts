// Decides which commands are refused outright, which always need the owner's OK, and which follow
// the approval level. Regexes are deliberately broad: a needless question is cheap, a missed one isn't.
import { resolve } from 'node:path'
import type { ApprovalLevel, RiskKind } from '@shared/types'
import { isInside } from './pathGuard'

export interface Risk {
  kind: RiskKind
  reason: string
}

export type CommandVerdict =
  | { verdict: 'refuse'; reason: string }
  /** `kind` and `reason` are the main risk; `risks` holds every one the command carries. */
  | { verdict: 'ask'; kind: RiskKind; reason: string; risks: Risk[] }
  | { verdict: 'ok' }

interface Rule {
  re?: RegExp
  test?: (command: string) => boolean
  reason: string
}

const matches = (rule: Pick<Rule, 're' | 'test'>, command: string): boolean => (rule.re ? rule.re.test(command) : false) || (rule.test?.(command) ?? false)

/**
 * Where a command or alias name starts: not inside a word, a path, a variable or an option. Quotes,
 * `=`, `{`, `(` and the like count, so `"shutdown"`, `$null=Restart-Computer` and `%{Stop-Computer}`
 * are all seen.
 */
const START = String.raw`(?:^|[^\w.\-\\/$@])`
/** Where a name ends: not followed by more of a word, a path or an extension. */
const END = String.raw`(?=$|[^\w.\-\\/])`

const REG_PATH = /\b(hk(lm|cu|cr|u|cc)|hkey_(local_machine|current_user|classes_root|users|current_config))\b|registry::/i
const REG_WRITE = new RegExp(String.raw`\b(set|new|remove|rename|clear|copy|move)-item(property)?\b|${START}(sp|ni|rp|rni|ri|rm|del|cpp|cpi|mp|mi|clp|cli)(?=$|\s)`, 'i')
const DOWNLOAD = /\b(iwr|irm|curl|wget|invoke-webrequest|invoke-restmethod|start-bitstransfer|downloadstring|downloaddata|downloadfile)\b|\bcertutil(\.exe)?\b[^\n]*-urlcache\b|\bbitsadmin\b[^\n]*\/transfer\b/i
const RUN_TEXT = /\b(iex|invoke-expression)\b|\[scriptblock\]::create\b/i

/**
 * A command run from a value rather than a name written out: `& $x`, `. $x`, `& "$a$b"`,
 * `Start-Process $x`, in a place where a command starts (so `"*.$ext"` in a filter doesn't count).
 */
const CALL_BY_VALUE =
  /(?:(?:^|[;|{(=,\n\r]|&&|\b(?:return|throw|exit)\s)\s*(?:[&.]\s*(?:\$|"[^"\n]*\$)|(?:start-process|saps|start|invoke-item|ii)\s+(?:-filepath[\s:]+)?(?:\$|"[^"\n]*\$|\())|["']\s*&\s*(?:\$|"[^"\n]*\$))/i
/** The same, with a target put together from several pieces: `& "$a$b"`, `& "$(...)x"`. */
const CALL_BY_PIECES =
  /(?:^|[;|{(=,\n\r]|&&|\b(?:return|throw|exit)\s)\s*(?:[&.]|(?:start-process|saps|start|invoke-item|ii)\s+(?:-filepath[\s:]+)?)\s*"(?:[^"\n]*\$){2}|(?:^|[;|{(=,\n\r]|&&|\b(?:return|throw|exit)\s)\s*[&.]\s*"[^"\n]*\$\(/i
/**
 * The rule before 0.19.0, kept so nothing it caught gets through: a call by value anywhere that
 * isn't inside a word, with a name joined from quoted pieces.
 */
const CALL_ANYWHERE = /(?:^|[^\w.\-\\/$@&])[&.]\s*\$/
const QUOTED_JOIN = /['"]\s*\+\s*['"]|-join\b|\s-f\s|\[char\]/i
/**
 * Ways to put a name together while the command runs: joining or splitting strings, replacing,
 * reversing, characters by number, decoding, and command names looked up by wildcard.
 */
const STRING_BUILD =
  /['"]\s*\+|\+\s*['"]|\+=|\$[\w:]+\s*\+\s*\$|-join\b|\s-f\s|\[char\]|-c?replace\b|-split\b|\.(?:replace|substring|insert|remove|split|reverse)\s*\(|\[(?:system\.)?string\]::|\[(?:system\.)?convert\]::|\.getstring\s*\(|-bxor\b|\[regex\]::|\[array\]::reverse|\b(?:gcm|get-command|gal|get-alias)\b[^\n;|]*[*?]/i

const REFUSE: Rule[] = [
  {
    re: /\bformat(\.com)?\s+[a-z]:|\b(format-volume|clear-disk|initialize-disk|remove-partition|new-partition|resize-partition|set-partition|diskpart|mkfs)\b|\bcipher(\.exe)?\s+\/w\b|\b(vssadmin|wbadmin)(\.exe)?\s+delete\b|\b(bcdedit|bootrec|bcdboot)\b/i,
    reason: 'It would format or repartition a disk, which can erase everything on it.'
  },
  {
    // reg.exe directly, or through Start-Process ('reg' -ArgumentList 'add', ...); registry drives
    // with any command that writes, anywhere in the command (a `cd HKCU:` earlier, a variable, a new line).
    re: new RegExp(String.raw`${START}reg(\.exe)?['"]?(\s+-argumentlist)?[\s,'"]+(add|delete|import|load|unload|restore|copy)\b|\bregedit\b|\[microsoft\.win32\.registry`, 'i'),
    test: (c) => REG_PATH.test(c) && REG_WRITE.test(c),
    reason: 'It would edit the Windows registry, which can break Windows or other programs.'
  },
  {
    re: /\b(set|add|remove)-mppreference\b|\bnetsh(\.exe)?\s+(adv)?firewall\b[^\n]*\b(off|disable)|\bset-netfirewallprofile\b[^\n]*-enabled[\s:]+(\$?false|0)|\b(sc(\.exe)?\s+(stop|config|delete)|stop-service|set-service|spsv)\b[^\n]*\b(windefend|wscsvc|mpssvc|sense|wdnissvc|securityhealthservice)\b/i,
    reason: 'It would turn off security protection on this computer.'
  },
  {
    re: new RegExp(String.raw`${START}(shutdown(\.exe)?|restart-computer|stop-computer|logoff(\.exe)?)${END}`, 'i'),
    reason: 'It would shut down, restart or sign out of the computer.'
  },
  {
    re: /\b(net(\.exe)?\s+(user|localgroup)\b[^\n]*\/(add|delete)|new-localuser|remove-localuser|add-localgroupmember)\b/i,
    reason: 'It would change the user accounts on this computer.'
  },
  {
    // Downloaded text run as code, wherever the two halves are (a pipe, a variable, a new line).
    test: (c) => DOWNLOAD.test(c) && RUN_TEXT.test(c),
    reason: 'It would download a script from the internet and run it without anyone checking it.'
  }
]

/** Code that is put together, decoded or downloaded while it runs: what it does can't be checked first. */
const HIDDEN: Rule[] = [
  { re: RUN_TEXT, reason: 'This runs text as code, so what it does can’t be checked first.' },
  { re: /\$executioncontext\.invokecommand\b/i, reason: 'This runs text as code, so what it does can’t be checked first.' },
  // `&("Set-Mp" + "Preference")`, `.(gcm *-Computer)`: a command whose name is worked out while it
  // runs (`&&` is not a call).
  { re: /(?:^|[^\w.\-\\/$@&])[&.]\s*\(/, reason: 'This runs a command whose name is put together while it runs, so it can’t be checked first.' },
  {
    // `$c = 'Restart-' + 'Computer'; & $c`, `$x = 'shut'; $x += 'down'; & $x`, `& "$a$b"`.
    // Also `return & $c`, `-Command "& $c"` (run by another PowerShell) and lines split by a bare CR.
    test: (c) => CALL_BY_PIECES.test(c) || (CALL_BY_VALUE.test(c) && STRING_BUILD.test(c)) || (CALL_ANYWHERE.test(c) && QUOTED_JOIN.test(c)),
    reason: 'This runs a command whose name is put together while it runs, so it can’t be checked first.'
  },
  {
    // An alias for a name put together while it runs: `Set-Alias x ('Stop-' + 'Computer'); x`.
    test: (c) => /\b(?:set-alias|new-alias|sal|nal)\b/i.test(c) && STRING_BUILD.test(c),
    reason: 'This runs a command whose name is put together while it runs, so it can’t be checked first.'
  },
  {
    // cmd's own variables: `cmd /c "set a=shut&& call %a%down"`.
    test: (c) => /\bcmd(?:\.exe)?\b[^\n]*\s\/[ck]\b/i.test(c) && /\bset\s+"?\w+=/i.test(c) && /%\w+%|!\w+!/.test(c),
    reason: 'This runs a command whose name is put together while it runs, so it can’t be checked first.'
  },
  {
    // (Not .com: web addresses end in it.)
    test: (c) => DOWNLOAD.test(c) && /\.(ps1|psm1|bat|cmd|exe|msi|vbs|wsf|hta|scr)\b/i.test(c),
    reason: 'This downloads a program or script from the internet.'
  }
]

const ASK: (Rule & { kind: RiskKind })[] = [
  {
    kind: 'admin',
    re: /-verb[\s:]+['"]?runas\b|\brunas(\.exe)?\s|\b(g?sudo)\b/i,
    reason: 'This asks for administrator rights.'
  },
  {
    kind: 'install-global',
    re: /\b(npm|pnpm)(\.cmd)?\b[^\n;|&]*\s(-g|--global|--location[=\s]+['"]?global)\b|\byarn\s+global\b|\b(winget|choco|scoop)(\.exe)?\s+(install|upgrade|uninstall|update)\b|\b(install|uninstall|update)-(module|package|script|psresource)\b|\bmsiexec\b|\bpipx\s+install\b|\bcargo\s+install\b|\bcorepack\s+enable\b|\.(msi|exe)['"]?\s+\/(s|silent|quiet|q)\b/i,
    reason: 'This installs software for your whole computer, not just this project.'
  },
  {
    kind: 'system-settings',
    re: /\bsetx(\.exe)?\b|\[(system\.)?environment\]::setenvironmentvariable\b[^\n]*['"](machine|user)['"]|\bset-executionpolicy\b|\bnetsh(\.exe)?\b|\bsc(\.exe)?\s+(config|create|delete|stop|start)\b|\b(new|set|remove|stop|start|restart|suspend)-service\b|\bschtasks(\.exe)?\b|\b(register|unregister|set)-scheduledtask\b|\bpowercfg\b|\bset-(date|timezone)\b|\btakeown\b|\bicacls\b|\b(new|set|remove)-netfirewallrule\b|\b(enable|disable)-windowsoptionalfeature\b|\bdism(\.exe)?\b/i,
    reason: 'This changes a Windows setting.'
  },
  {
    kind: 'internet',
    re: /\bgit(\.exe)?((\s+-c\s+("[^"]*"|'[^']*'|\S+))|(\s+--?[\w-]+(=\S+)?))*\s+push\b|\b(npm|yarn|pnpm)(\.cmd)?\s+publish\b|\btwine\s+upload\b|\bgh(\.exe)?\s+\w|\bdocker\s+push\b|\b(vercel|netlify|surge|firebase)(\.cmd)?\s+(deploy|publish|login|--prod)\b|\bnpx\s+(vercel|netlify|surge)\b|\bgh-pages\b|\b(scp|sftp|ftp|rsync|ssh)(\.exe)?\s|\bcurl(\.exe)?\b[^\n]*\s(-d|--data\S*|-F|--form|-T|--upload-file|-X\s*['"]?(POST|PUT|PATCH|DELETE))\b|\b(iwr|irm|invoke-webrequest|invoke-restmethod)\b[^\n]*(-method[\s:]+['"]?(post|put|patch|delete)\b|-body\b|-infile\b)|\b(iwr|irm|invoke-webrequest|invoke-restmethod|curl|wget)\b[^\n]*https?:\/\/[^\s'"]*\$\(|\bsend-mailmessage\b/i,
    reason: 'This sends something to the internet.'
  },
  {
    kind: 'stop-programs',
    re: /\b(taskkill|stop-process|pkill|killall)\b|(^|[\s;&|])(kill|spps)\s/i,
    reason: 'This closes programs that are running on your computer.'
  }
]

/** pip for the whole computer: a bare `pip`/`python -m pip` (not one inside a virtual environment's folder, and none activated). */
const GLOBAL_PIP = /(?:^|[;&|'"(\n{])\s*(?:(?:py|python3?)(?:\.exe)?\s+(?:-\S+\s+)*?-m\s+pip3?|pip3?(?:\.exe)?)\s+install\b/i

const DELETE = new RegExp(
  String.raw`${START}(remove-item|ri|rm|rmdir|rd|del|erase|rimraf)(?=$|[\s;&|)'"])|\bnpx\s+rimraf\b|\bgit\s+clean\b|\bshutil\.rmtree\b|\bos\.(remove|unlink|rmdir)\b|\bfs\.(rm|rmdir|unlink)(sync)?\b|\[(system\.)?io\.(file|directory)\]::delete\b|\brobocopy\b[^\n]*\s\/(mir|purge)\b`,
  'i'
)

const ENV_PATH =
  /\$\{?env:(userprofile|appdata|localappdata|programfiles|programdata|windir|systemroot|systemdrive|temp|tmp|homepath|homedrive|public|onedrive|allusersprofile)\b|%(userprofile|appdata|localappdata|programfiles|programdata|windir|systemroot|systemdrive|temp|tmp|homepath|homedrive|public|onedrive)%|\$home\b|\$profile\b|\[(system\.)?environment\]::getfolderpath\b|(^|[\s'"(=,;:])~(?=$|[\\/\s'";)])|\[(system\.)?environment\]::getenvironmentvariable\s*\(\s*['"](userprofile|appdata|localappdata|programfiles|programdata|windir|systemroot|systemdrive|temp|tmp|homepath|homedrive|public|onedrive|allusersprofile)['"]|(^|[\s'"(=,;])env:[\\/]?(userprofile|appdata|localappdata|programfiles|programdata|windir|systemroot|systemdrive|temp|tmp|homepath|homedrive|public|onedrive|allusersprofile)\b/i
const QUOTED_PATH = /(["'])([a-z]:[\\/][^"'\n]*|\\\\[^"'\n]+)\1/gi
const ABS_PATH = /(?:^|[\s'"(=,;:>])([a-z]:[\\/][^\s'"`;|&)<>,]*|\\\\[^\s'"`;|&)<>,]+)/gi
/** `\Users\me`: a path from the root of the project's drive. */
const ROOT_PATH = /(?:^|[\s'"(=,;>])(\\(?![\\?])[^\s'"`;|&)<>,]+)/g
/** `//server/share`: a network folder written with forward slashes (a URL has `:` before it, so it doesn't match). */
const SLASH_UNC = /(?:^|[\s'"(=,;>])(\/\/[\w.-]+\/[^\s'"`;|&)<>,]*)/g
/** A path with `..` in it, checked by where it really leads. */
const PARENT_PATH = /(?:^|[\s'"(=,;:>])((?:[^\s'"`;|&()<>,]*[\\/])?\.\.(?:[\\/][^\s'"`;|&()<>,]*)?)(?=$|[\s'"`;|&()<>,])/g
// Drive-only references like "C:" also count.
const BARE_DRIVE = /(?:^|[\s'"(=,;])([a-z]:)(?=$|[\s'"`;|&)])/gi

/**
 * The command as it is checked (it still runs as written): typographic dashes and quotes become
 * plain ones (PowerShell treats them the same), backtick line continuations are joined, and escape
 * characters and empty quotes that can split a name (shut`down, shut^down, shut""down) are dropped.
 */
export function normalizeCommand(command: string): string {
  return command
    .replace(/[‐-―−]/g, '-')
    .replace(/[‘-‛]/g, "'")
    .replace(/[“-‟]/g, '"')
    .replace(/`\r?\n/g, ' ')
    .replace(/[`^]/g, '')
    .replace(/""|''/g, '')
}

/** `powershell -enc <base64>` (PowerShell takes -e, -ec and any start of -EncodedCommand). */
const ENCODED = /\b(?:powershell|pwsh)(?:\.exe)?['"]?[^\n;|]*?\s-(?:e|ec|en\w*)[\s:]+['"]?([A-Za-z0-9+/]{8,}={0,2})(?=$|[\s'"])/gi

/** Scripts passed base64-encoded (`powershell -enc …`), decoded; null for one that doesn't decode to text. */
export function encodedScripts(command: string): (string | null)[] {
  return [...command.matchAll(ENCODED)].map((m) => {
    const text = Buffer.from(m[1], 'base64').toString('utf16le')
    // eslint-disable-next-line no-control-regex
    return text && !/[\u0000-\u0008\u000e-\u001f�]/.test(text) ? text : null
  })
}

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
  for (const m of rest.matchAll(ROOT_PATH)) if (!isInside(root, resolve(root, m[1]))) found.push(m[1])
  for (const m of rest.matchAll(SLASH_UNC)) found.push(m[1])
  for (const m of rest.matchAll(PARENT_PATH)) if (!isInside(root, resolve(root, m[1]))) found.push(m[1])
  for (const m of command.matchAll(BARE_DRIVE)) found.push(m[1])
  const env = command.match(ENV_PATH)
  if (env) found.push(env[0].trim())
  return found
}

/**
 * The check for a start command the model wrote (its RUN: line), which "Run my app" later runs with
 * one click and no question: refused commands, and commands that hide what they run, are not taken.
 * Other risks are only pointed out, since starting an app can need the internet or a delete.
 */
export function classifyStartCommand(command: string, root: string): CommandVerdict {
  // `open:<file>` opens the file with its Windows program, which runs a .bat, .exe or shortcut:
  // only a web page inside the project is opened.
  if (command.startsWith('open:')) {
    const page = command.slice('open:'.length).trim()
    if (/\.html?$/i.test(page) && isInside(root, resolve(root, page))) return { verdict: 'ok' }
    return { verdict: 'refuse', reason: '"Run my app" only opens a web page (an .html file) that is in the project.' }
  }
  const v = classifyCommand(command, root)
  if (v.verdict === 'ask' && v.risks.some((r) => r.kind === 'hidden')) return { verdict: 'refuse', reason: v.risks.find((r) => r.kind === 'hidden')!.reason }
  return v
}

export function classifyCommand(command: string, root: string): CommandVerdict {
  let text = normalizeCommand(command)
  const risks: Risk[] = []
  const add = (kind: RiskKind, reason: string): void => {
    if (!risks.some((r) => r.kind === kind)) risks.push({ kind, reason })
  }
  // A script passed base64-encoded is checked as what it really runs.
  for (let depth = 0; depth < 3; depth++) {
    const inner = encodedScripts(text)
    if (!inner.length) break
    if (inner.includes(null)) add('hidden', 'This runs a hidden (encoded) script, so what it does can’t be checked first.')
    text = [text, ...inner.filter((s): s is string => s !== null).map(normalizeCommand)].join('\n')
  }

  for (const r of REFUSE) if (matches(r, text)) return { verdict: 'refuse', reason: r.reason }
  const outside = outsidePaths(text, root)
  const deletes = DELETE.test(text)
  if (deletes && outside.length) {
    return { verdict: 'refuse', reason: `It would delete something outside the project folder (${outside[0]}).` }
  }
  for (const r of HIDDEN) if (matches(r, text)) add('hidden', r.reason)
  for (const r of ASK) if (matches(r, text)) add(r.kind, r.reason)
  if (GLOBAL_PIP.test(text) && !/\bactivate(\.ps1|\.bat)?\b/i.test(text)) {
    add('install-global', 'This installs Python packages for your whole computer, not just this project.')
  }
  if (outside.length) add('outside-project', `This touches something outside the project folder (${outside[0]}).`)
  if (deletes) add('delete', 'This deletes files.')
  if (!risks.length) return { verdict: 'ok' }
  return { verdict: 'ask', kind: risks[0].kind, reason: risks[0].reason, risks }
}

export type ActionKind =
  | { type: 'read' }
  | { type: 'write' }
  | { type: 'delete' }
  | { type: 'command'; risk?: RiskKind }
  /** A tool from a tool server; `destructive` when the server says it may change things or run programs. */
  | { type: 'tool'; server: string; tool: string; destructive: boolean }

/**
 * The "Allow for this task" key an action falls under. A server's ordinary tools share one key, so
 * one OK covers a whole analysis; each destructive tool has its own.
 */
export function approvalKey(action: ActionKind): string {
  if (action.type === 'command') return action.risk ? `risk:${action.risk}` : 'commands'
  if (action.type === 'tool') return action.destructive ? `tool:${action.server}:${action.tool}` : `tool:${action.server}`
  return action.type === 'delete' ? 'delete' : 'files'
}

/** Whether the owner must be asked, given the approval level and what they allowed for this task. */
export function needsApproval(level: ApprovalLevel, action: ActionKind, allowedForTask: ReadonlySet<string>): boolean {
  if (action.type === 'read') return false
  // Hidden code can be anything, so one OK never covers the next one.
  if (action.type === 'command' && action.risk === 'hidden') return true
  if (allowedForTask.has(approvalKey(action))) return false
  switch (action.type) {
    case 'delete':
      return true
    case 'write':
      return level === 'careful'
    case 'command':
      return action.risk !== undefined || level !== 'handsoff'
    case 'tool':
      return action.destructive || level !== 'handsoff'
  }
}
