// Tool definitions sent to the model, argument parsing that forgives common mistakes, and plain
// step titles for the Steps panel.

const summaryParam = {
  type: 'string',
  description:
    'A few plain words for someone who does not code, with no technical terms, saying what this step does for them, e.g. "Creating the main window" or "Installing the pieces your app needs".'
}

const tool = (name: string, description: string, properties: Record<string, unknown>, required: string[]) => ({
  type: 'function' as const,
  function: { name, description, parameters: { type: 'object', properties, required } }
})

export const TOOL_DEFS = [
  tool(
    'list_files',
    'List the files and folders in the project (or a folder inside it) with sizes. Folders end with "/". Installed and generated folders such as node_modules are shown but not opened.',
    {
      path: { type: 'string', description: 'Folder inside the project, e.g. "src". Leave out for the whole project.' },
      depth: { type: 'integer', description: 'How many folder levels to show, 1 to 4 (default 2).' }
    },
    []
  ),
  tool(
    'find_files',
    'Find files or folders in the project by name. The pattern is a wildcard like "*.js" or plain text found anywhere in the name. Case-insensitive.',
    {
      pattern: { type: 'string', description: 'Name pattern, e.g. "*.py", "main", "*.test.js".' },
      path: { type: 'string', description: 'Folder to search in. Leave out for the whole project.' }
    },
    ['pattern']
  ),
  tool(
    'search_text',
    'Search for text inside the project files (case-insensitive, plain text, not a regex). Returns file, line number and the line.',
    {
      query: { type: 'string', description: 'The text to find.' },
      path: { type: 'string', description: 'File or folder to search in. Leave out for the whole project.' },
      glob: { type: 'string', description: 'Only search files matching this, e.g. "*.js" or "src/**/*.ts".' }
    },
    ['query']
  ),
  tool(
    'read_file',
    'Read a text file in the project. Lines are shown with "N| " numbers in front, which are not part of the file. Reads up to 2000 lines at a time.',
    {
      path: { type: 'string', description: 'File path inside the project, e.g. "src/main.js".' },
      start_line: { type: 'integer', description: 'First line to read (default 1).' },
      end_line: { type: 'integer', description: 'Last line to read.' }
    },
    ['path']
  ),
  tool(
    'write_file',
    'Create a file or replace a whole file with new content. Folders are created as needed. For small changes to an existing file, use edit_file instead.',
    {
      path: { type: 'string', description: 'File path inside the project, e.g. "index.html" or "src/app.py".' },
      content: { type: 'string', description: 'The complete text of the file.' },
      summary: summaryParam
    },
    ['path', 'content', 'summary']
  ),
  tool(
    'edit_file',
    'Change part of an existing file: replaces old_text with new_text. old_text must match the file exactly (spaces included, line numbers excluded) and appear only once; include a few surrounding lines to make it unique.',
    {
      path: { type: 'string', description: 'File path inside the project.' },
      old_text: { type: 'string', description: 'The exact text to replace, copied from read_file without the "N| " prefixes.' },
      new_text: { type: 'string', description: 'The text to put in its place.' },
      summary: summaryParam
    },
    ['path', 'old_text', 'new_text', 'summary']
  ),
  tool(
    'delete_path',
    'Move a file or folder in the project to the Recycle Bin. The owner is always asked first.',
    {
      path: { type: 'string', description: 'File or folder path inside the project.' },
      summary: summaryParam
    },
    ['path', 'summary']
  ),
  tool(
    'run_command',
    'Run a PowerShell command in the project folder and wait for it to finish, e.g. "npm install", "npm test", "python -m venv .venv". ' +
      'Returns the exit code and output. Do not use it for apps or servers that keep running (use start_background_command). ' +
      'Commands cannot ask questions, so use non-interactive options (for example "-y").',
    {
      command: { type: 'string', description: 'The PowerShell command. Chain steps with ";".' },
      timeout_seconds: { type: 'integer', description: 'Time limit in seconds (default 600).' },
      summary: summaryParam
    },
    ['command', 'summary']
  ),
  tool(
    'start_background_command',
    'Start a command that keeps running (a desktop app, a web server, a watcher) without waiting for it. Returns an id such as "bg1". ' +
      'Then use check_background_command to read its output and see whether it started without errors, and stop_background_command when done.',
    {
      command: { type: 'string', description: 'The PowerShell command, e.g. "npm start".' },
      summary: summaryParam
    },
    ['command', 'summary']
  ),
  tool(
    'check_background_command',
    'Wait a few seconds, then show the new output of a background command and whether it is still running.',
    {
      id: { type: 'string', description: 'The id from start_background_command, e.g. "bg1".' },
      wait_seconds: { type: 'integer', description: 'Seconds to wait first, 0 to 60 (default 5).' }
    },
    ['id']
  ),
  tool(
    'stop_background_command',
    'Stop a background command (and everything it started).',
    { id: { type: 'string', description: 'The id, e.g. "bg1".' } },
    ['id']
  ),
  tool(
    'ask_user',
    'Ask the owner a question and wait for the answer. Only for real choices that change what you build (for example two very different designs); otherwise pick a sensible default yourself.',
    { question: { type: 'string', description: 'The question, in plain words.' } },
    ['question']
  )
]

export const TOOL_NAMES = new Set(TOOL_DEFS.map((t) => t.function.name))

const INTEGER_ARGS = new Set(['depth', 'start_line', 'end_line', 'timeout_seconds', 'wait_seconds'])

/**
 * Parses tool-call arguments. Accepts strict JSON, then Python-style dicts; coerces numbers given
 * as text and joins lists given where a single path or text is expected.
 */
export function parseToolArgs(raw: string): { args: Record<string, unknown> } | { error: string } {
  const text = (raw ?? '').trim()
  let parsed: unknown = {}
  if (text) {
    try {
      parsed = JSON.parse(text)
    } catch {
      try {
        parsed = JSON.parse(pythonToJson(text))
      } catch {
        return { error: `The arguments were not valid JSON: ${text.slice(0, 200)}` }
      }
    }
  }
  if (typeof parsed === 'string') {
    try {
      parsed = JSON.parse(parsed)
    } catch {
      return { error: 'The arguments must be a JSON object, like {"path": "index.html"}.' }
    }
  }
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    return { error: 'The arguments must be a JSON object, like {"path": "index.html"}.' }
  }
  const args: Record<string, unknown> = {}
  for (const [k, v] of Object.entries(parsed as Record<string, unknown>)) {
    if (INTEGER_ARGS.has(k) && typeof v === 'string' && v.trim() !== '' && Number.isFinite(Number(v))) args[k] = Number(v)
    else if (Array.isArray(v) && ['path', 'command', 'pattern', 'query', 'question'].includes(k)) {
      args[k] = k === 'command' ? v.map(String).join('; ') : String(v[0] ?? '')
    } else args[k] = v
  }
  return { args }
}

/** Converts a Python-style literal ({'a': True}) to JSON, leaving quoted text alone. */
function pythonToJson(text: string): string {
  let out = ''
  let i = 0
  while (i < text.length) {
    const ch = text[i]
    if (ch === "'" || ch === '"') {
      let j = i + 1
      let body = ''
      while (j < text.length && text[j] !== ch) {
        if (text[j] === '\\' && j + 1 < text.length) {
          body += text[j] + text[j + 1]
          j += 2
        } else body += text[j++]
      }
      const unescaped = ch === "'" ? body.replace(/\\'/g, "'").replace(/"/g, '\\"') : body
      out += `"${unescaped}"`
      i = j + 1
      continue
    }
    const word = text.slice(i).match(/^(True|False|None)\b/)
    if (word) {
      out += word[1] === 'True' ? 'true' : word[1] === 'False' ? 'false' : 'null'
      i += word[1].length
      continue
    }
    out += ch
    i++
  }
  return out
}

const base = (p: unknown): string => (typeof p === 'string' && p ? (p.split(/[\\/]/).filter(Boolean).pop() ?? p) : 'the project')

/** Plain description of a command when the model gave no summary. */
export function describeCommand(command: string): string {
  const c = command.toLowerCase()
  if (/\b(npm|pnpm|yarn)\s+(i|install|ci|add)\b|\bpip3?\s+install\b|-m\s+pip\s+install/.test(c)) return 'Installing the pieces your app needs'
  if (/\b-m\s+venv\b/.test(c)) return 'Setting up a private Python folder for this app'
  if (/\b(npm|pnpm|yarn)\s+(run\s+)?test\b|\bpytest\b|\bvitest\b|\bjest\b/.test(c)) return 'Running the tests'
  if (/\b(npm|pnpm|yarn)\s+(run\s+)?build\b/.test(c)) return 'Building the app'
  if (/\b(npm|pnpm|yarn)\s+(start|run\s+(dev|start))\b|\belectron\s/.test(c)) return 'Starting your app'
  const py = c.match(/\bpython3?(?:\.exe)?\s+([\w./\\-]+\.py)/)
  if (py) return `Running ${base(py[1])}`
  const node = c.match(/\bnode\s+([\w./\\-]+\.m?js)/)
  if (node) return `Running ${base(node[1])}`
  if (/\bgit\s/.test(c)) return 'Working with version history'
  if (/^\s*(mkdir|new-item)\b/.test(c)) return 'Creating a folder'
  if (/\b--version\b|\bget-command\b|\bwhere(\.exe)?\s/.test(c)) return 'Checking what is installed'
  return 'Running a command'
}

/** Plain-language title for a step. */
export function describeStep(name: string, args: Record<string, unknown>): string {
  const summary = typeof args.summary === 'string' ? args.summary.trim().replace(/\.$/, '') : ''
  if (summary && summary.length <= 120) return summary
  switch (name) {
    case 'list_files':
      return args.path ? `Looking inside ${base(args.path)}` : 'Looking at the project files'
    case 'find_files':
      return `Looking for ${String(args.pattern ?? 'files')}`
    case 'search_text':
      return `Searching for "${String(args.query ?? '').slice(0, 40)}"`
    case 'read_file':
      return `Reading ${base(args.path)}`
    case 'write_file':
      return `Writing ${base(args.path)}`
    case 'edit_file':
      return `Changing ${base(args.path)}`
    case 'delete_path':
      return `Deleting ${base(args.path)}`
    case 'run_command':
      return describeCommand(String(args.command ?? ''))
    case 'start_background_command':
      return describeCommand(String(args.command ?? '')).replace(/^Running a command$/, 'Starting a program')
    case 'check_background_command':
      return 'Checking how it is going'
    case 'stop_background_command':
      return 'Stopping the program'
    case 'ask_user':
      return 'Asking you a question'
    default:
      return name
  }
}
