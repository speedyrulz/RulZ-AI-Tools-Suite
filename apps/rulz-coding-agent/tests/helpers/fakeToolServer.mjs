// A stand-in tool server (MCP over stdio) for tests: an ordinary tool, one marked destructive,
// and one with a very long description (to test the loaded-tools budget).
import { Server } from '@modelcontextprotocol/sdk/server/index.js'
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js'
import { CallToolRequestSchema, ListToolsRequestSchema } from '@modelcontextprotocol/sdk/types.js'

const text = { type: 'object', properties: { text: { type: 'string' } }, required: ['text'] }
const tools = [
  { name: 'count_letters', description: 'Count the letters in a text. Returns the number.', inputSchema: text },
  {
    name: 'wipe_notes',
    description: 'Wipe the analysis notes. This changes things.',
    inputSchema: { type: 'object', properties: {} },
    annotations: { destructiveHint: true }
  },
  { name: 'huge_tool', description: `A tool with a very long description. ${'Lots of detail. '.repeat(1200)}`, inputSchema: text }
]

const server = new Server({ name: 'fake', version: '1.0.0' }, { capabilities: { tools: {} }, instructions: 'Fake server advice: count before you wipe.' })
server.setRequestHandler(ListToolsRequestSchema, async () => ({ tools }))
server.setRequestHandler(CallToolRequestSchema, async (req) => {
  const args = req.params.arguments ?? {}
  if (req.params.name === 'count_letters') return { content: [{ type: 'text', text: `Letters: ${String(args.text ?? '').replace(/[^a-z]/gi, '').length}` }] }
  if (req.params.name === 'wipe_notes') return { content: [{ type: 'text', text: 'Notes wiped.' }] }
  if (req.params.name === 'huge_tool') return { content: [{ type: 'text', text: 'Huge result.' }] }
  return { content: [{ type: 'text', text: `Unknown tool ${req.params.name}` }], isError: true }
})
await server.connect(new StdioServerTransport())
