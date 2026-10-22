import * as Collapsible from '@radix-ui/react-collapsible'
import { ChevronRight, Copy, Ellipsis, Folder, FolderInput, FolderPlus, Pencil, Pin, PinOff, Plus, Search, Trash2 } from 'lucide-react'
import { useMemo, useState } from 'react'
import type { ConversationSummary } from '@shared/types'
import { cn, relativeTime } from '../../lib/format'
import { useChat } from '../../stores/chat'
import { Button, Dialog, DropdownMenu, IconButton, Input, Select, type MenuItem } from '../ui'

export function ConversationList() {
  const summaries = useChat((s) => s.summaries)
  const folders = useChat((s) => s.folders)
  const newChat = useChat((s) => s.newChat)
  const addFolder = useChat((s) => s.addFolder)
  const removeFolder = useChat((s) => s.removeFolder)
  const [query, setQuery] = useState('')
  const [folderDialog, setFolderDialog] = useState(false)
  const [folderName, setFolderName] = useState('')

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase()
    return q ? summaries.filter((s) => s.title.toLowerCase().includes(q)) : summaries
  }, [summaries, query])

  const pinned = filtered.filter((s) => s.pinned)
  const loose = filtered.filter((s) => !s.pinned && (!s.folder || !folders.includes(s.folder)))

  return (
    <aside className="flex w-64 shrink-0 flex-col border-r border-border bg-panel">
      <div className="flex items-center gap-1.5 p-3 pb-2">
        <Button variant="primary" className="flex-1" onClick={() => void newChat()}>
          <Plus size={15} /> New chat
        </Button>
        <IconButton label="New folder" onClick={() => setFolderDialog(true)}>
          <FolderPlus size={16} />
        </IconButton>
      </div>
      <div className="px-3 pb-2">
        <div className="relative">
          <Search size={13} className="absolute top-1/2 left-2.5 -translate-y-1/2 text-faint" />
          <Input className="h-8 pl-7.5" placeholder="Search chats" value={query} onChange={(e) => setQuery(e.target.value)} />
        </div>
      </div>
      <div className="min-h-0 flex-1 space-y-3 overflow-y-auto px-2 pb-3">
        {pinned.length > 0 && <Section title="Pinned" items={pinned} folders={folders} />}
        {folders.map((f) => {
          const items = filtered.filter((s) => !s.pinned && s.folder === f)
          if (query && !items.length) return null
          return (
            <FolderSection key={f} name={f} items={items} folders={folders} onRemove={() => void removeFolder(f)} />
          )
        })}
        <Section title={folders.length || pinned.length ? 'Chats' : undefined} items={loose} folders={folders} />
        {filtered.length === 0 && <div className="px-3 py-6 text-center text-xs text-faint">{query ? 'No matching chats' : 'No chats yet'}</div>}
      </div>

      <Dialog
        open={folderDialog}
        onOpenChange={setFolderDialog}
        title="New folder"
        footer={
          <>
            <Button variant="ghost" onClick={() => setFolderDialog(false)}>
              Cancel
            </Button>
            <Button
              variant="primary"
              disabled={!folderName.trim()}
              onClick={() => {
                void addFolder(folderName)
                setFolderName('')
                setFolderDialog(false)
              }}
            >
              Create
            </Button>
          </>
        }
      >
        <div className="p-5">
          <Input autoFocus placeholder="Folder name" value={folderName} onChange={(e) => setFolderName(e.target.value)} />
        </div>
      </Dialog>
    </aside>
  )
}

function Section({ title, items, folders }: { title?: string; items: ConversationSummary[]; folders: string[] }) {
  if (!items.length) return null
  return (
    <div>
      {title && <div className="px-2 pb-1 text-[11px] font-medium tracking-wide text-faint uppercase">{title}</div>}
      <div className="space-y-px">
        {items.map((s) => (
          <ChatRow key={s.id} summary={s} folders={folders} />
        ))}
      </div>
    </div>
  )
}

function FolderSection({ name, items, folders, onRemove }: { name: string; items: ConversationSummary[]; folders: string[]; onRemove: () => void }) {
  const [open, setOpen] = useState(true)
  return (
    <Collapsible.Root open={open} onOpenChange={setOpen}>
      <div className="group flex items-center">
        <Collapsible.Trigger className="flex flex-1 items-center gap-1 rounded-md px-1.5 py-1 text-[12px] font-medium text-muted hover:text-text">
          <ChevronRight size={13} className={open ? 'rotate-90 transition-transform' : 'transition-transform'} />
          <Folder size={13} />
          <span className="truncate">{name}</span>
          <span className="text-faint">{items.length}</span>
        </Collapsible.Trigger>
        <DropdownMenu
          trigger={
            <button className="rounded p-1 text-faint opacity-0 group-hover:opacity-100 hover:bg-hover hover:text-text" aria-label="Folder options">
              <Ellipsis size={14} />
            </button>
          }
          items={[{ label: 'Delete folder (keeps chats)', icon: <Trash2 size={13} />, onSelect: onRemove, danger: true }]}
        />
      </div>
      <Collapsible.Content className="ml-3 space-y-px border-l border-border pl-1">
        {items.map((s) => (
          <ChatRow key={s.id} summary={s} folders={folders} />
        ))}
        {items.length === 0 && <div className="px-2 py-1 text-[11px] text-faint">Empty</div>}
      </Collapsible.Content>
    </Collapsible.Root>
  )
}

function ChatRow({ summary, folders }: { summary: ConversationSummary; folders: string[] }) {
  const currentId = useChat((s) => s.current?.id)
  const generatingId = useChat((s) => s.generation?.conversationId)
  const { open, rename, togglePin, duplicate, remove } = useChat.getState()
  const [renaming, setRenaming] = useState(false)
  const [confirmDelete, setConfirmDelete] = useState(false)
  const [moving, setMoving] = useState(false)
  const active = currentId === summary.id

  const items: (MenuItem | 'sep')[] = [
    { label: 'Rename', icon: <Pencil size={13} />, onSelect: () => setRenaming(true) },
    {
      label: summary.pinned ? 'Unpin' : 'Pin',
      icon: summary.pinned ? <PinOff size={13} /> : <Pin size={13} />,
      onSelect: () => void togglePin(summary.id)
    },
    { label: 'Move to folder…', icon: <FolderInput size={13} />, onSelect: () => setMoving(true), disabled: !folders.length },
    { label: 'Duplicate', icon: <Copy size={13} />, onSelect: () => void duplicate(summary.id) },
    'sep',
    { label: 'Delete', icon: <Trash2 size={13} />, onSelect: () => setConfirmDelete(true), danger: true }
  ]

  return (
    <>
      <div
        className={cn(
          'group flex cursor-default items-center gap-1 rounded-md py-1.5 pr-1 pl-2.5 transition-colors',
          active ? 'bg-hover' : 'hover:bg-hover/60'
        )}
        onClick={() => !renaming && void open(summary.id)}
        onDoubleClick={() => setRenaming(true)}
      >
        {renaming ? (
          <input
            autoFocus
            defaultValue={summary.title}
            className="h-6 min-w-0 flex-1 rounded border border-accent bg-panel-2 px-1.5 text-[13px] focus:outline-none"
            onClick={(e) => e.stopPropagation()}
            onBlur={(e) => {
              void rename(summary.id, e.target.value)
              setRenaming(false)
            }}
            onKeyDown={(e) => {
              if (e.key === 'Enter') e.currentTarget.blur()
              if (e.key === 'Escape') setRenaming(false)
            }}
          />
        ) : (
          <div className="min-w-0 flex-1">
            <div className={cn('truncate text-[13px]', active ? 'text-text' : 'text-text/85')}>{summary.title}</div>
            <div className="flex items-center gap-1 text-[10.5px] text-faint">
              {generatingId === summary.id && <span className="size-1.5 animate-pulse rounded-full bg-accent" />}
              {relativeTime(summary.updatedAt)} · {summary.messageCount} msgs
            </div>
          </div>
        )}
        <div onClick={(e) => e.stopPropagation()}>
          <DropdownMenu
            trigger={
              <button
                className={cn('rounded p-1 text-faint hover:bg-panel hover:text-text', active ? 'opacity-100' : 'opacity-0 group-hover:opacity-100')}
                aria-label="Chat options"
              >
                <Ellipsis size={14} />
              </button>
            }
            items={items}
          />
        </div>
      </div>

      <Dialog
        open={confirmDelete}
        onOpenChange={setConfirmDelete}
        title="Delete chat?"
        description={`"${summary.title}" will be moved to the Recycle Bin.`}
        footer={
          <>
            <Button variant="ghost" onClick={() => setConfirmDelete(false)}>
              Cancel
            </Button>
            <Button
              variant="danger"
              onClick={() => {
                setConfirmDelete(false)
                void remove(summary.id)
              }}
            >
              Delete
            </Button>
          </>
        }
      />
      <MoveDialog open={moving} onOpenChange={setMoving} summary={summary} folders={folders} />
    </>
  )
}

function MoveDialog({
  open,
  onOpenChange,
  summary,
  folders
}: {
  open: boolean
  onOpenChange: (o: boolean) => void
  summary: ConversationSummary
  folders: string[]
}) {
  const moveToFolder = useChat((s) => s.moveToFolder)
  const [target, setTarget] = useState(summary.folder ?? '')
  return (
    <Dialog
      open={open}
      onOpenChange={onOpenChange}
      title="Move to folder"
      footer={
        <>
          <Button variant="ghost" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button
            variant="primary"
            onClick={() => {
              void moveToFolder(summary.id, target || null)
              onOpenChange(false)
            }}
          >
            Move
          </Button>
        </>
      }
    >
      <div className="p-5">
        <Select value={target} onChange={(e) => setTarget(e.target.value)}>
          <option value="">No folder</option>
          {folders.map((f) => (
            <option key={f} value={f}>
              {f}
            </option>
          ))}
        </Select>
      </div>
    </Dialog>
  )
}
