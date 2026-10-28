import { useEffect, useState } from 'react'
import { MAX_INSTRUCTION_CHARS as MAX_CHARS } from '@shared/defaults'
import { useWorkspace } from '../stores/workspace'
import { Button, Dialog, Textarea } from './ui'

/** "Things to always remember": the owner's notes, sent to the AI with every request. */
export function NotesDialog({ open, onOpenChange }: { open: boolean; onOpenChange: (open: boolean) => void }) {
  const memory = useWorkspace((s) => s.memory)
  const setMemory = useWorkspace((s) => s.setMemory)
  const [text, setText] = useState(memory)
  useEffect(() => {
    if (open) setText(memory)
  }, [open, memory])

  return (
    <Dialog
      open={open}
      onOpenChange={onOpenChange}
      title="Things to always remember"
      description="Write anything I should keep in mind for this project, one per line. I read these before every step, even after a long conversation."
      className="w-[min(600px,calc(100vw-48px))]"
      footer={
        <>
          <Button onClick={() => onOpenChange(false)}>Cancel</Button>
          <Button
            variant="primary"
            onClick={async () => {
              if (await setMemory(text)) onOpenChange(false)
            }}
          >
            Save
          </Button>
        </>
      }
    >
      <div className="px-5 py-4">
        <Textarea
          autoFocus
          rows={8}
          maxLength={MAX_CHARS}
          aria-label="Things to always remember"
          placeholder={'For example:\nUse blue, not purple.\nKeep everything in one window.\nThe app is for my grandmother, so use big text.'}
          value={text}
          onChange={(e) => setText(e.target.value)}
        />
        <div className="mt-1 text-right text-[11px] text-faint">
          {text.length.toLocaleString('en-US')} / {MAX_CHARS.toLocaleString('en-US')}
        </div>
      </div>
    </Dialog>
  )
}
