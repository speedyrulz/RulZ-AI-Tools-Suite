import * as RDialog from '@radix-ui/react-dialog'
import { ChevronLeft, ChevronRight, ImageOff, Maximize2, X } from 'lucide-react'
import { useEffect, useState } from 'react'
import { cn } from '../lib/format'

/** A large preview with a thumbnail strip; clicking the preview opens a full-window viewer. */
export function Gallery({ images, name }: { images: string[]; name: string }) {
  const [index, setIndex] = useState(0)
  const [open, setOpen] = useState(false)

  // Keyed on the list's contents: each app list update brings a new array with the same pictures.
  const key = images.join('\n')
  useEffect(() => {
    setIndex(0)
  }, [key])

  if (!images.length) {
    return (
      <div className="flex aspect-[16/5] flex-col items-center justify-center gap-2 rounded-xl border border-dashed border-border bg-panel text-faint">
        <ImageOff className="size-7" />
        <span className="text-sm">No screenshots yet</span>
      </div>
    )
  }

  const step = (d: number): void => setIndex((i) => (i + d + images.length) % images.length)

  return (
    <div>
      <button
        type="button"
        onClick={() => setOpen(true)}
        className="group relative block aspect-[16/9] w-full cursor-zoom-in overflow-hidden rounded-xl border border-border bg-media"
      >
        <img src={images[index]} alt={`${name} screenshot ${index + 1}`} className="h-full w-full object-contain" draggable={false} />
        <span className="absolute right-3 bottom-3 flex items-center gap-1.5 rounded-md bg-black/70 px-2 py-1 text-xs text-white/80 opacity-0 transition-opacity group-hover:opacity-100">
          <Maximize2 className="size-3.5" /> View full size
        </span>
        {images.length > 1 && (
          <>
            <NavArrow side="left" onClick={() => step(-1)} />
            <NavArrow side="right" onClick={() => step(1)} />
          </>
        )}
      </button>

      {images.length > 1 && (
        <div className="mt-3 flex gap-2 overflow-x-auto pb-1">
          {images.map((src, i) => (
            <button
              key={src}
              type="button"
              onClick={() => setIndex(i)}
              className={cn(
                'relative aspect-[16/10] w-32 shrink-0 cursor-pointer overflow-hidden rounded-lg border-2 bg-media transition-all',
                i === index ? 'border-accent' : 'border-transparent opacity-55 hover:opacity-100'
              )}
            >
              <img src={src} alt="" className="h-full w-full object-cover object-top" draggable={false} />
            </button>
          ))}
        </div>
      )}

      <RDialog.Root open={open} onOpenChange={setOpen}>
        <RDialog.Portal>
          <RDialog.Overlay className="fixed inset-0 z-40 bg-black/95" />
          <RDialog.Content
            className="fixed inset-0 z-50 flex items-center justify-center p-12 outline-none"
            onKeyDown={(e) => {
              if (e.key === 'ArrowLeft') step(-1)
              if (e.key === 'ArrowRight') step(1)
            }}
          >
            <RDialog.Title className="sr-only">{name} screenshots</RDialog.Title>
            <RDialog.Description className="sr-only">Use the arrow keys to browse.</RDialog.Description>
            <img src={images[index]} alt="" className="max-h-full max-w-full rounded-lg object-contain" draggable={false} />
            {images.length > 1 && (
              <>
                <NavArrow side="left" onClick={() => step(-1)} large />
                <NavArrow side="right" onClick={() => step(1)} large />
                <div className="absolute bottom-4 left-1/2 -translate-x-1/2 text-xs text-muted tabular-nums">
                  {index + 1} / {images.length}
                </div>
              </>
            )}
            <RDialog.Close className="absolute top-4 right-4 flex size-10 cursor-pointer items-center justify-center rounded-full bg-white/10 text-white hover:bg-accent">
              <X className="size-5" />
            </RDialog.Close>
          </RDialog.Content>
        </RDialog.Portal>
      </RDialog.Root>
    </div>
  )
}

function NavArrow({ side, onClick, large }: { side: 'left' | 'right'; onClick: () => void; large?: boolean }) {
  const Icon = side === 'left' ? ChevronLeft : ChevronRight
  return (
    <span
      role="button"
      aria-label={side === 'left' ? 'Previous' : 'Next'}
      onClick={(e) => {
        e.stopPropagation()
        onClick()
      }}
      className={cn(
        'absolute top-1/2 flex -translate-y-1/2 cursor-pointer items-center justify-center rounded-full bg-black/60 text-white transition-colors hover:bg-accent',
        large ? 'size-12' : 'size-9 opacity-0 group-hover:opacity-100',
        side === 'left' ? 'left-3' : 'right-3'
      )}
    >
      <Icon className={large ? 'size-6' : 'size-5'} />
    </span>
  )
}
