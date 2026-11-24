// What the window and the main process agree on for PDFs attached to a chat.

/** Largest PDF that can be attached to a chat (the read_pdf tool takes larger files from disk). */
export const MAX_PDF_ATTACH_BYTES = 50 * 1024 * 1024

/** A PDF by its type or its name (a dropped file often has no type). */
export function isPdfFile(file: { name: string; type: string }): boolean {
  return file.type === 'application/pdf' || /\.pdf$/i.test(file.name)
}

/** The text of an attached PDF, or why there is none. */
export type PdfAttachmentResult =
  | {
      ok: true
      /** Page markers and page text, ready to go into a message. */
      text: string
      pageCount: number
      pagesIncluded: number
      /** The text stops before the end of the PDF (character cap). */
      cut: boolean
    }
  | { ok: false; error: string }
