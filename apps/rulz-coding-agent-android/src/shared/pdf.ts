export const MAX_PDF_ATTACH_BYTES = 50 * 1024 * 1024

export interface PdfAttachment {
  id: string
  name: string
  pages: number
  bytes: number
}

/** At most this many PDFs go with one message. */
export const MAX_PDFS_PER_MESSAGE = 10
/** The system prompt lists this many of the project's attached PDFs (the newest). */
export const MAX_PDFS_IN_PROMPT = 20
/** An attachment id: 16 lowercase hex digits and nothing else. */
export const PDF_ID = /^[0-9a-f]{16}$/

/** What the conversation shows for a message of PDFs without text. */
export const attachedLabel = (count: number): string => `Attached ${count} PDF${count === 1 ? '' : 's'}`
