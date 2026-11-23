import type { AdventureEdit, StatValue } from '@shared/adventure'

export interface ChoiceRow {
  key: string
  id?: string
  text: string
}

export interface StatRow {
  key: string
  name: string
  value: string
}

export function parseStatValue(value: string): StatValue {
  const trimmed = value.trim()
  if (trimmed !== '') {
    const number = Number(trimmed)
    if (Number.isFinite(number)) return number
  }
  return value
}

export function choicesForEdit(rows: ChoiceRow[]): NonNullable<AdventureEdit['choices']> {
  return rows
    .filter((row) => row.text.trim())
    .map((row) => row.id === undefined ? { text: row.text.trim() } : { id: row.id, text: row.text.trim() })
}
