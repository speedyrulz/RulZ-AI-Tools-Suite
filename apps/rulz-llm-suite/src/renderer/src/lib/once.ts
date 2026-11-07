/**
 * Wraps an async setup so it runs once; later calls share the first call's promise.
 * React's StrictMode runs effects twice in development, which would otherwise register every
 * IPC listener twice (and, for chat events, run each MCP tool call twice).
 */
export function once<T>(fn: () => Promise<T>): () => Promise<T> {
  let started: Promise<T> | null = null
  return () => (started ??= fn())
}
