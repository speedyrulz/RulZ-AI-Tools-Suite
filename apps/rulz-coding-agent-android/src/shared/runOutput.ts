// Shared by the run panel ("Fix this" button) and its tests.

const ERROR_LINE =
  /\b\w*(?:Error|Exception)\b:|Traceback \(most recent call last\)|\bUncaught\b|Cannot find module|npm (?:ERR!|error)|\bERR_[A-Z_]+\b|\bfatal error\b/

/** True when app output looks like it contains an error message. */
export function outputLooksLikeError(output: string): boolean {
  return ERROR_LINE.test(output)
}
