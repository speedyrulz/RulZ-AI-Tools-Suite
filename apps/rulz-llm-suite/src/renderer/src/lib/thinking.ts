/**
 * Whether a reasoning block stays open once the model stops thinking. "Expand reasoning blocks by default"
 * and "Keep reasoning open after the answer" both leave it as it is (so a box the user folded stays folded);
 * otherwise it folds away.
 */
export function openAfterThinking(open: boolean, { defaultOpen, keepOpen }: { defaultOpen: boolean; keepOpen: boolean }): boolean {
  return defaultOpen || keepOpen ? open : false
}
