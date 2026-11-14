"""Writes three made-up chats into a RulZ LLM Suite profile, for the Chat screenshots.

    python make_demo_llm_chats.py R:\\llm-profile

They go in <profile>/conversations/ (the folder the app reads, see src/main/chat/conversations.ts). The text is
written for the screenshot; no model produced it and none is loaded to show it.
"""
import json
import sys
import time
from pathlib import Path

SAMPLING = {"temperature": 0.7, "topK": 40, "topP": 0.95, "minP": 0.05, "repeatPenalty": 1.1, "presencePenalty": 0,
            "frequencyPenalty": 0, "maxTokens": None, "stop": [], "seed": None, "jsonSchema": "", "thinking": "auto",
            "reasoningEffort": "default", "thinkingBudget": None, "thinkingBudgetMessage": ""}

FIB_ANSWER = r"""Here's a quick comparison of the three usual ways:

| Method | Time | Memory |
| --- | --- | --- |
| Recursive | $O(2^n)$ | $O(n)$ |
| Memoized | $O(n)$ | $O(n)$ |
| Iterative | $O(n)$ | $O(1)$ |

The closed form uses the golden ratio $\varphi = \frac{1+\sqrt{5}}{2}$:

$$F_n = \frac{\varphi^n - (1-\varphi)^n}{\sqrt{5}}$$

An iterative version in Python:

```python
def fib(n: int) -> int:
    # Keep only the last two values.
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

print([fib(i) for i in range(10)])
```

**Tip:** for very large `n`, use fast doubling, which needs only $O(\log n)$ steps."""


def chat(cid, title, updated, messages, model="Qwen3-14B-Q5_K_M"):
    return {"id": cid, "title": title, "folder": None, "pinned": False, "createdAt": updated - 60_000,
            "updatedAt": updated, "modelId": None, "systemPrompt": "", "sampling": SAMPLING, "presetId": None,
            "messages": messages}


def msg(mid, role, content, at, **extra):
    return {"id": mid, "role": role, "content": content, "createdAt": at, **extra}


def main() -> None:
    out = Path(sys.argv[1]) / "conversations"
    out.mkdir(parents=True, exist_ok=True)
    now = int(time.time() * 1000)
    chats = [
        chat("demo-fibonacci", "Fibonacci approaches", now - 30_000, [
            msg("m1", "user", "Compare ways to compute Fibonacci numbers, with complexity and a Python example.", now - 60_000),
            msg("m2", "assistant", FIB_ANSWER, now - 30_000,
                reasoning="The user wants a comparison: naive recursion, memoization, iteration, maybe the closed form. "
                          "A small table for complexity, then a short iterative example.",
                reasoningMs=2100,
                stats={"tokensPerSecond": 48.6, "predictedTokens": 342, "promptTokens": 41, "ttftMs": 180,
                       "totalMs": 7210, "stopReason": "eos", "modelName": "Qwen3-14B-Q5_K_M"}),
        ]),
        chat("demo-packing", "Trip packing list", now - 2 * 3600_000, [
            msg("p1", "user", "Make a packing list for a three-day hiking trip in autumn.", now - 2 * 3600_000 - 60_000),
            msg("p2", "assistant", "**Clothing:** base layers, a fleece, a rain jacket, hiking socks (3 pairs).\n\n"
                                   "**Gear:** backpack, headlamp, water filter, map and compass, first-aid kit.",
                now - 2 * 3600_000),
        ]),
        chat("demo-regex", "Regex for dates", now - 3 * 3600_000, [
            msg("r1", "user", "A regex that matches dates like 2026-11-13.", now - 3 * 3600_000 - 60_000),
            msg("r2", "assistant", "`^\\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\\d|3[01])$` matches YYYY-MM-DD with a valid "
                                   "month and day range (it doesn't check February 30).", now - 3 * 3600_000),
        ]),
    ]
    for c in chats:
        (out / f"{c['id']}.json").write_text(json.dumps(c, indent=2), encoding="utf-8")
    print(f"wrote {len(chats)} chats to {out}")


if __name__ == "__main__":
    main()
