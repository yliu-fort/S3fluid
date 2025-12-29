## 2024-05-22 - [CLI Output Hygiene]
**Learning:** Simulation scripts often default to spamming stdout with raw numbers (`print(t)`), which obscures actual errors and looks broken.
**Action:** Always wrap long-running loops in a progress bar. Even a simple text-based fallback (using `\r` to overwrite lines) provides vastly better UX than a scrolling wall of numbers.
