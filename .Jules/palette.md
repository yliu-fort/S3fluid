## 2024-05-22 - [CLI Progress Indicators]
**Learning:** Users running long simulation scripts (>5s) often assume the process has hung if there is no feedback. A simple iteration counter (spamming new lines) is messy and hard to read.
**Action:** Always wrap long-running loops in a progress bar. Use `rich.progress` if available for a delightful experience, but provide a robust fallback (text-based bar) so the code remains portable without hard dependencies.
