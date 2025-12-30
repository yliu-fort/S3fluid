## 2024-05-23 - [CLI Progress Indicators]
**Learning:** For long-running CLI processes (like simulations), console spam from `print(t)` is a major UX anti-pattern. Users prefer a single line progress bar that shows completion status and ETA.
**Action:** Always wrap main simulation loops in a `progress_bar` utility that handles output cleaning and provides visual feedback. Prioritize libraries like `rich` or `tqdm` but provide a robust fallback for minimal environments.
